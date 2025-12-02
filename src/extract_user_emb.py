import argparse
import json
import html
from tqdm import tqdm
import torch
import gc

from transformers import pipeline, AutoConfig
from sentence_transformers import SentenceTransformer

USER_TEXT_FROM_ITEM_DESCRIPTION_TEMPLATE = """
This user recently purchased:
{item_description}

This user's other recent preferences:
{recent_item_descriptions}
""".strip()

def extract_by_tag(user_text, tag):
    """
    extract by tag 
    <tag> [text] </tag> 
    """
    try:
        user_text = user_text.split(f"<{tag}>")[1].split(f"</{tag}>")[0]
    except:
        pass
    return user_text

def get_item_meta_text(item_meta_dict):
    item_meta_text = ""
    # for key in ["title", "brand", "category", "description"]:
    for key in ["title", "brand", "category"]:
        if key in item_meta_dict:
            item_meta_text += f"{key}: {item_meta_dict[key]}\n"
    return item_meta_text

def get_user_text(args, user_seq_data, user_preference, item_meta, user_to_target_item=None):
    user_inputs = ["Unknown"]
    for user_id in range(1, len(user_seq_data)+1):
        history_iids = user_seq_data[user_id]
        recent_iid = history_iids[-1]
        history_text = user_preference[user_id]
        if args.add_item_meta:
            history_text = "This user recently purchased:\n" + item_meta[recent_iid] + "\n" + \
                "Here is the reasoning and recommendation:\n" + history_text
        elif args.add_target_item_meta:
            target_iid = user_to_target_item[user_id]   
            history_text = "This user will purchase:\n" + item_meta[target_iid] + "\n" + \
                "Here is the reasoning:\n" + history_text
        user_inputs.append(history_text)
    return user_inputs

def get_title_review_text(user_seq_data, item_meta, user2item2_review):
    user_inputs = ["Unknown"]
    for user_id in range(1, len(user_seq_data)+1):
        history_iids = user_seq_data[user_id][-8:]
        history_iids = history_iids[::-1]

        recent_iid = history_iids[0]
        recent_item_meta_text = get_item_meta_text(item_meta[recent_iid])
        recent_review_text = user2item2_review[user_id][recent_iid]

        recent_text = recent_item_meta_text + "\n" + recent_review_text

        for iid in history_iids:
            history_text += item_meta[iid] + "\n"
            history_text += user2item2_review[user_id][iid] + "\n"
        user_inputs.append(history_text)
    return user_inputs


def extract_embeddings(args, input_texts, llm_pipeline, output_file, max_length):
    if "sentence-transformers" in args.embedding_model_name.lower() or args.use_sentence_transformers:
        # split batch_size 10000 
        total_embedding_list = []
        for i in range(0, len(input_texts), args.batch_size*50):
            batch_text = input_texts[i: i + args.batch_size*50]
            embeddings = llm_pipeline.encode(batch_text, batch_size=args.batch_size, show_progress_bar=True, convert_to_tensor=True).cpu()
            total_embedding_list.append(embeddings)
            del embeddings
            gc.collect()
        total_embeddings = torch.cat(total_embedding_list, dim=0)
        torch.save(total_embeddings, output_file)
        del total_embeddings
        gc.collect()
        return
    try:
        hidden_size = llm_pipeline.model.config.hidden_size
    except:
        hidden_size = llm_pipeline.model.config.text_config.hidden_size
    
    total_embeddings = torch.zeros((len(input_texts), hidden_size), dtype=torch.float32)
    for i in tqdm(range(0, len(input_texts), args.batch_size)):
        with torch.no_grad():
            batch_text = input_texts[i: i + args.batch_size]
            batch_embeddings = llm_pipeline(batch_text, batch_size=args.batch_size, return_tensors="pt", max_length=max_length, truncation=True)
            for j, emb in enumerate(batch_embeddings):
                emb = emb[:, args.target_index, :]
                total_embeddings[i+j] = emb.squeeze(0)
                del emb
            gc.collect()

    # check the shape of the embeddings
    assert total_embeddings.size(0) == len(input_texts),\
        f"Total embeddings size: {total_embeddings.size(0)} != {len(input_texts)}"
    torch.save(total_embeddings, output_file)
    return



def user_text_from_item_description(args, user_seq_data, item_meta, item_preference):
    """
    construct user text from item description
    recent item comes first
    """
    user_inputs = ["Unknown"]
    for user_id in range(1, len(user_seq_data)+1):
        history_iids = user_seq_data[user_id][-8:]
        # history_iids = [int(iid) for iid in history_iids]
        last_iid = history_iids[-1]
        recent_iids = history_iids[:-1][::-1]
        last_item_meta = item_meta[last_iid]
        last_item_description = f"{last_item_meta}\n{item_preference[last_iid]}"

        recent_item_meta_list = [item_meta[iid] for iid in recent_iids]
        recent_item_description_list = [item_preference[iid] for iid in recent_iids]
        # recent_item_descriptions = [f"{meta}\n{desc}" for meta, desc in zip(recent_item_meta_list, recent_item_description_list)]
        recent_item_descriptions = [f"{desc}" for desc in recent_item_description_list]
        # recent_item_descriptions = [" ".join(desc.split()[:64]) for meta, desc in zip(recent_item_meta_list, recent_item_description_list)]
        recent_item_descriptions = "\n".join(recent_item_descriptions)
        
        user_text = USER_TEXT_FROM_ITEM_DESCRIPTION_TEMPLATE.format(
            item_description=last_item_description,
            recent_item_descriptions=recent_item_descriptions,
        )
        user_inputs.append(user_text)
    return user_inputs



def process_user_representation(args):
    """
    Processes item features to extract title embeddings for a specified dataset.

    Args:
        data_name (str): Name of the dataset (e.g., Luxury_Beauty, Video_Games).
        model_name (str): Name of the SentenceTransformer model.
        device (str): Device to use ('cuda:1' or 'cpu').

    Returns:
        None: Saves the processed embeddings as a .pt file.
    """
    data_name = args.data_name
    model_name = args.embedding_model_name
    device = args.device

    # File paths
    model_dir_name = model_name.split("/")[-1]

    reviews_file = f"data/{data_name}/user2reviews_with_date.json"
    with open(reviews_file, 'r') as f:
        user2reviews = json.load(f)
    user2reviews = {int(k): v for k, v in user2reviews.items()}

    data_file = f"data/{data_name}/sequential_data.txt"
    pre_train_user_seq_data = {}
    train_user_seq_data = {}
    val_user_seq_data = {}
    test_user_seq_data = {}
    train_user_to_target_item = {}
    val_user_to_target_item = {}
    test_user_to_target_item = {}
    with open(data_file, "r") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            user_id = int(parts[0])
            parts = [int(p) for p in parts[1:]]

            train_user_to_target_item[user_id] = parts[-1]
            val_user_to_target_item[user_id] = parts[-2]
            test_user_to_target_item[user_id] = parts[-3]
            
            pre_train_date = int(user2reviews[user_id][-4]["timestamp"])
            train_date = int(user2reviews[user_id][-3]["timestamp"])
            val_date = int(user2reviews[user_id][-2]["timestamp"])
            test_date = int(user2reviews[user_id][-1]["timestamp"])

            # filter out items that are not in the last 14 days
            pre_train_parts = []
            train_parts = []
            val_parts = []
            test_parts = []
            gap = args.days * 24 * 60 * 60

            pre_train_reviews = user2reviews[user_id][:-4]
            train_reviews = user2reviews[user_id][:-3]
            val_reviews = user2reviews[user_id][:-2]
            test_reviews = user2reviews[user_id][:-1]

            for i in range(len(pre_train_reviews)):
                timestamp = int(pre_train_reviews[i]["timestamp"])
                item_id = int(pre_train_reviews[i]["item_id"])
                if pre_train_date - timestamp < gap:
                    pre_train_parts.append(item_id)
            if len(pre_train_parts) == 0:
                pre_train_parts.append(parts[-5])

            for i in range(len(train_reviews)):
                timestamp = int(train_reviews[i]["timestamp"])
                item_id = int(train_reviews[i]["item_id"])
                if train_date - timestamp < gap:
                    train_parts.append(item_id)
            if len(train_parts) == 0:
                train_parts.append(parts[-4])
            
            for i in range(len(val_reviews)):
                timestamp = int(val_reviews[i]["timestamp"])
                item_id = int(val_reviews[i]["item_id"])
                if val_date - timestamp < gap:
                    val_parts.append(item_id)
            if len(val_parts) == 0:
                val_parts.append(parts[-3])
            
            for i in range(len(test_reviews)):
                timestamp = int(test_reviews[i]["timestamp"])
                item_id = int(test_reviews[i]["item_id"])
                if test_date - timestamp < gap:
                    test_parts.append(item_id)
            if len(test_parts) == 0:
                test_parts.append(parts[-2])

            pre_train_user_seq_data[user_id] = pre_train_parts
            train_user_seq_data[user_id] = train_parts
            val_user_seq_data[user_id] = val_parts
            test_user_seq_data[user_id] = test_parts

    print("Train seq example")
    print(train_user_seq_data[1])
    print(val_user_seq_data[1])
    print(test_user_seq_data[1])
    print(train_user_seq_data[2])
    print(val_user_seq_data[2])
    print(test_user_seq_data[2])

    # load item preference json 
    item_preference_file = f"data_processed/{args.data_name}_{args.item_model_name}_item_{args.item_feature_name}.json"
    with open(item_preference_file, 'r') as f:
        item_preference = json.load(f)
    item_preference = {int(k): v for k, v in item_preference.items()}
    
    # load item meta json
    item_meta_file = f"data/{args.data_name}/meta_text_fix.json"
    with open(item_meta_file, 'r') as f:
        item_meta = json.load(f)
    item_meta = {int(k): get_item_meta_text(v) for k, v in item_meta.items()}


    FEATURE_BASED_TARGETS = [
        "item_description",
        "item_feature_direct",
        "item_feature_no_review",
        "review_description",
        # "user_preference_reasoning",
        # "reasoning4step",
        # "user_preference",
    ]


    if args.target == "item_review":
        # load user2review.json
        user2review_file = f"data/{args.data_name}/user2reviews.json"
        with open(user2review_file, 'r') as f:
            user2review = json.load(f)
        user2review = {int(k): v for k, v in user2review.items()}
        user2item2review = {}
        for user_id, reviews in user2review.items():
            user2item2review[user_id] = {}
            for review in reviews:
                iid = int(review["item_id"])
                user2item2review[user_id][iid]=review["text"]
        # get user text
        train_user_text = get_title_review_text(train_user_seq_data, item_meta, user2item2review)
        val_user_text = get_title_review_text(val_user_seq_data, item_meta, user2item2review)
        test_user_text = get_title_review_text(test_user_seq_data, item_meta, user2item2review)
    elif args.target in FEATURE_BASED_TARGETS:
        # pre_train_user_text = user_text_from_item_description(args, pre_train_user_seq_data, item_meta, item_preference)
        train_user_text = user_text_from_item_description(args, train_user_seq_data, item_meta, item_preference)
        val_user_text = user_text_from_item_description(args, val_user_seq_data, item_meta, item_preference)
        test_user_text = user_text_from_item_description(args, test_user_seq_data, item_meta, item_preference)
    else:
        # load user preference json
        train_user_preference_file = f"data_processed/{args.data_name}_{args.model_name}_train_{args.target}.json"
        valid_user_preference_file = f"data_processed/{args.data_name}_{args.model_name}_valid_{args.target}.json"
        test_user_preference_file = f"data_processed/{args.data_name}_{args.model_name}_test_{args.target}.json"
        with open(train_user_preference_file, 'r') as f:
            train_user_preference = json.load(f)
        train_user_preference = {int(k): v for k, v in train_user_preference.items()}
        with open(valid_user_preference_file, 'r') as f:
            val_user_preference = json.load(f)
        val_user_preference = {int(k): v for k, v in val_user_preference.items()}
        with open(test_user_preference_file, 'r') as f:
            test_user_preference = json.load(f)
        test_user_preference = {int(k): v for k, v in test_user_preference.items()}

        print(train_user_preference_file)

        train_user_text = get_user_text(args, train_user_seq_data, train_user_preference, item_meta, user_to_target_item=train_user_to_target_item)
        val_user_text = get_user_text(args, val_user_seq_data, val_user_preference, item_meta, user_to_target_item=val_user_to_target_item)
        test_user_text = get_user_text(args, test_user_seq_data, test_user_preference, item_meta, user_to_target_item=test_user_to_target_item)

    if args.split_tag:
        train_user_text = [extract_by_tag(text, args.split_tag) for text in train_user_text]
        val_user_text = [extract_by_tag(text, args.split_tag) for text in val_user_text]
        test_user_text = [extract_by_tag(text, args.split_tag) for text in test_user_text]

    for idx in [10, 20, 30+1, 30, 29]:
        print(train_user_text[idx])
        print("="*50)

    from transformers import AutoModel, AutoTokenizer

    if "llama" in model_name.lower():
        llm_pipeline = pipeline('feature-extraction', model=args.embedding_model_name, device=0)
        llm_pipeline.tokenizer.add_special_tokens({'pad_token': llm_pipeline.tokenizer.eos_token})
        llm_pipeline.tokenizer.padding_side = "left"
        max_length = args.token_limit
    elif "sentence-transformers" in model_name.lower() or args.use_sentence_transformers:
        llm_pipeline = SentenceTransformer(
            args.embedding_model_name,
            trust_remote_code=True,
        )
        max_length = args.token_limit
    else:
        llm_pipeline = pipeline('feature-extraction', model=args.embedding_model_name, device=0)
        max_length = args.token_limit
        llm_pipeline.tokenizer.padding_side = "left"
        if "gemma" in model_name.lower():
            config = AutoConfig.from_pretrained(args.embedding_model_name)
            config.text_config.use_cache = False
            llm_pipeline = pipeline(
                    'feature-extraction', 
                    model=args.embedding_model_name, 
                    config=config,
                    device=0,
                )

        # llm_pipeline.tokenizer.padding_side = "left"

    # extract_embeddings(
    #         args, pre_train_user_text, llm_pipeline, 
    #         f"data_emb/{args.run_name}_{args.token_limit}_user_preference_{args.data_name}_{args.model_name}_{args.embedding_model_name_dir}_pre_train_pred_emb.pt", 
    #         max_length,
    #     )

    embedding_file_prefix = f"{args.run_name}_{args.token_limit}_user_preference_{args.data_name}_{args.model_name}_{args.embedding_model_name_dir}"
    if args.split_tag:
        embedding_file_prefix = f"{args.run_name}_{args.token_limit}_user_preference_{args.split_tag}_{args.data_name}_{args.model_name}_{args.embedding_model_name_dir}"

    extract_embeddings(
            args, train_user_text, llm_pipeline, 
            f"data_emb/{embedding_file_prefix}_train_pred_emb.pt", 
            max_length,
        )
    extract_embeddings(
            args, val_user_text, llm_pipeline, 
            f"data_emb/{embedding_file_prefix}_valid_pred_emb.pt", 
            max_length,
        )
    extract_embeddings(
            args, test_user_text, llm_pipeline, 
            f"data_emb/{embedding_file_prefix}_test_pred_emb.pt", 
            max_length,
        )

    
def main():
    """
    Main function to process item features for a given dataset.
    """
    parser = argparse.ArgumentParser(description="Extract item title embeddings for a given dataset.")
    parser.add_argument(
        "--data_name", type=str, required=True, help="Name of the dataset (e.g., Luxury_Beauty, Video_Games)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for processing (default: cuda:1)"
    )
    parser.add_argument("--run_name", type=str, default="pref", help="Name of the dataset to use.")
    parser.add_argument("--model_name", type=str, default="Llama-3.2-3B-Instruct", help="Name of the SentenceTransformer model.")
    parser.add_argument("--item_model_name", type=str, default="gemma-3-4b-it", help="Name of the SentenceTransformer model.")
    parser.add_argument("--embedding_model_name", type=str, default="mixedbread-ai/mxbai-embed-large-v1", help="Name of the SentenceTransformer model.")
    parser.add_argument("--target", type=str, default="title", help="Target feature to extract embeddings.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for processing item features.")
    parser.add_argument("--target_index", type=int, default=0, help="Target index for extracting embeddings.")

    # prompt options
    parser.add_argument("--add_item_meta", action="store_true", help="Add item meta information to the embeddings.")
    parser.add_argument("--add_target_item_meta", action="store_true", help="Add target item meta information to the embeddings.")
    
    parser.add_argument("--split_tag", default=None, help="extract by tag")
    
    # parser.add_argument("--split", type=str, default="train", help="Split to process (train/test).")
    parser.add_argument("--token_limit", type=int, default=1024, help="Token limit for the model.")
    parser.add_argument("--use_sentence_transformers", action="store_true", help="Use SentenceTransformers for embedding extraction.")
    parser.add_argument("--days", type=int, default=14, help="Days to consider for the user preference.")
    parser.add_argument("--item_feature_name", type=str, default="review_description", help="Name of the item feature to use.")

    args = parser.parse_args()

    args.target_list = args.target.split("_")
    args.embedding_model_name_dir = args.embedding_model_name.split("/")[-1]

    print(f"Processing {args.data_name} dataset with target: {args.target_list}")

    # Process the dataset
    process_user_representation(args)

if __name__ == "__main__":
    main()