import argparse
import json
import html
from tqdm import tqdm
import torch
import gc
import random

from collections import defaultdict

from transformers import pipeline
from sentence_transformers import SentenceTransformer

from extract_user_emb import extract_embeddings


def process_item_features(args):
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
    meta_file = f"data/{data_name}/meta_text_fix.json"
    output_file = f"data_emb/{data_name}_{args.target}_{model_dir_name}_emb.pt"

    with open(meta_file, 'r') as f:
        item_meta = json.load(f)
    item_meta = {int(k): v for k, v in item_meta.items()}

    # load item_reviews
    item_reviews_file = f"data/{data_name}/item2reviews_with_date.json"
    with open(item_reviews_file, 'r') as f:
        item_reviews = json.load(f)
    item_reviews = {int(k): v for k, v in item_reviews.items()}

    item2user2review_dict = defaultdict(dict)
    for item_id, reviews in item_reviews.items():
        for review in reviews:
            user_id = int(review["user_id"])
            review_text = review["text"]
            item2user2review_dict[item_id][user_id] = review_text

    # load sequential data
    item2sampled_user_dict = defaultdict(list)
    sequential_data_file = f"data/{data_name}/sequential_data.txt"
    with open(sequential_data_file, 'r') as f:
        for line in f.readlines():
            seq = line.split()
            seq = list(map(int, seq))
            user_id = seq[0]
            train_items = seq[1:-3]
            for item_id in train_items:
                item2sampled_user_dict[item_id].append(user_id)
            
    for item_id, user_ids in item2sampled_user_dict.items():
        if len(user_ids) > 8:
            user_ids = random.sample(user_ids, 8)
        item2sampled_user_dict[item_id] = user_ids

    item2reviews = defaultdict(str)
    for item_id, user_ids in item2sampled_user_dict.items():
        reviews = []
        for user_id in user_ids:
            reviews.append(item2user2review_dict[item_id][user_id])
        item2reviews[item_id] = "\n\n".join(reviews)

    FEATURE_BASED_TARGETS = [
            "review_description", 
            "item_feature_direct", 
            "item_feature_direct_128",
            "item_feature_no_review",
        ]

    ## Single target or multiple targets
    if args.target == "item_meta_only":
        # only use item metadata (title, brand, category, description)
        item_text_formated = {}
        for item_id, meta in item_meta.items():
            title = meta.get("title", "None")
            brand = meta.get("brand", "None")
            category = meta.get("category", "None")
            description = meta.get("description", "None")
            
            # limit description to specified token limit (default: 128 words)
            if args.description_word_limit > 0:
                description = " ".join(description.split()[:args.description_word_limit])
            
            # Format metadata into text
            meta_text = f"Title: {title}\n" + \
                        f"Brand: {brand}\n" + \
                        f"Category: {category}\n" + \
                        f"Description: {description}"
            item_text_formated[item_id] = meta_text
    
    elif args.target == "preference":
        # load f"data_processed/item_pref_text_dict_{data_name}.json"
        pref_file = f"data_processed/item_pref_text_dict_{data_name}.json"
        with open(pref_file, 'r') as f:
            item_pref_text_dict = json.load(f)
        item_pref_text_dict = {int(k): v for k, v in item_pref_text_dict.items()}
        item_text_formated = item_pref_text_dict 

    elif args.target == "title_raw_review":
        # only use the title and random reviews
        item_text_formated ={
            item_id: meta.get("title", "None") + "\n" + item2reviews[item_id]
            for item_id, meta in item_meta.items()
        }

    elif args.target in FEATURE_BASED_TARGETS:
        # load f"data_processed/item_summary_text_dict_{data_name}.json"
        summary_file = f"data_processed/{data_name}_{args.model_name}_item_{args.target}.json"
        output_file = f"data_emb/{data_name}_{args.target}_{args.model_name}_{model_dir_name}_emb.pt"

        with open(summary_file, 'r') as f:
            item_summary_text_dict = json.load(f)
        item_summary_text_dict = {int(k): v for k, v in item_summary_text_dict.items()}
        # add title to the summary
        for item_id, meta in item_meta.items():
            if item_id in item_summary_text_dict:
                title = meta.get("title", "None")
                summary = item_summary_text_dict[item_id]
                # item_summary_text_dict[item_id] = title + "\n" + summary
                item_summary_text_dict[item_id] = summary
            else:
                print(f"Item {item_id} not found in item_summary_text_dict.")

        if args.add_item_meta:
            for item_id, meta in item_meta.items():
                if item_id in item_summary_text_dict:
                    title = meta.get("title", "None")
                    brand = meta.get("brand", "None")
                    category = meta.get("category", "None")
                    description = meta.get("description", "None")
                    # limit description to 128 words
                    description = " ".join(description.split()[:128])

                    summary = item_summary_text_dict[item_id]
                    item_summary_text_dict[item_id] = "Here is the meta information of the item:\n" + \
                        f"Title: {title}\n" + \
                        f"Brand: {brand}\n" + \
                        f"Category: {category}\n" + \
                        f"Related User Preference: {summary}\n"
                        # f"Description: {description}\n" + \
                else:
                    print(f"Item {item_id} not found in item_summary_text_dict.")

        item_text_formated = item_summary_text_dict

    elif len(args.target_list) == 1:
        item_text_formated = {item_id: meta.get(args.target, "None") for item_id, meta in item_meta.items()}
    else:
        item_text_formated = {}
        for item_id, meta in item_meta.items():
            item_text = ""
            for target in args.target_list:
                item_text += target + ": " + meta.get(target, "None") + "\n"
            item_text_formated[item_id] = item_text

    # Be aware for the order of the item_text_list
    item_text_list = ["None" for _ in range(len(item_text_formated)+1)]
    for k, v in item_text_formated.items():
        item_text_list[k] = v

    for idx in [30, 40, 50]:
        print(item_text_list[idx])
        print("="*50)

    if not args.use_sentence_transformers:
        from transformers import (
            AutoModel, 
            AutoTokenizer,
            Gemma3Model,
            AutoProcessor,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if "gemma-3" in model_name.lower():
            model = Gemma3Model.from_pretrained(model_name).to(device)
            processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
        elif "llama" in model_name.lower():
            tokenizer.add_special_tokens({'pad_token': "<|reserved_special_token_0|>"})
            model = AutoModel.from_pretrained(model_name).to(device)
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model = AutoModel.from_pretrained(model_name).to(device)
            
        # Tokenize input texts
        if args.target_index == 0:
            # padding to right
            tokenizer.padding_side = "right"
        else:
            # padding to left
            tokenizer.padding_side = "left"

        def get_messages(prompt):
            return [{
                "role": "user", "content": [
                    {"type": "text", "text": prompt},
                ]
            }]
        
        if "gemma-3" in model_name.lower():
            item_text_list = [get_messages(item_text) for item_text in item_text_list]
            item_text_list = processor.apply_chat_template(
                item_text_list,
                padding=False,
                # truncation=True,
                # tokenizer=True,
                return_dict=True,
                return_tensors="pt",
            )
            print(item_text_list[10])

        max_length = 512
        if args.target == "title":
            max_length = 96
        inputs = tokenizer(
            item_text_list,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print(inputs.keys())
        print(inputs["input_ids"][0])
        print(inputs["input_ids"][3])

        # Move inputs to the device

        # Compute embeddings
        with torch.no_grad():
            batch_size = args.batch_size
            total_embeddings = []
            
            for i in tqdm(range(0, len(item_text_list), batch_size), desc="Computing Embeddings"):
                batch_inputs = {k: v[i: i + batch_size] for k, v in inputs.items()}
                # batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                # if "gemma-3" in model_name.lower():            
                #     outputs = model(**batch_inputs, return_dict=True)
                # else:
                batch_embeddings = model(**batch_inputs, return_dict=True)
                batch_embeddings = batch_embeddings.last_hidden_state
                # try:
                #     batch_embeddings = batch_embeddings.last_hidden_state
                # except :
                #     print(batch_embeddings.keys())
                #     batch_embeddings = batch_embeddings[0]
                #     print(batch_embeddings.shape)
                # print(batch_embeddings)
                # print(batch_embeddings.shape)
                batch_embeddings = batch_embeddings[:, args.target_index, :]
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=1, keepdim=True)  # Normalize
                
                total_embeddings.append(batch_embeddings.cpu())
            total_embeddings = torch.cat(total_embeddings, dim=0)
        print(total_embeddings.dtype)
        # Save the embeddings
        torch.save(total_embeddings, output_file)
        print(f"Embeddings saved to {output_file}")
    
    else:
        # if "llama" in model_name.lower():
        #     llm_pipeline = pipeline('feature-extraction', model=args.embedding_model_name, device=0)
        #     llm_pipeline.tokenizer.add_special_tokens({'pad_token': llm_pipeline.tokenizer.eos_token})
        #     llm_pipeline.tokenizer.padding_side = "left"
        #     max_length = args.token_limit
        # else:
        #     llm_pipeline = pipeline('feature-extraction', model=args.embedding_model_name, device=0)
        #     max_length = args.token_limit

        llm_pipeline = SentenceTransformer(
            args.embedding_model_name,
            trust_remote_code=True,
        )
        max_length = args.token_limit
        extract_embeddings(
                args, 
                item_text_list, 
                llm_pipeline, 
                output_file, 
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
    parser.add_argument("--model_name", type=str, default="Llama-3.2-3B-Instruct", help="Name of the SentenceTransformer model.")
    parser.add_argument("--embedding_model_name", type=str, default="mixedbread-ai/mxbai-embed-large-v1", help="Name of the SentenceTransformer model.")
    parser.add_argument("--target", type=str, default="title", help="Target feature to extract embeddings.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for processing item features.")
    parser.add_argument("--target_index", type=int, default=0, help="Target index for extracting embeddings.")
    parser.add_argument("--add_item_meta", action="store_true", help="Add item meta information to the embeddings.")
    parser.add_argument("--token_limit", type=int, default=1024, help="Token limit for the model.")
    parser.add_argument("--use_sentence_transformers", action="store_true", help="Use SentenceTransformers for embedding extraction.")
    parser.add_argument("--description_word_limit", type=int, default=128, help="Word limit for item description when using item_meta_only target.")
    args = parser.parse_args()

    args.target_list = args.target.split("_")

    print(f"Processing {args.data_name} dataset with target: {args.target_list}")

    # Process the dataset
    process_item_features(args)

if __name__ == "__main__":
    main()