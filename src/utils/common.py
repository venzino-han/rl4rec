import json
import re
import pandas as pd
from collections import defaultdict
from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer
import numpy as np


# Initialize tokenizer with proper padding
def initialize_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left",)
    # tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
    return tokenizer

def calculate_ndcg(hit_list, k):
    """
    Calculate the NDCG@k metric.
    """
    true_relevance = np.zeros(k)
    true_relevance[0] = 1  # Only the target is considered relevant
    ideal_dcg = np.sum(true_relevance / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(np.array(hit_list[:k]) / np.log2(np.arange(2, k + 2)))
    return dcg / ideal_dcg


def calculate_hit_rate(hit_list, k):
    """
    Calculate the Hit@k metric.
    """
    return int(1 in hit_list[:k])


def get_vllm_model(model_name, num_gpus=1, max_model_len=1024*9):
    llm = LLM(
        model=model_name, 
        tensor_parallel_size=num_gpus,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        gpu_memory_utilization = 0.95,
        )
    return llm

def get_sampling_params(args):
    sampling_params = SamplingParams(
        temperature=args.temperature,
        min_tokens=args.max_tokens//8,
        max_tokens=args.max_tokens,
    )
    return sampling_params

def generate_response_with_vllm(llm, prompts, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    outputs = [response.outputs[0].text for response in outputs]
    return outputs

def parse_number_list(number_string):
    numbers = re.findall(r'\d+', number_string)
    return [int(num) for num in numbers]

def parse_number_format(number_string):
    numbers = re.findall(r'\[\d+\]', number_string)
    return numbers

# def check_label_ratio(inference_results, labels):
#     label_count = sum(1 for result, label in zip(inference_results, labels) if label in result or result in label)
#     return round(label_count / len(labels), 4)

import numpy as np

def find_index_of_label_in_results(inference_title_list, target_title):
    for i, title in enumerate(inference_title_list):
        if target_title in title:
            return i
    return -1

def check_label_ratio(inference_results, labels, total=-1):
    """
    Calculate hit@5, hit@10, NDCG@5, and NDCG@10 for inference results.

    Args:
        inference_results (list of lists): Predicted rankings for each instance.
        labels (list): Ground truth labels for each instance.

    Returns:
        dict: A dictionary containing hit@5, hit@10, NDCG@5, and NDCG@10 scores.
    """
    hit5, hit10 = 0, 0
    ndcg5, ndcg10 = 0.0, 0.0

    for results, label in zip(inference_results, labels):
        results = results.replace("\n\n", "\n")
        results = results.replace(", ", "\n")
        results = results.split("\n")
        results = results.replace("\n\n", "\n")
        results = results.replace(", ", "\n")
        results = results.split("\n")

        # Check hit@5 and hit@10
        if label in " ".join(results[:5]):
            hit5 += 1
        if label in " ".join(results[:10]):
            hit10 += 1

        # Calculate NDCG@5
        rank = find_index_of_label_in_results(results[:5], label)
        if rank != -1:
            ndcg5 += 1 / np.log2(rank + 2)

        # Calculate NDCG@10
        rank = find_index_of_label_in_results(results[:10], label)
        if rank != -1:
            ndcg10 += 1 / np.log2(rank + 2)

    if total == -1:
        total_labels = len(labels)
    else:
        total_labels = total
    return (
        round(hit5 / total_labels, 4),
        round(hit10 / total_labels, 4),
        round(ndcg5 / total_labels, 4),
        round(ndcg10 / total_labels, 4),
    )

def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def load_data(args):
    data_name = args.data_name
    user_interaction_data = load_json(
        f"data/{data_name}_user_interaction_dict.json"
    )
    item_interaction_data = load_json(
        f"data/{data_name}_item_interaction_dict.json"
    )
    print("Users ", len(user_interaction_data))
    print("Items ", len(item_interaction_data))

    user_interaction_data = {int(k): v for k, v in user_interaction_data.items()}
    item_interaction_data = {int(k): v for k, v in item_interaction_data.items()}

    user_id_mapping = load_json(f"data/{data_name}_user_old2new_id_dict.json")
    item_id_mapping = load_json(f"data/{data_name}_item_old2new_id_dict.json")
    user_id_mapping = {v: k for k, v in user_id_mapping.items()}
    item_id_mapping = {v: k for k, v in item_id_mapping.items()}

    return (
        user_interaction_data,
        item_interaction_data,
        user_id_mapping,
        item_id_mapping,
    )



def filter_interactions(interactions):
    """
    Filter the interactions list to keep only the interactions up to the most recent high-rating interaction.
    Last interaction rating should be 4 or higher.
    """
    last_high_rating_index = None
    for i in range(len(interactions)-1, -1, -1):
        if interactions[i]['rating'] >= 4:
            last_high_rating_index = i+1
            break

    # Return interactions up to the last high-rating interaction
    if last_high_rating_index is not None:
        return interactions[:last_high_rating_index]
    else:
        return []

from transformers import AutoTokenizer

def get_token_length(text, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    # Load the tokenizer for the specified model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Tokenize the input text
    tokens = tokenizer.encode(text, return_tensors="pt")
    # Return the token length
    return tokens.shape[-1]


def write_temp_text(
    args,
    user_feature_prompt,
    user_feature_text,
    system_prompt,
    prompt,
    pred,
    label,
    total_token_length,
):
    # Writing temporary text to a file with better formatting
    
    with open(f"temp/{args.run_name}_{args.model_name_dir}_{args.data_name}.txt", "a") as f:
        # Section divider for easier readability
        f.write(f"{'-'*50}\n")
        
        # Writing the user feature prompt (this is the text prompt related to user preferences)
        f.write("User Feature Prompt:\n")
        f.write(f"{user_feature_prompt}\n")
        
        # Section divider to separate sections
        f.write(f"{'-'*50}\n")
        
        # Writing the extracted user feature text (summary of user preferences, traits, or attributes)
        f.write("User Feature Text:\n")
        f.write(f"{user_feature_text}\n")
        
        # Section divider to separate sections
        f.write(f"{'-'*50}\n")
        
        # Writing the system prompt (this is the system-generated instruction for performing tasks)
        f.write("System Prompt:\n")
        f.write(f"{system_prompt}\n")
        
        # Writing the actual prompt that is used for generating predictions or outputs
        f.write(f"Prompt:\n{prompt}\n\n")
        
        # Section divider to separate sections
        f.write(f"{'-'*50}\n")
        
        # Writing the generated prediction (this is the output from the model based on the prompt)
        f.write("Prediction:\n")
        f.write(f"{pred}\n")
        
        # Section divider to separate sections
        f.write(f"{'-'*50}\n\n")
        
        # Writing the random index (likely used to select random samples or generate random outputs)
        f.write(f"Label: {label}\n\n")
        
        # Writing the total token length (total number of tokens used in this process, helpful for token management)
        f.write(f"Total Token Length: {total_token_length}\n\n")
        
        # Final section divider to close the log entry
        f.write(f"{'-'*50}\n")

def get_item_feature_dict(args):    
    item_feature_file = f"data_processed/{args.data_name}_{args.feature_model_name}_item_summary.json"
    item_features = load_json(item_feature_file)
    item_features = {int(k): v for k, v in item_features.items()}
    return item_features

def get_product_category_and_interaction_type(data_name):
    if data_name == "Luxury_Beauty":
        return "products", "purchases"
    elif data_name == "Video_Games":
        return "games or related products", "plays or purchases"
    else:
        return "products", "purchases"

def load_generated_query_data(data_name, model_name):
    t_df_score_gap = pd.read_csv(f"data_processed/target_only_{data_name}_{model_name}_generated_query_score_gap.csv")
    w_df_score_gap = pd.read_csv(f"data_processed/with_label_{data_name}_{model_name}_generated_query_score_gap.csv")
    wo_df_score_gap = pd.read_csv(f"data_processed/without_label_{data_name}_{model_name}_generated_query_score_gap.csv")
    return t_df_score_gap, w_df_score_gap, wo_df_score_gap

def merge_data(t_df, w_df, wo_df, data_option="all"):
    if data_option == "all":
        merged_df = pd.concat([t_df, w_df, wo_df], axis=0)
    elif data_option == "target":
        merged_df = t_df
    elif data_option == "with_label":
        merged_df = w_df
    elif data_option == "without_label":
        merged_df = wo_df
    #shuffle the data
    merged_df = merged_df.sample(frac=1).reset_index(drop=True)
    return merged_df

def generate_userid_query_sample_dict(df):
    pos_df = df[df["label"] == 1]
    neg_df = df[df["label"] == 0]
    pos_user_ids = pos_df["user_id"].unique()
    neg_user_ids = neg_df["user_id"].unique()
    pos_user_query_sample_dict = defaultdict(list)
    neg_user_query_sample_dict = defaultdict(list)
    for user_id in pos_user_ids:
        pos_user_query_sample_dict[user_id] = pos_df[pos_df["user_id"] == user_id]["query"].tolist()
    for user_id in neg_user_ids:
        neg_user_query_sample_dict[user_id] = neg_df[neg_df["user_id"] == user_id]["query"].tolist()
    return pos_user_query_sample_dict, neg_user_query_sample_dict


import numpy as np
def get_ndcg(predicted_list, label_list, n=10):
    """
    Compute nDCG@N.

    :param predicted_list: List of lists, where each sublist contains predicted items for a user.
    :param label_list: List of ground truth items for each user.
    :param n: Cutoff for calculating nDCG.
    :return: Average nDCG@N over all users.
    """
    ndcg_scores = []
    for predicted, label in zip(predicted_list, label_list):
        dcg = 0.0
        for i, item in enumerate(predicted[:n]):
            if item == label:
                dcg += 1 / np.log2(i + 2)  # log2(i+2) because rank starts at 1
                break
        ndcg_scores.append(dcg)
    return np.mean(ndcg_scores)

def get_hit_rate(predicted_list, label_list, n=10):
    """
    Compute Hit Rate@N.

    :param predicted_list: List of lists, where each sublist contains predicted items for a user.
    :param label_list: List of ground truth items for each user.
    :param n: Cutoff for calculating Hit Rate.
    :return: Average Hit Rate@N over all users.
    """
    hit_scores = []
    for predicted, label in zip(predicted_list, label_list):
        hit = any(item in label for item in predicted[:n])
        hit_scores.append(1 if hit else 0)
    return np.mean(hit_scores)

def calculate_metrics_for_lists(predicted_list, label_list, k_list):
    """
    Calculate Hit@K and NDCG@K for a list of predicted rankings and ground truth labels.

    :param predicted_list: List of lists, where each sublist contains ranked item titles for a user.
    :param label_list: List of ground truth item titles (one per user).
    :param k_list: List of K values for Hit@K and NDCG@K.
    :return: Dictionary with average Hit@K and NDCG@K for each K in k_list.
    """
    metrics = defaultdict(float)
    num_users = len(label_list)

    for predicted, label in zip(predicted_list, label_list):
        for k in k_list:
            # Check if the label is in the top-K predictions
            if label in predicted[:k]:
                metrics[f"Hit@{k}"] += 1
                # Calculate NDCG@K
                rank = predicted.index(label)
                if rank <= k:
                    metrics[f"NDCG@{k}"] += 1 / np.log2(rank + 2)

    # Average the metrics over the number of users
    for k in k_list:
        metrics[f"Hit@{k}"] /= num_users
        metrics[f"NDCG@{k}"] /= num_users

    return metrics

import argparse

def print_arguments(args):
    """
    Print parsed arguments in a formatted way.

    Args:
        args (argparse.Namespace): Parsed arguments.
    """
    for key, value in vars(args).items():
        print(f"{key:<32}: {value}")
