import torch
import argparse
import gc

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Gemma3TextModel,
    Gemma3Model,
    Gemma3ForCausalLM,
    BitsAndBytesConfig, 
)
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, PeftModel

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import random
from collections import defaultdict
from typing import Callable, List, Dict, Tuple, Any

from utils.common import (
    load_json,
    print_arguments,
    get_vllm_model,
    get_sampling_params,
    generate_response_with_vllm,
    initialize_tokenizer,
    )
from utils.dataset import PromptGenerator
from evaluator import RecommendationEvaluator

# off warning messages
import warnings
warnings.filterwarnings("ignore") 

import argparse

def get_item_meta(args):
    """아이템 메타데이터 로드"""
    input_file = f"data/{args.data_name}/meta_text_fix.json"
    with open(input_file, 'r') as f:
        item_metadata = json.load(f)
    item_metadata = {int(k): v for k, v in item_metadata.items()}
    return item_metadata

def get_uid_to_seq_data(args):
    """시퀀셜 데이터 로드"""
    sequential_file = f"data/{args.data_name}/sequential_data.txt"
    
    train_user_seq_data = {}
    train_pos_item_ids = {}
    train_user_target = {}
    
    val_user_seq_data = {}
    val_pos_item_ids = {}
    val_user_target = {}
    
    test_user_seq_data = {}
    test_pos_item_ids = {}
    test_user_target = {}
    
    with open(sequential_file, "r") as f:
        for line in f:
            parts = [int(p) for p in line.strip().split()]
            user_id = parts[0]
            items = parts[1:]
            
            # train: up to -3
            train_history = items[:-3]
            train_target_item = items[-3]
            train_user_seq_data[user_id] = train_history
            train_pos_item_ids[user_id] = train_history
            train_user_target[user_id] = train_target_item
            
            # valid: up to -2
            val_history = items[:-2]
            val_target_item = items[-2]
            val_user_seq_data[user_id] = val_history
            val_pos_item_ids[user_id] = val_history
            val_user_target[user_id] = val_target_item
            
            # test: up to -1
            test_history = items[:-1]
            test_target_item = items[-1]
            test_user_seq_data[user_id] = test_history
            test_pos_item_ids[user_id] = test_history
            test_user_target[user_id] = test_target_item
    
    return (
        train_user_seq_data, val_user_seq_data, test_user_seq_data,
        train_pos_item_ids, val_pos_item_ids, test_pos_item_ids,
        train_user_target, val_user_target, test_user_target
    )

def load_filtered_users(file_path):
    """Load filtered user IDs from embedding comparison JSON file"""
    if not file_path or not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    filtered_user_ids = set(data['top_user_ids'])
    print(f"Loaded {len(filtered_user_ids)} filtered users from {file_path}")
    print(f"  Threshold similarity: {data.get('threshold_similarity', 'N/A')}")
    print(f"  Mean similarity: {data.get('mean_similarity', 'N/A')}")
    
    return filtered_user_ids

def get_user_text(args, user_seq_data, user_preference, item_meta, 
                  user_to_target_item=None, add_item_meta=False, 
                  add_target_item_meta=False):
    """사용자 타겟 텍스트 생성"""
    user_text_list = []
    
    for user_id in sorted(user_seq_data.keys()):
        if user_id not in user_preference:
            user_text_list.append("")
            continue
            
        text = user_preference[user_id]
        
        # 필요시 타겟 아이템 메타데이터 추가
        if add_target_item_meta and user_to_target_item and user_id in user_to_target_item:
            target_item = user_to_target_item[user_id]
            if target_item in item_meta:
                item_info = item_meta[target_item]
                text += f"\n\nTarget Item:\nTitle: {item_info.get('title', '')}\n"
                text += f"Brand: {item_info.get('brand', '')}\n"
                text += f"Category: {item_info.get('category', '')}"
        
        user_text_list.append(text)
    
    return user_text_list

def get_time_aware_user_history_text(args, split="train", item_meta=None, prefix=""):
    """PromptGenerator를 사용하여 사용자 히스토리 텍스트 생성"""
    # user2reviews 로드
    reviews_file = f"data/{args.data_name}/user2reviews_with_date.json"
    with open(reviews_file, 'r') as f:
        user2reviews = json.load(f)
    user2reviews = {int(k): v for k, v in user2reviews.items()}
    
    # PromptGenerator 생성
    prompt_generator = PromptGenerator(
        item_metadata=item_meta,
        data_name=args.data_name,
        prompt_type='seq_rec',  # 기본 프롬프트 타입
        use_brand=True,
        use_category=True,
        use_description=False,
        use_features=False,
        use_last_item=True,
        use_date=True,
        max_history_len=args.max_history_len,
        history_text_max_length=100,
        use_reviews=False,
        days_filter=args.days,
    )
    
    # 분할 인덱스 결정
    if split == "train":
        index = -3
    elif split == "valid":
        index = -2
    elif split == "test":
        index = -1
    else:
        raise ValueError(f"Invalid split: {split}")
    
    # 사용자별 프롬프트 생성
    uid_to_prompt = {}
    for user_id, reviews in user2reviews.items():
        if len(reviews) < abs(index):
            # 리뷰가 충분하지 않은 경우 스킵
            continue
            
        target_timestamp = int(reviews[index]["timestamp"])
        # 히스토리 아이템 ID 추출
        history_item_ids = [int(review["item_id"]) for review in reviews[:index]]
        
        # 히스토리 길이 제한 적용
        if len(history_item_ids) > args.max_history_len:
            history_item_ids = history_item_ids[-args.max_history_len:]
        
        # PromptGenerator를 사용하여 히스토리 텍스트 생성
        history_text = prompt_generator.generate_prompt(
            item_ids=history_item_ids,
            user_id=user_id,
            target_timestamp=target_timestamp
        )
        
        # prefix 추가
        if prefix:
            history_text = prefix + history_text
        
        uid_to_prompt[user_id] = history_text
    
    return uid_to_prompt

def parse_arguments():
    """
    Parse command-line arguments for inference and training settings.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run inference and generate test results.")

    # fix seed
    parser.add_argument("--seed", type=int, default=22)

    # General settings
    parser.add_argument("--run_name", type=str, default="lora_sft")
    parser.add_argument("--data_name", type=str, default="toys")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--target_model_name", type=str, default="gemma-3-12b-it")
    # parser.add_argument("--pretrained_run_name", type=str, default=None)

    # Training settings
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_input_tokens", type=int, default=1024+512)
    parser.add_argument("--batch_size", type=int, default=16)
    
    parser.add_argument("--num_train_samples", type=int, default=5000)
    parser.add_argument("--num_test_samples", type=int, default=100)

    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--max_output_tokens", type=int, default=1024)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.92)

    # Target settings
    parser.add_argument("--target", type=str, default="reasoning")
    parser.add_argument("--add_item_meta", action="store_true")
    parser.add_argument("--add_target_item_meta", action="store_true")


    # User history settings
    parser.add_argument("--max_history_len", type=int, default=8)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--revearse", action="store_true")

    # Filtered user settings
    parser.add_argument("--use_filtered_users", action="store_true", 
                        help="Use only filtered top users from embedding comparison")
    parser.add_argument("--filtered_user_file", type=str, default=None,
                        help="Path to filtered user JSON file (e.g., top25_target_vs_vanilla_beauty_train.json)")

    # Item meta settings
    parser.add_argument("--item_meta_list_text", type=str, default="title_brand_category")

    # Quantization settings
    parser.add_argument("--quantization_option", type=str, default="None")
    parser.add_argument("--rank_dim", type=int, default=8)

    # Evaluation settings
    parser.add_argument("--run_evaluation", action="store_true",
                        help="Run evaluation after training using RecommendationEvaluator")
    parser.add_argument("--save_responses", action="store_true",
                        help="Save generated responses to JSON files")
    parser.add_argument("--emb_model_name", type=str, default="mixedbread-ai/mxbai-embed-large-v1",
                        help="Embedding model name for evaluation")
    parser.add_argument("--emb_type", type=str, default="title",
                        help="Embedding type (title, description, etc.)")
    parser.add_argument("--eval_emb_max_length", type=int, default=512,
                        help="Max length for embedding computation")
    parser.add_argument("--eval_emb_batch_size", type=int, default=512,
                        help="Batch size for embedding computation")
    parser.add_argument("--eval_samples", type=int, default=100000,
                        help="Maximum number of samples to evaluate")

    args = parser.parse_args()
    args.item_meta_list_text = args.item_meta_list_text.split("_")
    args.history_limit = args.max_history_len


    return args


def get_formatted_prompt_list(args, uid_to_prompt):
    prompt_list = []
    tokenizer = initialize_tokenizer(args.model_name)
    for uid in range(1, len(uid_to_prompt)+1):
        prompt = uid_to_prompt[uid]
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if "Qwen" in args.model_name:
            # don't use thinking
            prompt += "<think> </think>"
        prompt_list.append(prompt)
    return prompt_list


def get_dataset(
        prompt_list: List[str],
        label_text_list: List[str],
    ) -> Dataset:
        
    data_df = pd.DataFrame({
        "prompt": prompt_list,
        "completion": label_text_list,
    })

    for i in [10, 20, 30]:
        print("Prompt "+"-"*30)
        print(prompt_list[i][-300:])
        print("Label "+"-"*30)
        print(label_text_list[i])
    
    return Dataset.from_pandas(data_df)


def get_quantization_config(args):
    print(f"Using {args.quantization_option} quantization")
    
    quantization_config = None
    
    if args.quantization_option == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    elif args.quantization_option == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=False,
            bnb_8bit_quant_type="nf4",
            bnb_8bit_compute_dtype=torch.bfloat16
        )
    
    return quantization_config


def generate_responses(model, tokenizer, args, batch_prompts):
    """
    Generate 20 responses for each prompt in batch_prompts using beam search.
    """
    inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=args.max_input_tokens,
        ).to(args.device)
    input_lengths = inputs["input_ids"].shape[1]

    generated_tokens = model.generate(
        **inputs,
        max_new_tokens=args.max_output_tokens,
        # num_beams=20,  # Apply beam search
        num_return_sequences=1,  # Generate 20 results per prompt
        pad_token_id=tokenizer.pad_token_id,
    )

    # Reshape generated tokens to match the batch size
    batch_size = len(batch_prompts)
    generated_tokens = generated_tokens.view(batch_size, -1)

    # Decode all responses in the batch at once
    decoded_responses = tokenizer.batch_decode(
        generated_tokens[:, input_lengths:],
        skip_special_tokens=True
    )

    return decoded_responses


def generate_lora_responses_with_vllm(
        args, 
        prompt_list,
        temp_dir="temp",
    ):
    """
    LoRA를 사용한 응답 생성 (현재 미사용)
    """
    response_list = []
    
    os.makedirs(temp_dir, exist_ok=True)

    # Initialize VLLM LLM
    llm = LLM(
        model=args.model_name,
        dtype=torch.bfloat16,  # Change dtype if needed
        trust_remote_code=True,
        quantization="bitsandbytes" if args.quantization_option == "4bit" else None,
        load_format="bitsandbytes" if args.quantization_option == "4bit" else "auto",
        enable_lora=True,
        gpu_memory_utilization = args.gpu_memory_utilization,
        max_model_len=args.max_input_tokens,
    )

    # Define sampling parameters
    sampling_params = SamplingParams(
        n=1,
        temperature=0.1,
        max_tokens=args.max_output_tokens,
        stop=["<|eot_id|>", "<|reserved_special_token_0|>", "<eos>"]
    )

    outputs = llm.generate(
        prompt_list,
        sampling_params,
        lora_request=LoRARequest(
            lora_name="all-linear",
            lora_int_id=1,
            lora_local_path=f"models/{args.run_name}_{args.data_name}_{args.model_name_dir}",
            base_model_name=args.model_name,
        )
    )
    
    for output in outputs:
        response_list.append(output.outputs[0].text)

    return pd.DataFrame({
        "prompt": prompt_list,
        "response": response_list,
        "label_text": label_text_list,
    })

def generate_responses_with_vllm(
        args, 
        prompt_list,
    ):
    # Initialize VLLM LLM
    llm = LLM(
        model=f"models/{args.run_name}_{args.data_name}_{args.model_name_dir}",
        tensor_parallel_size=1,
        dtype=torch.bfloat16,
        gpu_memory_utilization = args.gpu_memory_utilization,
        tokenizer=args.model_name,
        max_model_len=args.max_input_tokens,
        max_num_seqs=args.batch_size,
    )

    # Define sampling parameters
    sampling_params = SamplingParams(
        n=1,
        temperature=0.01,
        max_tokens=args.max_output_tokens,
        stop=["<|eot_id|>", "<|reserved_special_token_0|>", "<eos>"]
    )

    outputs = llm.generate(
        prompt_list,
        sampling_params,
    )
    responses = [output.outputs[0].text for output in outputs]

    response_dict = {}
    for i, res in enumerate(responses):
        response_dict[i+1] = res
    return response_dict


def prepare_eval_dataset(prompt_list, user_seq_data, user_target):
    """
    평가를 위한 데이터셋 준비
    
    Args:
        prompt_list: 프롬프트 리스트
        user_seq_data: 사용자별 시퀀스 데이터 (히스토리)
        user_target: 사용자별 타겟 아이템
    
    Returns:
        eval_data: 평가용 데이터 리스트 (각 항목은 dict)
    """
    eval_data = []
    
    for user_id in sorted(user_seq_data.keys()):
        # user_id는 1-indexed
        idx = user_id - 1
        if idx >= len(prompt_list):
            continue
            
        eval_data.append({
            'prompt': prompt_list[idx],
            'target': user_target[user_id],
            'history': user_seq_data[user_id],
        })
    
    return eval_data


def evaluate_model(args, eval_data, split="test"):
    """
    학습된 모델을 평가
    
    Args:
        args: 학습 설정 파라미터
        eval_data: 평가용 데이터셋
        split: 데이터셋 split 이름
    
    Returns:
        results: 평가 결과 딕셔너리
    """
    print(f"\n{'='*80}")
    print(f"🔍 Starting Evaluation on {split.upper()} Set")
    print(f"{'='*80}")
    
    # GPU 메모리 정리
    torch.cuda.empty_cache()
    gc.collect()
    
    # Evaluator 설정을 위한 args 준비
    # args에 필요한 속성들이 없으면 추가
    checkpoint_dir = f"models/{args.run_name}_{args.data_name}_{args.model_name_dir}"
    
    # evaluator에 필요한 속성 설정
    if not hasattr(args, 'checkpoint_dir'):
        args.checkpoint_dir = checkpoint_dir
    if not hasattr(args, 'policy_model'):
        args.policy_model = args.model_name
    if not hasattr(args, 'max_length'):
        args.max_length = args.max_input_tokens
    if not hasattr(args, 'max_new_tokens'):
        args.max_new_tokens = args.max_output_tokens
    if not hasattr(args, 'bf16'):
        args.bf16 = True
    if not hasattr(args, 'eval_batch_size'):
        args.eval_batch_size = args.batch_size
    if not hasattr(args, 'emb_model_name'):
        args.emb_model_name = 'mixedbread-ai/mxbai-embed-large-v1'
    if not hasattr(args, 'emb_type'):
        args.emb_type = 'title'
    if not hasattr(args, 'dataset_name'):
        args.dataset_name = args.data_name
    if not hasattr(args, 'eval_emb_max_length'):
        args.eval_emb_max_length = 512
    if not hasattr(args, 'eval_emb_batch_size'):
        args.eval_emb_batch_size = 512
    if not hasattr(args, 'eval_samples'):
        args.eval_samples = 100000
    
    # Evaluator 인스턴스 생성 및 평가 실행
    evaluator = RecommendationEvaluator(args, checkpoint_dir)
    
    try:
        results = evaluator.evaluate(eval_data, split=split, save_log=True)
    finally:
        # 평가 완료 후 메모리 정리
        evaluator.cleanup()
        del evaluator
        torch.cuda.empty_cache()
        gc.collect()
        
        if torch.cuda.is_available():
            print(f"\n💾 GPU Memory after evaluation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    return results


if __name__ == "__main__":
    args = parse_arguments()
    random.seed(args.seed)
    print_arguments(args)
    args.model_name_dir = args.model_name.split("/")[-1]

    # load data
    (
        train_user_seq_data, val_user_seq_data, test_user_seq_data, 
        train_pos_item_ids, val_pos_item_ids, test_pos_item_ids, 
        train_user_target, val_user_target, test_user_target
    ) = get_uid_to_seq_data(args)

    item_meta = get_item_meta(args)

    # get prompt
    train_uid_to_prompt = get_time_aware_user_history_text(args, split="train", item_meta=item_meta, prefix="")
    val_uid_to_prompt = get_time_aware_user_history_text(args, split="valid", item_meta=item_meta, prefix="")
    test_uid_to_prompt = get_time_aware_user_history_text(args, split="test", item_meta=item_meta, prefix="")

    train_uid_to_prompt = get_formatted_prompt_list(args, train_uid_to_prompt)
    val_uid_to_prompt = get_formatted_prompt_list(args, val_uid_to_prompt)
    test_uid_to_prompt = get_formatted_prompt_list(args, test_uid_to_prompt)
    
    # load target text
    train_user_preference_file = f"data_processed/{args.data_name}_{args.target_model_name}_train_{args.target}.json"
    valid_user_preference_file = f"data_processed/{args.data_name}_{args.target_model_name}_valid_{args.target}.json"
    test_user_preference_file = f"data_processed/{args.data_name}_{args.target_model_name}_test_{args.target}.json"
    with open(train_user_preference_file, 'r') as f:
        train_user_preference = json.load(f)
    train_user_preference = {int(k): v for k, v in train_user_preference.items()}
    with open(valid_user_preference_file, 'r') as f:
        val_user_preference = json.load(f)
    val_user_preference = {int(k): v for k, v in val_user_preference.items()}
    with open(test_user_preference_file, 'r') as f:
        test_user_preference = json.load(f)
    test_user_preference = {int(k): v for k, v in test_user_preference.items()}

    train_target_text = get_user_text(
            args, train_user_seq_data, train_user_preference, item_meta, 
            user_to_target_item=train_user_target, 
            add_item_meta=args.add_item_meta, 
            add_target_item_meta=args.add_target_item_meta
        )
    train_target_text = train_target_text[1:]
    val_target_text = get_user_text(
            args, val_user_seq_data, val_user_preference, item_meta, 
            user_to_target_item=val_user_target, 
            add_item_meta=args.add_item_meta, 
            add_target_item_meta=args.add_target_item_meta
        )
    val_target_text = val_target_text[1:]
    # test_target_text = get_user_text(
    #         args, test_user_seq_data, test_user_preference, item_meta, 
    #         user_to_target_item=test_user_target, 
    #         add_item_meta=args.add_item_meta, 
    #         add_target_item_meta=args.add_target_item_meta
    #     )
    # test_target_text = test_target_text[1:]

    # print("="*30)
    # for i in [0, 10, 20, 30]:
    #     print(train_uid_to_prompt[i])
    #     print("="*30)
    #     print(train_target_text[i])
    #     print("="*50)

    train_prompt_list = [train_uid_to_prompt[i] for i in range(len(train_uid_to_prompt))]
    val_prompt_list = [val_uid_to_prompt[i] for i in range(len(val_uid_to_prompt))]
    test_prompt_list = [test_uid_to_prompt[i] for i in range(len(test_uid_to_prompt))]

    # Apply user filtering if enabled
    if args.use_filtered_users and args.filtered_user_file:
        print("="*50)
        print("Applying user filtering...")
        
        # Determine file path pattern
        if args.filtered_user_file:
            # Load train filtered users
            train_filter_file = args.filtered_user_file.replace("_train.json", "_train.json")
            val_filter_file = args.filtered_user_file.replace("_train.json", "_valid.json")
            test_filter_file = args.filtered_user_file.replace("_train.json", "_test.json")
        else:
            train_filter_file = None
            val_filter_file = None
            test_filter_file = None
        
        # Load filtered user IDs
        train_filtered_users = load_filtered_users(train_filter_file)
        val_filtered_users = load_filtered_users(val_filter_file)
        test_filtered_users = load_filtered_users(test_filter_file)
        
        # Filter train data
        if train_filtered_users:
            original_train_size = len(train_prompt_list)
            filtered_train_prompts = []
            filtered_train_targets = []
            for i in range(len(train_prompt_list)):
                user_id = i + 1  # 1-indexed
                if user_id in train_filtered_users:
                    filtered_train_prompts.append(train_prompt_list[i])
                    filtered_train_targets.append(train_target_text[i])
            train_prompt_list = filtered_train_prompts
            train_target_text = filtered_train_targets
            print(f"Train: Filtered {original_train_size} → {len(train_prompt_list)} users")
        
        # Filter validation data
        if val_filtered_users:
            original_val_size = len(val_prompt_list)
            filtered_val_prompts = []
            filtered_val_targets = []
            for i in range(len(val_prompt_list)):
                user_id = i + 1  # 1-indexed
                if user_id in val_filtered_users:
                    filtered_val_prompts.append(val_prompt_list[i])
                    filtered_val_targets.append(val_target_text[i])
            val_prompt_list = filtered_val_prompts
            val_target_text = filtered_val_targets
            print(f"Valid: Filtered {original_val_size} → {len(val_prompt_list)} users")
        
        # Filter test data (if needed)
        if test_filtered_users:
            original_test_size = len(test_prompt_list)
            filtered_test_prompts = []
            for i in range(len(test_prompt_list)):
                user_id = i + 1  # 1-indexed
                if user_id in test_filtered_users:
                    filtered_test_prompts.append(test_prompt_list[i])
            test_prompt_list = filtered_test_prompts
            print(f"Test: Filtered {original_test_size} → {len(test_prompt_list)} users")
        
        print("="*50)

    train_prompt_list = train_prompt_list[:args.num_train_samples]
    train_target_text = train_target_text[:args.num_train_samples]

    val_prompt_list = val_prompt_list[:args.num_test_samples]
    val_target_text = val_target_text[:args.num_test_samples]

    test_prompt_list = test_prompt_list[:args.num_test_samples]

    train_dataset = get_dataset(
        train_prompt_list,
        train_target_text,
    )

    val_dataset = get_dataset(
        val_prompt_list,
        val_target_text,
    )

    # test_dataset = get_dataset(
    #     test_prompt_list,
    #     test_target_text,
    # )

    """ Prepare the model """
    quantization_config = get_quantization_config(args)
    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.5,
        r=args.rank_dim,
        # target_modules="all-linear",
        target_modules=["q_proj", "v_proj", "k_proj"],
        task_type="CAUSAL_LM",
    )
    tokenizer = initialize_tokenizer(args.model_name)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        dtype=torch.bfloat16, 
        )

    # fine-tune only Gemma3TextModel
    if "gemma-3-1b" in args.model_name:
        model = Gemma3ForCausalLM.from_pretrained(
            args.model_name,
            dtype=torch.bfloat16, 
            )
    elif "gemma-3" in args.model_name:
        model = model.model


    torch_dtype = model.dtype

    if args.epochs > 0:
        training_args = SFTConfig(
            output_dir="temp",              # directory to save and repository id
            max_length=args.max_input_tokens,      # max sequence length for model and packing of the dataset
            packing=True,                          # Groups multiple samples in the dataset into a single sequence
            num_train_epochs=args.epochs,                     # number of training epochs
            per_device_train_batch_size=4,          # batch size per device during training
            gradient_checkpointing=False,           # Caching is incompatible with gradient checkpointing
            max_grad_norm=1.0,
            optim="adamw_torch_fused",              # use fused adamw optimizer
            logging_steps=100,                        # log every step
            save_strategy="epoch",                  # save checkpoint every epoch
            eval_strategy="epoch",                  # evaluate checkpoint every epoch
            learning_rate=args.learning_rate,            # learning rate
            # fp16=True if torch_dtype == torch.float16 else False,   # use float16 precision
            bf16=True,  # use bfloat16 precision
            lr_scheduler_type="constant",           # use constant learning rate scheduler
            # push_to_hub=True,                       # push model to hub
            report_to="wandb",                # report metrics to tensorboard
            dataset_kwargs={
                "add_special_tokens": False, # Template with special tokens
                "append_concat_token": True, # Add EOS token as separator token between examples
            }
        )
        # 4. Initialize the SFTTrainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            # peft_config=lora_config, # Pass the LoRA configuration
            processing_class=tokenizer,
            # packing=True,
            # max_seq_length=512, # Maximum sequence length for training
            # Add other training arguments as needed, e.g., learning_rate, num_train_epochs
        )
        trainer.train()
        os.makedirs(f"models/{args.run_name}_{args.data_name}_{args.model_name_dir}", exist_ok=True)
        model.save_pretrained(f"models/{args.run_name}_{args.data_name}_{args.model_name_dir}")
        
        # 학습 완료 후 메모리 정리
        print("\n" + "="*80)
        print("🧹 Cleaning up training resources...")
        print("="*80)
        
        # Trainer와 model 정리
        if hasattr(trainer, 'model'):
            trainer.model = None
        if hasattr(trainer, 'optimizer'):
            trainer.optimizer = None
        if hasattr(trainer, 'lr_scheduler'):
            trainer.lr_scheduler = None
        
        del model
        del trainer
        
        # GPU 메모리 강제 해제
        torch.cuda.empty_cache()
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            print(f"💾 GPU Memory after training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"💾 GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # # 

    # Optional: generate and save the results
    if args.save_responses:
        print("\n" + "="*80)
        print("📝 Generating and saving responses...")
        print("="*80)
        
        response_dict = generate_responses_with_vllm(args, train_prompt_list)
        with open(f"data_processed/{args.run_name}_{args.data_name}_{args.model_name_dir}_train_results.json", "w") as f:
            json.dump(response_dict, f)

        response_dict = generate_responses_with_vllm(args, val_prompt_list)
        with open(f"data_processed/{args.run_name}_{args.data_name}_{args.model_name_dir}_valid_results.json", "w") as f:
            json.dump(response_dict, f)

        response_dict = generate_responses_with_vllm(args, test_prompt_list)
        with open(f"data_processed/{args.run_name}_{args.data_name}_{args.model_name_dir}_test_results.json", "w") as f:
            json.dump(response_dict, f)

        print("-"*50)
        for i in [1, 10, 20, 30]:
            if i in response_dict:
                print(response_dict[i])
                print("-"*50)

    # Evaluation with RecommendationEvaluator
    if args.run_evaluation:
        print("\n" + "="*80)
        print("🎯 Starting Model Evaluation")
        print("="*80)
        
        # Prepare evaluation datasets
        # Get the original user_seq_data and user_target for evaluation
        # We need to map prompt_list back to user_seq_data
        
        # For test evaluation - use full test data if available
        test_eval_data = prepare_eval_dataset(
            test_prompt_list,
            test_user_seq_data,
            test_user_target
        )
        
        print(f"Prepared {len(test_eval_data)} test samples for evaluation")
        
        # Run evaluation
        test_results = evaluate_model(args, test_eval_data, split="test")
        
        print("\n" + "="*80)
        print("✅ Evaluation Complete!")
        print("="*80)
        print("\nTest Results:")
        for metric_name, value in test_results.items():
            print(f"  {metric_name.upper()}: {value:.4f}")
        print("="*80)

    # from transformers import pipeline
    # # Load the model and tokenizer into the pipeline
    # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    # response_list = pipe(
    #     test_prompt_list, max_new_tokens=1024, disable_compile=True, 
    #     device=args.device, torch_dtype=torch.bfloat16, do_sample=False, num_return_sequences=1
    #     )
    # with open(f"data_processed/{args.run_name}_{args.data_name}_{args.model_name_dir}_test_results.json", "w") as f:
    #     json.dump(response_list, f)
    