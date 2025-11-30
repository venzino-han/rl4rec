"""
Dataset utilities for RL4Rec
추천 시스템 학습을 위한 데이터셋 및 데이터로더 유틸리티
"""

import os
import json
import pickle
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class PromptGenerator:
    """
    사용자 시퀀스로부터 프롬프트를 생성하는 클래스
    """
    
    def __init__(
        self,
        item_metadata: Dict,
        use_brand: bool = True,
        use_category: bool = True,
        use_description: bool = False,
        use_features: bool = False,
        use_last_item: bool = True,
        max_history_len: int = 5,
        history_text_max_length: int = 100,
    ):
        """
        Args:
            item_metadata: 아이템 메타데이터 딕셔너리
            use_brand: 브랜드 정보 포함 여부
            use_category: 카테고리 정보 포함 여부
            use_description: 설명 정보 포함 여부
            use_features: 특징 정보 포함 여부
            use_last_item: 마지막 아이템 강조 여부
            max_history_len: 최대 히스토리 길이
            history_text_max_length: 히스토리 텍스트 최대 단어 수
        """
        self.item_metadata = item_metadata
        self.use_brand = use_brand
        self.use_category = use_category
        self.use_description = use_description
        self.use_features = use_features
        self.use_last_item = use_last_item
        self.max_history_len = max_history_len
        self.history_text_max_length = history_text_max_length
    
    def generate_prompt(self, item_ids: List[int]) -> str:
        """
        사용자 시퀀스로부터 프롬프트 생성
        
        Args:
            item_ids: 아이템 ID 리스트
        
        Returns:
            생성된 프롬프트 문자열
        """
        # 히스토리 텍스트 리스트
        history_text_list = []
        
        # 각 아이템 처리
        for item_id in item_ids:
            item_data = self.item_metadata.get(item_id)
            if item_data is None:
                # 메타데이터가 없는 경우 스킵
                continue
            
            item_title = item_data.get('title', 'Unknown Item')
            item_brand = item_data.get('brand', 'Unknown Brand')
            item_categories = item_data.get('category', 'Unknown Category')
            item_description = item_data.get('description', '')
            
            # 기본 히스토리 포맷
            item_history_text = f"**Title:** `{item_title}`"
            
            if self.use_brand:
                item_history_text += f"\n**Brand:** {item_brand}"
            
            if self.use_category:
                item_history_text += f"\n**Categories:** {item_categories}"
            
            if self.use_description and item_description:
                item_description = item_description.replace("\n", " ")
                if len(item_description.split()) > self.history_text_max_length:
                    item_description = " ".join(
                        item_description.split()[:self.history_text_max_length]
                    ) + "..."
                item_history_text += f"\n**Description:** {item_description}"
            
            if self.use_features and item_features:
                if len(item_features.split("\n")) > 10:
                    item_features = "\n".join(item_features.split("\n")[:10])
                item_features = item_features.replace("\n-", ",").replace("- ", "")
                item_history_text += f"\n**Features:**\n{item_features}"
            
            history_text_list.append(item_history_text)
        
        # 히스토리 길이 제약 적용
        if len(history_text_list) > self.max_history_len:
            history_text_list = history_text_list[-self.max_history_len:]
        
        # 최종 히스토리 텍스트 구성
        history_text = "\n---\n".join(
            f"{i+1}. {history}" for i, history in enumerate(history_text_list)
        )
        
        # 마지막 아이템 강조
        if self.use_last_item and len(item_ids) > 0:
            last_item = self.item_metadata.get(item_ids[-1], {})
            last_item_title = last_item.get('title', 'Unknown Item')
            history_text += f"\n\n`{last_item_title}` is the most recently purchased item."
        
        # 최종 프롬프트 생성
        prompt = (
            f"# User Purchase History\n\n"
            f"{history_text}\n\n"
            f"# Task\n"
            f"Based on this user's purchase history, describe user's preference:\n"
        )
        
        return prompt


class RecommendationDataset(Dataset):
    """
    추천 시스템용 데이터셋
    사용자 히스토리, 타겟 아이템, 프롬프트를 포함
    """
    
    def __init__(
        self,
        data_name: str,
        item_metadata: Dict,
        prompt_generator: PromptGenerator,
        split: str = "train",
        num_negs: int = 0,
        num_items: Optional[int] = None,
    ):
        """
        Args:
            data_name: 데이터셋 이름
            item_metadata: 아이템 메타데이터 딕셔너리
            prompt_generator: 프롬프트 생성기
            split: 데이터 분할 ("train", "valid", "test")
            num_negs: 사전 샘플링할 negative 아이템 수 (0이면 비활성화)
            num_items: 전체 아이템 수 (negative sampling에 필요)
        """
        self.item_metadata = item_metadata
        self.prompt_generator = prompt_generator
        self.split = split
        self.num_negs = num_negs
        self.num_items = num_items
        
        sequential_file = f"data/{data_name}/sequential_data.txt"
        self._load_real_data(sequential_file, split)
        
        # 프롬프트 미리 생성 (초기화 시점)
        print(f"✍️  Pre-generating prompts for {len(self.user_ids)} users...")
        self.prompt_dict = {}
        for user_id in self.user_ids:
            history = self.history_dict[user_id]
            self.prompt_dict[user_id] = self.prompt_generator.generate_prompt(history)
        
        # Negative items 미리 샘플링 (초기화 시점)
        self.neg_items_dict = {}
        if self.num_negs > 0:
            if self.num_items is None:
                raise ValueError("num_items must be provided when num_negs > 0")
            print(f"🎲 Pre-sampling {self.num_negs} negative items for each user...")
            self._sample_negative_items()
        
        print(f"✓ {split.upper()} Dataset loaded: {len(self.user_ids)} users")
    
    def _load_real_data(
        self,
        sequential_file: str,
        split: str,
    ):
        """실제 데이터 로드"""
        all_user_ids = []
        all_history = {}
        all_targets = {}
        
        if split == "train":
            target_index = -3
        elif split == "valid":
            target_index = -2
        elif split == "test":
            target_index = -1
        else:
            raise ValueError(f"Invalid split: {split}")
        
        with open(sequential_file, "r") as f:
            for line in f:
                parts = [int(p) for p in line.strip().split()]
                user_id = parts[0]
                history = parts[1:target_index]
                target = parts[target_index]
                
                all_user_ids.append(user_id)
                all_history[user_id] = history
                all_targets[user_id] = target
        
        self.user_ids = all_user_ids
        self.history_dict = all_history
        self.target_dict = all_targets
    
    def _sample_negative_items(self):
        """각 사용자별로 negative items 사전 샘플링"""
        rng = np.random.RandomState(42)  # 재현성을 위한 고정 seed
        
        for user_id in self.user_ids:
            history = self.history_dict[user_id]
            target = self.target_dict[user_id]
            
            # 제외할 아이템 (history + target)
            excluded = set(history + [target])
            
            # 가능한 negative items (전체 아이템 - 제외 아이템)
            all_items = set(range(self.num_items))
            candidate_items = list(all_items - excluded)
            
            # 랜덤 샘플링
            if len(candidate_items) >= self.num_negs:
                neg_items = rng.choice(candidate_items, size=self.num_negs, replace=False).tolist()
            else:
                # 후보가 부족한 경우 중복 샘플링
                neg_items = rng.choice(candidate_items, size=self.num_negs, replace=True).tolist()
            
            self.neg_items_dict[user_id] = neg_items
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        history = self.history_dict[user_id]
        target = self.target_dict[user_id]
        
        # 미리 생성된 프롬프트 사용
        prompt = self.prompt_dict[user_id]
        
        result = {
            "prompt": prompt,
            "history": history,
            "target": target,
            "user_id": user_id,
        }
        
        # Negative items가 있으면 추가
        if self.num_negs > 0:
            result["neg_items"] = self.neg_items_dict[user_id]
        
        return result


def collate_fn(batch):
    """
    DataLoader용 collate function
    """
    return {
        "queries": [item["query"] for item in batch],
        "histories": [item["history"] for item in batch],
        "targets": [item["target"] for item in batch],
        "user_ids": [item["user_id"] for item in batch],
    }


def load_item_metadata(dataset_name: str, data_dir: str = "data") -> Dict:
    """
    아이템 메타데이터 로드
    
    Args:
        dataset_name: 데이터셋 이름 (e.g., "beauty")
        data_dir: 데이터 디렉토리
    
    Returns:
        아이템 메타데이터 딕셔너리
    """
    # 메타데이터 파일 경로 시도
    possible_paths = [
        f"{data_dir}/{dataset_name}/meta_text.json",
    ]
    
    item_metadata = {}
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading item metadata from: {path}")
            
            if path.endswith('.json'):
                with open(path, 'r') as f:
                    data = json.load(f)
                    # Key를 int로 변환
                    item_metadata = {int(k): v for k, v in data.items()}
            elif path.endswith('.pkl'):
                with open(path, 'rb') as f:
                    item_metadata = pickle.load(f)
            
            print(f"✓ Loaded {len(item_metadata)} items")
            return item_metadata
    
    # 메타데이터 파일을 찾지 못한 경우 경고
    print(f"⚠️  Item metadata file not found. Using dummy metadata.")
    return {}


def create_dataloaders(
    args: argparse.Namespace,
) -> Tuple[DataLoader, DataLoader, DataLoader, PromptGenerator, Dict]:
    """
    Train/Valid/Test DataLoader 생성
    
    Args:
        args: argparse.Namespace
    
    Returns:
        (train_dataloader, valid_dataloader, test_dataloader, prompt_generator, item_metadata)
    """
    # 아이템 메타데이터 로드
    print(f"📦 Loading item metadata...")
    item_metadata = load_item_metadata(args.dataset_name)
    
    # num_items 가져오기 (args에 있으면 사용, 없으면 메타데이터에서 추출)
    num_items = getattr(args, 'num_items', None)
    if num_items is None and len(item_metadata) > 0:
        num_items = max(item_metadata.keys()) + 1
        print(f"  Inferred num_items from metadata: {num_items}")
    
    # num_negs 가져오기 (args에 있으면 사용, 없으면 0)
    num_negs = getattr(args, 'num_negs', 0)
    
    # 프롬프트 생성기
    print(f"✍️  Creating prompt generator...")
    prompt_generator = PromptGenerator(
        item_metadata=item_metadata,
        use_brand=args.use_brand,
        use_category=args.use_category,
        use_description=args.use_description,
        max_history_len=args.max_history_len,
        history_text_max_length=args.history_text_max_length,
    )
    
    # 데이터셋 생성
    print(f"📊 Creating datasets...")
    
    # Train dataset
    train_dataset = RecommendationDataset(
        data_name=args.dataset_name,
        item_metadata=item_metadata,
        prompt_generator=prompt_generator,
        split="train",
        num_negs=num_negs,
        num_items=num_items,
    )
    
    # Valid dataset
    valid_dataset = RecommendationDataset(
        data_name=args.dataset_name,
        item_metadata=item_metadata,
        prompt_generator=prompt_generator,
        split="valid",
        num_negs=num_negs,
        num_items=num_items,
    )
    
    # Test dataset
    test_dataset = RecommendationDataset(
        data_name=args.dataset_name,
        item_metadata=item_metadata,
        prompt_generator=prompt_generator,
        split="test",
        num_negs=num_negs,
        num_items=num_items,
    )
    
    # DataLoaders
    print(f"🔄 Creating dataloaders...")
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    print(f"✓ DataLoaders created:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Valid samples: {len(valid_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    if num_negs > 0:
        print(f"  Negative samples per user: {num_negs}")
    
    return train_dataset, valid_dataset, test_dataset, prompt_generator, item_metadata
    # return train_dataloader, valid_dataloader, test_dataloader, prompt_generator, item_metadata

