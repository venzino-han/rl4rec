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
    
    지원하는 프롬프트 타입:
    - 'preference': 사용자 선호도 묘사
    - 'next_item': 다음 아이템 예측
    - 'recommendation': 추천 아이템 생성
    - 'user_profile': 사용자 프로필 생성
    - 'recent_preference': 최근 선호도 묘사
    """
    
    # 프롬프트 템플릿 정의
    PROMPT_TEMPLATES = {
        'seq_rec': {
        'title': 'You are an intelligent shopping assistant that helps predict what users may want to purchase next. Below is a list of items a user has purchased recently.\n' +\
                   'Your task is to infer one or multiple kinds of products they may want to buy next, and generate relevant query terms that can be used to search for these potential products.\n' +\
                   'Below is the user purchase history:\n',
        'task': 'Based on this user\'s purchase history, generate relevant query terms that can be used to search for these potential products.',
        },
        'preference': {
            'title': '# User Purchase History',
            'task': '# Task\nBased on this user\'s purchase history, describe user\'s preference:',
        },
        'next_item': {
            'title': '# User Purchase History',
            'task': '# Task\nBased on this user\'s purchase history, predict what item the user will purchase next:',
        },
        'recommendation': {
            'title': '# User Purchase History',
            'task': '# Task\nBased on this user\'s purchase history, recommend suitable items for the user:',
        },
        'user_profile': {
            'title': '# User Purchase History',
            'task': '# Task\nBased on this user\'s purchase history, create a detailed user profile describing their interests and preferences:',
        },
        'recent_preference': {
            'title': '# User Purchase History',
            'task': '# Task\nBased on this user\'s purchase history, describe user\'s most recent preference:',
        },
    }
    
    def __init__(
        self,
        item_metadata: Dict,
        data_name: str = None,
        prompt_type: str = 'seq_rec',
        use_brand: bool = True,
        use_category: bool = True,
        use_description: bool = False,
        use_features: bool = False,
        use_last_item: bool = True,
        use_date: bool = True,
        max_history_len: int = 5,
        history_text_max_length: int = 100,
        use_reviews: bool = False,
        days_filter: int = None,
    ):
        """
        Args:
            item_metadata: 아이템 메타데이터 딕셔너리
            data_name: 데이터셋 이름 (날짜 정보 로드에 사용)
            prompt_type: 프롬프트 타입 ('preference', 'next_item', 'recommendation', 'user_profile', 'recent_preference', 'reasoning')
            use_brand: 브랜드 정보 포함 여부
            use_category: 카테고리 정보 포함 여부
            use_description: 설명 정보 포함 여부
            use_features: 특징 정보 포함 여부
            use_last_item: 마지막 아이템 강조 여부
            use_date: 날짜 정보 포함 여부
            max_history_len: 최대 히스토리 길이
            history_text_max_length: 히스토리 텍스트 최대 단어 수 (review text에도 적용)
            use_reviews: 리뷰 텍스트 포함 여부
            days_filter: 최근 N일 이내의 리뷰만 포함 (None이면 필터링 안함)
        """
        self.item_metadata = item_metadata
        self.data_name = data_name
        self.use_brand = use_brand
        self.use_category = use_category
        self.use_description = use_description
        self.use_features = use_features
        self.use_last_item = use_last_item
        self.use_date = use_date
        self.max_history_len = max_history_len
        self.history_text_max_length = history_text_max_length
        self.use_reviews = use_reviews
        self.days_filter = days_filter
        
        # 프롬프트 타입 설정
        if prompt_type not in self.PROMPT_TEMPLATES:
            print(f"⚠️  Unknown prompt type '{prompt_type}'. Available types: {list(self.PROMPT_TEMPLATES.keys())}")
            print(f"   Using default 'recent_preference' type.")
            self.prompt_type = 'recent_preference'
        else:
            self.prompt_type = prompt_type
            print(f"✓ Using prompt type: '{self.prompt_type}'")
        
        # user2reviews_with_date.json 로드
        self.user_reviews_with_date = {}
        if data_name:
            date_file_path = f"data/{data_name}/user2reviews_with_date.json"
            if os.path.exists(date_file_path):
                print(f"Loading date information from: {date_file_path}")
                with open(date_file_path, 'r') as f:
                    self.user_reviews_with_date = json.load(f)
                print(f"✓ Loaded date information for {len(self.user_reviews_with_date)} users")
            else:
                print(f"⚠️  Date file not found: {date_file_path}. Dates will not be included.")
                self.use_date = False
    
    def generate_prompt(self, item_ids: List[int], user_id: Optional[int] = None, target_timestamp: Optional[int] = None) -> str:
        """
        사용자 시퀀스로부터 프롬프트 생성
        
        Args:
            item_ids: 아이템 ID 리스트
            user_id: 사용자 ID (날짜 정보 조회용, 선택적)
            target_timestamp: 타겟 타임스탬프 (days_filter 적용시 기준, 선택적)
        
        Returns:
            생성된 프롬프트 문자열
        """
        # 사용자의 리뷰 정보 가져오기
        user_reviews = []
        if user_id is not None:
            user_id_str = str(user_id)
            user_reviews = self.user_reviews_with_date.get(user_id_str, [])
        
        # 아이템 ID를 키로 하는 리뷰 매핑 생성
        item_to_review = {}
        if user_reviews:
            for review in user_reviews:
                item_id = int(review.get('item_id', -1))
                if item_id != -1:
                    item_to_review[item_id] = review
        
        # 히스토리 텍스트 리스트
        history_text_list = []
        
        # 각 아이템 처리
        for idx, item_id in enumerate(item_ids):
            item_data = self.item_metadata.get(item_id)
            if item_data is None:
                # 메타데이터가 없는 경우 스킵
                continue
            
            # 시간 필터링 (days_filter가 설정되어 있고 target_timestamp가 주어진 경우)
            if self.days_filter is not None and target_timestamp is not None and item_id in item_to_review:
                review = item_to_review[item_id]
                timestamp = int(review.get('timestamp', 0))
                if target_timestamp - timestamp > self.days_filter * 24 * 60 * 60:
                    continue
            
            item_title = item_data.get('title', 'Unknown Item')
            item_brand = item_data.get('brand', 'Unknown Brand')
            item_categories = item_data.get('category', 'Unknown Category')
            item_description = item_data.get('description', '')
            
            # reasoning 타입일 경우 간단한 포맷
            if self.prompt_type == "reasoning":
                item_history_text = f"{idx+1}) {item_title} "
            else:
                item_history_text = ""
                # 날짜 정보 추가
                if self.use_date and item_id in item_to_review:
                    item_date = item_to_review[item_id].get('date', '')
                    if item_date:
                        item_history_text += f"Date: {item_date}\n"
                
                # 기본 히스토리 포맷
                item_history_text += f"Item Title: {item_title}\n"
                
                if self.use_brand:
                    item_history_text += f"Brand: {item_brand}\n"
                
                if self.use_category:
                    item_history_text += f"Categories: {item_categories}\n"
                
                if self.use_description and item_description:
                    item_description = item_description.replace("\n", " ")
                    if len(item_description.split()) > self.history_text_max_length:
                        item_description = " ".join(
                            item_description.split()[:self.history_text_max_length]
                        ) + "..."
                    item_history_text += f"Description: {item_description}\n"
                
                if self.use_features and item_features:
                    if len(item_features.split("\n")) > 10:
                        item_features = "\n".join(item_features.split("\n")[:10])
                    item_features = item_features.replace("\n-", ",").replace("- ", "")
                    item_history_text += f"Features:\n{item_features}\n"
            
            # 리뷰 텍스트 추가
            if self.use_reviews and item_id in item_to_review:
                review_text = item_to_review[item_id].get('text', '')
                # limit review text words
                if review_text and len(review_text.split()) > self.history_text_max_length:
                    review_text = " ".join(review_text.split()[:self.history_text_max_length])
                if review_text:
                    item_history_text += f"Review:\n{review_text}\n"
            
            if item_history_text:
                history_text_list.append(item_history_text)
        
        # 히스토리가 비어있는 경우 마지막 아이템이라도 포함
        if len(history_text_list) == 0 and len(item_ids) > 0:
            last_item_id = item_ids[-1]
            item_data = self.item_metadata.get(last_item_id)
            if item_data:
                item_title = item_data.get('title', 'Unknown Item')
                item_brand = item_data.get('brand', 'Unknown Brand')
                
                if self.prompt_type == "reasoning":
                    item_history_text = f"1) {item_title} "
                else:
                    item_history_text = f"Item Title: {item_title}\n"
                    if self.use_brand:
                        item_history_text += f"Brand: {item_brand}\n"
                
                if self.use_reviews and last_item_id in item_to_review:
                    review_text = item_to_review[last_item_id].get('text', '')
                    if review_text and len(review_text.split()) > self.history_text_max_length:
                        review_text = " ".join(review_text.split()[:self.history_text_max_length])
                    if review_text:
                        item_history_text += f"Review:\n{review_text}\n"
                
                history_text_list.append(item_history_text)
        
        # 히스토리 길이 제약 적용
        if len(history_text_list) > self.max_history_len:
            history_text_list = history_text_list[-self.max_history_len:]
        
        # 최종 히스토리 텍스트 구성
        history_text = "\n\n".join(
            f"{i+1}. {history}" for i, history in enumerate(history_text_list)
        )
        
        # 마지막 아이템 강조
        if self.use_last_item and len(item_ids) > 0:
            last_item = self.item_metadata.get(item_ids[-1], {})
            last_item_title = last_item.get('title', 'Unknown Item')
            history_text += f"\n\n`{last_item_title}` is the most recently purchased item."
        
        # 선택된 프롬프트 템플릿 가져오기
        template = self.PROMPT_TEMPLATES[self.prompt_type]
        
        # 최종 프롬프트 생성
        prompt = (
            f"{template['title']}\n\n"
            f"{history_text}\n\n"
            f"{template['task']}\n"
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
            self.prompt_dict[user_id] = self.prompt_generator.generate_prompt(history, user_id=user_id)

        # print sample prompts
        for user_id in [10, 20, 30]:
            print(f"User {user_id}: \n{self.prompt_dict[user_id]}")
            print("-" * 100)
        
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
    
    # use_date 파라미터 가져오기 (args에 있으면 사용, 없으면 True)
    use_date = getattr(args, 'use_date', True)
    
    # prompt_type 파라미터 가져오기 (args에 있으면 사용, 없으면 'recent_preference')
    prompt_type = getattr(args, 'prompt_type', 'seq_rec')
    
    prompt_generator = PromptGenerator(
        item_metadata=item_metadata,
        data_name=args.dataset_name,
        prompt_type=prompt_type,
        use_brand=args.use_brand,
        use_category=args.use_category,
        use_description=args.use_description,
        use_date=use_date,
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

