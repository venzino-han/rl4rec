# Negative Sampling 사용 가이드

## 개요

`num_negs` 기능을 통해 효율적인 negative sampling을 지원합니다. 전체 아이템 중에서 랜덤하게 negative items을 사전 샘플링하여 메모리와 계산 효율성을 높입니다.

## 주요 기능

### 1. Dataset에서 Negative Items 사전 샘플링

초기화 시점에 각 사용자별로 랜덤 negative items를 샘플링하여 저장합니다.

```python
from src.train_utils.dataset import RecommendationDataset, load_item_metadata, PromptGenerator

# 메타데이터 로드
item_metadata = load_item_metadata("beauty")
num_items = max(item_metadata.keys()) + 1

# 프롬프트 생성기
prompt_generator = PromptGenerator(item_metadata=item_metadata)

# Negative sampling 활성화 (num_negs=5)
train_dataset = RecommendationDataset(
    data_name="beauty",
    item_metadata=item_metadata,
    prompt_generator=prompt_generator,
    split="train",
    num_negs=5,  # 각 사용자당 5개의 negative items 샘플링
    num_items=num_items,
)

# 샘플 데이터 확인
sample = train_dataset[0]
print(sample.keys())  # ['prompt', 'history', 'target', 'user_id', 'neg_items']
print(f"Target: {sample['target']}")
print(f"Negatives: {sample['neg_items']}")  # [1234, 5678, 9012, 3456, 7890]
```

### 2. RetrievalService에서 효율적 스코어 계산

`targets`와 `neg_items`가 제공되면, 전체 아이템이 아닌 **target + negatives**에 대해서만 스코어를 계산합니다.

```python
import ray

# RetrievalService 연결
retrieval_service = ray.get_actor("RetrievalService", namespace="rl4rec")

# 텍스트, 타겟, negative items
texts = ["User prefers skincare products..."]
targets = [42]  # target item ID
neg_items = [[100, 200, 300, 400, 500]]  # negative item IDs

# 효율적 스코어 계산 (target + negatives만)
scores_ref = retrieval_service.calculate_reward.remote(
    texts,
    dataset_name="beauty",
    targets=targets,
    neg_items=neg_items,
)
scores = ray.get(scores_ref)

# scores.shape: [1, 6]  # 1 target + 5 negatives
# scores[0, 0]: target item에 대한 스코어
# scores[0, 1:]: negative items에 대한 스코어
```

### 3. Reward Function에서 사용

```python
from src.train_utils.reward_funtion import RecRewardFrunction

# Reward function 생성
reward_fn = RecRewardFrunction(
    retrieval_service_name="RetrievalService",
    namespace="rl4rec",
    dataset_name="beauty",
    reward_type="ndcg",
    k=10,
)

# Negative items와 함께 리워드 계산
generated_texts = ["User prefers..."]
targets = [42]
histories = [[1, 2, 3, 4, 5]]
neg_items = [[100, 200, 300, 400, 500]]

rewards = reward_fn(
    generated_texts=generated_texts,
    targets=targets,
    histories=histories,
    neg_items=neg_items,  # negative items 전달
)

print(f"Reward: {rewards}")
```

## 장점

### 1. 메모리 효율성
- **전체 인덱스 사용**: `[batch_size, 12102]` (전체 아이템)
- **Negative sampling 사용**: `[batch_size, 1 + num_negs]` (예: 1 + 5 = 6개)
- **메모리 절약**: ~2000배

### 2. 계산 효율성
- 전체 아이템 대신 소수의 아이템(target + negatives)만 계산
- GPU 메모리와 연산 시간 대폭 감소

### 3. 학습 효과
- Hard negative mining과 유사한 효과
- 더 빠른 학습 수렴 가능

## 사용 시나리오

### 시나리오 1: 전체 아이템 평가 (평가/테스트)
```python
# num_negs=0 또는 neg_items=None
test_dataset = RecommendationDataset(
    data_name="beauty",
    item_metadata=item_metadata,
    prompt_generator=prompt_generator,
    split="test",
    num_negs=0,  # negative sampling 비활성화
)

# 전체 아이템에 대한 스코어 계산
rewards = reward_fn(
    generated_texts=texts,
    targets=targets,
    histories=histories,
    # neg_items=None (기본값)
)
```

### 시나리오 2: Negative Sampling (학습)
```python
# num_negs > 0
train_dataset = RecommendationDataset(
    data_name="beauty",
    item_metadata=item_metadata,
    prompt_generator=prompt_generator,
    split="train",
    num_negs=99,  # target(1) + negatives(99) = 100개 아이템만 평가
    num_items=num_items,
)

# target + negatives만 평가
sample = train_dataset[0]
rewards = reward_fn(
    generated_texts=[sample['prompt']],
    targets=[sample['target']],
    histories=[sample['history']],
    neg_items=[sample['neg_items']],
)
```

## 검증

Negative items가 history와 target에 포함되지 않음을 자동 검증:

```python
# 자동 검증
sample = train_dataset[0]
history_set = set(sample['history'])
target = sample['target']
neg_set = set(sample['neg_items'])

# history와 target은 negatives에 포함되지 않음
assert target not in neg_set
assert len(history_set & neg_set) == 0
print("✓ Validation passed")
```

## 주의사항

1. **num_items 필수**: `num_negs > 0`일 때 `num_items`를 반드시 제공해야 함
2. **재현성**: `seed=42`로 고정되어 있어 동일한 negative items 생성
3. **메트릭 계산**: `use_negatives_only=True`일 때 NDCG, Hit@K, MRR은 target+negatives 범위 내에서만 계산

## 성능 비교

| 방식 | 스코어 계산 | 메모리 사용 | 계산 시간 |
|------|------------|------------|----------|
| 전체 아이템 | [batch, 12102] | 높음 | 느림 |
| Negative Sampling (5) | [batch, 6] | 매우 낮음 | 매우 빠름 |
| Negative Sampling (99) | [batch, 100] | 낮음 | 빠름 |

## 예제: 학습 파이프라인

```python
import argparse

# Arguments
args = argparse.Namespace(
    dataset_name="beauty",
    num_negs=99,  # 학습 시 negative sampling 사용
    num_items=12102,
    batch_size=32,
    # ... 기타 설정
)

# 데이터셋 생성
from src.train_utils.dataset import create_dataloaders
train_ds, valid_ds, test_ds, prompt_gen, item_meta = create_dataloaders(args)

# 학습 루프
for batch in train_ds:
    rewards = reward_fn(
        generated_texts=batch['prompts'],
        targets=batch['targets'],
        histories=batch['histories'],
        neg_items=batch['neg_items'],  # 사전 샘플링된 negatives
    )
    # ... 학습 진행
```

## 참고

- Negative sampling은 **학습 시**에만 사용하고, **평가/테스트**에서는 전체 아이템을 사용하는 것을 권장
- `num_negs` 값은 데이터셋 크기와 GPU 메모리에 따라 조정 (일반적으로 10~1000 사이)

