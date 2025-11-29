# TRL 기반 RL4Rec 학습 가이드

## 개요

TRL (Transformer Reinforcement Learning) 라이브러리를 활용하여 NDCG 기반 리워드로 추천 시스템을 학습하는 가이드입니다.

## 주요 기능

### 1. NDCG 기반 리워드 함수 (`reward_funtion.py`)

다양한 평가 메트릭을 제공합니다:

- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Hit@K**: 타겟 아이템이 Top-K에 포함되는지 여부
- **MRR@K**: Mean Reciprocal Rank
- **Mixed**: NDCG + Hit@K의 가중 평균

```python
from train_utils.reward_funtion import create_reward_function

# NDCG 리워드 함수 생성
reward_fn = create_reward_function(
    retrieval_service_name="RetrievalService",
    namespace="rl4rec",
    dataset_name="beauty",
    reward_type="ndcg",  # 'ndcg', 'hit', 'mrr', 'mixed'
    k=10,
)

# 리워드 계산
rewards = reward_fn(
    generated_texts=["product description 1", "product description 2"],
    target_items=[123, 456],
    history_items=[[1, 2, 3], [4, 5, 6]],
)
```

### 2. TRL PPO Trainer (`train_with_trl.py`)

PPO (Proximal Policy Optimization) 알고리즘을 사용한 학습:

```python
from train_with_trl import TRLTrainer

# Trainer 초기화
trainer = TRLTrainer(args)

# 학습 시작
trainer.train()
```

## 사용 방법

### 사전 준비

1. **Retrieval Service 실행**
   ```bash
   ./runs/run_retrieval.sh
   ```

2. **데이터 준비**
   - 프롬프트 파일: `data_processed/beauty_gemma-3-1b-it_test_user_preference.json`
   - 시퀀셜 데이터: `data/beauty/sequential_data.txt`

### 학습 실행

```bash
./runs/run_trl_train.sh
```

또는 직접 실행:

```bash
python src/train_with_trl.py \
    --model_name "gpt2" \
    --dataset_name "beauty" \
    --reward_type "ndcg" \
    --k 10 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --num_epochs 3 \
    --checkpoint_dir "checkpoints/trl_ppo"
```

### 주요 파라미터

#### 모델 설정
- `--model_name`: HuggingFace 모델 이름 (기본: "gpt2")
- `--device`: 학습 디바이스 (기본: "cuda")
- `--max_length`: 입력 최대 길이 (기본: 128)
- `--max_new_tokens`: 생성 토큰 수 (기본: 64)

#### 리워드 설정
- `--reward_type`: 리워드 타입 (ndcg/hit/mrr/mixed)
- `--k`: Top-K 값 (기본: 10)
- `--dataset_name`: 데이터셋 이름 (기본: "beauty")

#### PPO 학습 설정
- `--learning_rate`: 학습률 (기본: 1e-5)
- `--batch_size`: 배치 크기 (기본: 16)
- `--mini_batch_size`: PPO 미니배치 크기 (기본: 4)
- `--ppo_epochs`: PPO 에포크 수 (기본: 4)
- `--num_epochs`: 전체 에포크 수 (기본: 3)
- `--max_steps`: 최대 스텝 수 (기본: 10000)

## 아키텍처

### 학습 파이프라인

```
1. 프롬프트 생성
   └─> 사용자 히스토리 기반 프롬프트

2. Rollout 생성 (vLLM)
   └─> Policy Model이 텍스트 생성

3. 리워드 계산 (Ray + RetrievalService)
   ├─> 생성된 텍스트를 임베딩
   ├─> 데이터베이스와 유사도 계산
   └─> NDCG/Hit@K/MRR 계산

4. PPO 업데이트
   └─> Policy Model 파라미터 업데이트
```

### 비동기 처리

Ray를 활용한 비동기 리워드 계산:

```python
# 리워드 요청 (비동기)
reward_ref = retrieval_service.calculate_reward.remote(
    generated_texts,
    dataset_name="beauty"
)

# 다른 작업 수행 가능...

# 결과 수신
scores = ray.get(reward_ref)
```

## 평가 메트릭

### NDCG@K (Normalized Discounted Cumulative Gain)

순위 품질을 평가하는 메트릭:

```
NDCG@K = DCG@K / IDCG@K

DCG@K = Σ(rel_i / log2(i + 1))
```

- 타겟 아이템이 상위 랭크에 있을수록 높은 점수
- 0~1 범위로 정규화

### Hit@K

타겟 아이템이 Top-K에 포함되는지 여부:

```
Hit@K = 1 if target in top_k else 0
```

### MRR@K (Mean Reciprocal Rank)

타겟 아이템의 역순위:

```
MRR@K = 1 / rank(target)
```

## 고급 기능

### vLLM 통합

`rec_model.py`의 `generate` 메서드는 vLLM을 사용하여 빠른 생성을 제공합니다:

```python
# vLLM SamplingParams
sampling_params = SamplingParams(
    temperature=1.0,
    top_p=0.95,
    max_tokens=64,
    n=1,
)

# 생성
outputs = vllm_model.generate(prompts, sampling_params)
```

### 커스텀 리워드 함수

직접 리워드 함수를 작성할 수 있습니다:

```python
class CustomRewardFunction(RLRewardFunction):
    def __call__(self, generated_texts, target_items, history_items, **kwargs):
        # 커스텀 리워드 로직
        scores = self.get_scores(generated_texts)
        
        # 예: diversity 보너스 추가
        diversity_bonus = calculate_diversity(generated_texts)
        
        # NDCG + diversity
        ndcg = calculate_ndcg(scores, target_items, history_items, k=self.k)
        rewards = ndcg + 0.1 * diversity_bonus
        
        return rewards
```

## 체크포인트 및 로깅

### 체크포인트 저장

자동으로 주기적 체크포인트를 저장합니다:

```
checkpoints/trl_ppo/
├── checkpoint_step_500/
├── checkpoint_step_1000/
└── checkpoint_final/
```

### W&B 로깅

Weights & Biases 로깅 활성화:

```bash
python src/train_with_trl.py \
    --log_with "wandb" \
    --project_name "rl4rec-experiment"
```

## 트러블슈팅

### RetrievalService 연결 실패

```
❌ Error: RetrievalService not found!
```

**해결**: Retrieval Service를 먼저 실행하세요.
```bash
./runs/run_retrieval.sh
```

### CUDA OOM 에러

**해결**: 배치 크기를 줄이세요.
```bash
--batch_size 8 --mini_batch_size 2
```

### 느린 생성 속도

**해결**: vLLM 설정을 최적화하세요.
- `gpu_memory_utilization` 조정
- `tensor_parallel_size` 증가 (멀티 GPU)

## 참고 자료

- [TRL 공식 문서](https://huggingface.co/docs/trl)
- [vLLM 문서](https://docs.vllm.ai/)
- [Ray 문서](https://docs.ray.io/)



