# GRPO 기반 RL4Rec 학습 가이드

## 개요

TRL (Transformer Reinforcement Learning) 라이브러리의 **GRPO (Group Relative Policy Optimization)**를 활용하여 NDCG 기반 리워드로 추천 시스템을 학습하는 가이드입니다.

## GRPO란?

GRPO (Group Relative Policy Optimization)는 PPO의 개선된 변형으로, 다음과 같은 특징이 있습니다:

- **Group-based Normalization**: 같은 프롬프트에서 생성된 여러 샘플들의 상대적 품질을 비교
- **Sample Efficiency**: 각 프롬프트당 여러 샘플을 생성하여 학습 효율 향상
- **Stable Training**: 그룹 내 정규화로 안정적인 학습

### PPO vs GRPO

| 특징 | PPO | GRPO |
|------|-----|------|
| 샘플링 | 프롬프트당 1개 | 프롬프트당 N개 |
| 정규화 | 배치 전체 | 그룹 내부 |
| 학습 안정성 | 보통 | 높음 |
| 샘플 효율성 | 보통 | 높음 |

## 주요 파일

```
src/
├── grpo_train.py           # GRPO 학습 스크립트 (메인)
├── train_utils/
│   └── reward_funtion.py   # NDCG 리워드 함수
└── rec_model.py            # vLLM 통합 모델

runs/
├── run_grpo_train.sh       # 전체 학습 실행
└── run_grpo_test.sh        # 빠른 테스트 실행

docs/
└── GRPO_TRAINING_GUIDE.md  # 이 문서
```

## 사용 방법

### 1. 사전 준비

#### Retrieval Service 시작
```bash
./runs/run_retrieval.sh
```

#### Ray 클러스터 확인
```bash
ray status
```

### 2. 빠른 테스트 (Dummy 데이터)

학습 파이프라인을 빠르게 테스트:

```bash
./runs/run_grpo_test.sh
```

**테스트 설정:**
- Dummy 데이터 100개
- 1 에포크, 50 스텝
- 배치 크기 4, 샘플 2개

### 3. 전체 학습 실행

실제 데이터로 학습:

```bash
./runs/run_grpo_train.sh
```

**학습 설정:**
- Beauty 데이터셋
- NDCG@10 리워드
- 3 에포크, 10,000 스텝
- 배치 크기 4, 샘플 4개

### 4. 커스텀 설정

```bash
python src/grpo_train.py \
    --policy_model "gpt2" \
    --dataset_name "beauty" \
    --reward_type "ndcg" \
    --k 10 \
    --batch_size 4 \
    --num_sample_generations 4 \
    --learning_rate 1e-5 \
    --num_epochs 3 \
    --checkpoint_dir "checkpoints/my_experiment" \
    --normalize_rewards
```

## 주요 파라미터

### 모델 설정

```bash
--policy_model "gpt2"              # HuggingFace 모델 이름
--device "cuda"                    # 디바이스 (cuda/cpu)
--max_length 128                   # 입력 최대 길이
--max_new_tokens 64                # 생성 토큰 수
--temperature 1.0                  # 샘플링 온도
--use_ref_model                    # Reference 모델 사용 (KL penalty)
```

### GRPO 학습 설정

```bash
--batch_size 4                     # 배치 크기
--num_sample_generations 4         # 프롬프트당 생성 샘플 수 (중요!)
--gradient_accumulation_steps 4    # Gradient accumulation
--learning_rate 1e-5               # 학습률
--max_grad_norm 1.0                # Gradient clipping
--num_epochs 3                     # 에포크 수
--max_steps 10000                  # 최대 스텝 수
```

**중요:** `num_sample_generations`는 GRPO의 핵심 파라미터입니다:
- 값이 클수록: 더 많은 샘플 비교, 학습 안정성 증가, 계산 비용 증가
- 권장 값: 2~8

### 리워드 설정

```bash
--reward_type "ndcg"               # 리워드 타입 (ndcg/hit/mrr/mixed)
--k 10                             # Top-K 값
--normalize_rewards                # 리워드 정규화
```

### 데이터 설정

```bash
--prompt_file "path/to/prompts.json"
--sequential_file "path/to/sequential.txt"
--use_dummy                        # Dummy 데이터 사용 (테스트용)
--dummy_size 1000                  # Dummy 데이터 크기
```

## 학습 프로세스

### 1. 데이터 로드

```python
dataset = RecommendationDataset(
    prompt_file="prompts.json",
    sequential_file="sequential.txt",
    tokenizer=tokenizer,
    use_dummy=False,  # 실제 데이터 사용
)
```

데이터 형식:
- **prompts.json**: `{user_id: prompt_text, ...}`
- **sequential.txt**: `user_id item1 item2 ... itemN target`

### 2. GRPO 학습 스텝

```
For each batch:
    1. 프롬프트 토크나이즈
    2. 각 프롬프트당 N개 샘플 생성 (num_sample_generations)
    3. 각 샘플의 NDCG 리워드 계산
    4. 그룹 내 상대적 품질 비교
    5. GRPO 알고리즘으로 모델 업데이트
```

### 3. 리워드 계산

```python
# 각 샘플의 리워드 계산
for each generation:
    # RetrievalService를 통해 유사도 점수 계산
    scores = retrieval_service.calculate(generation)
    
    # NDCG 계산
    ndcg = calculate_ndcg(
        scores=scores,
        target=target_item,
        history=history_items,
        k=10
    )
    
    rewards.append(ndcg)
```

## 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                      GRPO Training Loop                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────┐
              │  1. Load Batch (Prompts)  │
              └───────────────┬───────────┘
                              │
                              ▼
              ┌───────────────────────────────────────┐
              │  2. Generate N samples per prompt     │
              │     (num_sample_generations=4)        │
              └───────────────┬───────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────────────┐
              │  3. Calculate Rewards (NDCG)          │
              │     - Ray RetrievalService            │
              │     - Similarity scores               │
              │     - NDCG@K calculation              │
              └───────────────┬───────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────────────┐
              │  4. Group Normalization               │
              │     - Compare samples in same group   │
              │     - Relative quality scoring        │
              └───────────────┬───────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────────────┐
              │  5. GRPO Update                       │
              │     - Policy gradient                 │
              │     - KL penalty (optional)           │
              └───────────────────────────────────────┘
```

## 평가 메트릭

### 학습 중 메트릭

```
Step    100 | Epoch 1 Batch   10 | Reward: 0.3245 ± 0.1234 | Loss: 2.3456
```

- **Reward**: 평균 NDCG@K
- **± (std)**: 리워드 표준편차 (낮을수록 안정적)
- **Loss**: GRPO 손실

### Epoch 요약

```
📊 Epoch 1 Summary:
  Average Reward: 0.3456
🏆 Best model saved: checkpoints/grpo/checkpoint_best (reward: 0.3456)
```

## 체크포인트 관리

### 자동 저장

```
checkpoints/grpo/
├── checkpoint_step_500/     # 주기적 저장
├── checkpoint_step_1000/
├── checkpoint_best/         # 최고 성능 모델
└── checkpoint_final/        # 최종 모델
```

### 체크포인트 로드

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("checkpoints/grpo/checkpoint_best")
tokenizer = AutoTokenizer.from_pretrained("checkpoints/grpo/checkpoint_best")
```

## 고급 기능

### 1. Reference Model 사용

KL Divergence penalty를 위한 reference 모델:

```bash
python src/grpo_train.py \
    --use_ref_model \
    --policy_model "gpt2" \
    ...
```

Reference 모델은 초기 모델의 복사본으로, 학습 중 고정됩니다.

### 2. Mixed Precision Training

```bash
# BFloat16 (권장, A100/H100)
python src/grpo_train.py --bf16 ...

# Float16 (V100/RTX)
python src/grpo_train.py --fp16 ...
```

### 3. W&B 로깅

```bash
python src/grpo_train.py \
    --report_to "wandb" \
    --run_name "grpo_ndcg10_experiment" \
    ...
```

### 4. 커스텀 리워드 함수

`train_utils/reward_funtion.py`를 수정하여 새로운 리워드 함수 추가:

```python
class CustomRewardFunction(RLRewardFunction):
    def __call__(self, generated_texts, target_items, history_items, **kwargs):
        # 기본 NDCG
        ndcg = calculate_ndcg(...)
        
        # 추가 메트릭 (예: diversity)
        diversity = calculate_diversity(generated_texts)
        
        # 결합
        rewards = 0.8 * ndcg + 0.2 * diversity
        return rewards
```

## 트러블슈팅

### RetrievalService 연결 실패

```
❌ Error: RetrievalService not found!
```

**해결:**
```bash
./runs/run_retrieval.sh
ray status  # 확인
```

### CUDA OOM 에러

**해결 1**: 배치 크기 줄이기
```bash
--batch_size 2 \
--num_sample_generations 2 \
--gradient_accumulation_steps 8
```

**해결 2**: Gradient checkpointing (TODO)

### 느린 생성 속도

**원인**: `num_sample_generations`가 너무 큼

**해결**: 샘플 수 줄이기
```bash
--num_sample_generations 2  # 4에서 2로 감소
```

### 불안정한 학습 (Loss 발산)

**해결 1**: 학습률 감소
```bash
--learning_rate 5e-6  # 1e-5에서 감소
```

**해결 2**: Gradient clipping 강화
```bash
--max_grad_norm 0.5  # 1.0에서 감소
```

**해결 3**: 리워드 정규화
```bash
--normalize_rewards
```

## 성능 최적화 팁

### 1. 효율적인 배치 설정

```bash
# GPU 메모리를 최대한 활용
batch_size × num_sample_generations × gradient_accumulation_steps = 효과적 배치 크기

# 예: 효과적 배치 크기 64
--batch_size 4 \
--num_sample_generations 4 \
--gradient_accumulation_steps 4
```

### 2. 데이터 로딩 최적화

- 데이터셋을 메모리에 캐싱
- DataLoader의 `num_workers` 조정 (TODO)

### 3. 생성 속도 최적화

- vLLM 사용 (rec_model.py에 이미 구현됨)
- Batched generation

## 실험 예제

### Baseline 실험

```bash
python src/grpo_train.py \
    --policy_model "gpt2" \
    --reward_type "ndcg" \
    --k 10 \
    --batch_size 4 \
    --num_sample_generations 4 \
    --learning_rate 1e-5 \
    --run_name "baseline_ndcg10"
```

### Hit@K 리워드 실험

```bash
python src/grpo_train.py \
    --reward_type "hit" \
    --k 10 \
    --run_name "hit10_experiment"
```

### Mixed 리워드 실험

```bash
python src/grpo_train.py \
    --reward_type "mixed" \
    --k 10 \
    --run_name "mixed_ndcg_hit"
```

## FAQ

**Q: GRPO와 PPO의 차이는?**
A: GRPO는 각 프롬프트당 여러 샘플을 생성하여 그룹 내에서 상대적 품질을 비교합니다. 이는 학습을 더 안정적이고 효율적으로 만듭니다.

**Q: num_sample_generations를 몇으로 설정해야 하나요?**
A: 2~8 사이를 권장합니다. 높을수록 안정적이지만 계산 비용이 증가합니다.

**Q: Dummy 데이터로 테스트하는 이유는?**
A: 학습 파이프라인을 빠르게 검증하고 버그를 찾기 위함입니다.

**Q: Reference model이 필요한가요?**
A: 선택사항입니다. KL penalty를 원하면 사용하세요.

## 참고 자료

- [TRL 공식 문서](https://huggingface.co/docs/trl)
- [GRPO 논문](https://arxiv.org/abs/...)
- [vLLM 문서](https://docs.vllm.ai/)
- [Ray 문서](https://docs.ray.io/)



