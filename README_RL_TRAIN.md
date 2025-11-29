# RL4Rec Training Guide

강화학습 기반 추천 시스템 학습 가이드

## 📋 목차

1. [시스템 구조](#시스템-구조)
2. [설치 및 설정](#설치-및-설정)
3. [사용 방법](#사용-방법)
4. [파일 설명](#파일-설명)
5. [커스터마이징](#커스터마이징)

---

## 🏗️ 시스템 구조

```
┌─────────────────┐         ┌──────────────────┐
│  RL Trainer     │         │ Retrieval Service│
│  (GPU 0)        │◄───────►│  (GPU 1)         │
│                 │  Ray    │                  │
│ - Policy Model  │         │ - vLLM Embedding │
│ - Generate Text │         │ - Item Embeddings│
│ - Compute Loss  │         │ - Score Compute  │
└─────────────────┘         └──────────────────┘
```

### 주요 컴포넌트

1. **RetrievalService** (`src/retrieval_service.py`)
   - vLLM을 사용한 텍스트 임베딩
   - 사전 계산된 아이템 임베딩과의 유사도 계산
   - Ray Actor로 실행되어 다른 프로세스와 통신

2. **RecPolicy** (`src/rec_model.py`)
   - 사용자 히스토리 기반 텍스트 생성 Policy Model
   - LLM(GPT-2 등) 기반 생성 모델
   - Policy Gradient로 학습

3. **RLTrainer** (`src/rl_train.py`)
   - 전체 학습 루프 관리
   - Policy Model과 RetrievalService 연동
   - 체크포인트 및 로깅 관리

---

## 🔧 설치 및 설정

### 1. 필수 패키지 설치

```bash
pip install torch transformers vllm ray
```

### 2. 데이터 준비

아이템 임베딩 파일을 `data_emb/` 디렉토리에 준비:

```
data_emb/
├── beauty_review_description_mxbai-embed-large-v1.pt
├── sports_review_description_mxbai-embed-large-v1.pt
├── toys_review_description_mxbai-embed-large-v1.pt
└── yelp_review_description_mxbai-embed-large-v1.pt
```

각 파일은 `torch.Tensor` 형태로 저장된 아이템 임베딩:
- Shape: `[num_items, embedding_dim]`
- dtype: `torch.float32`

### 3. Ray 클러스터 초기화 (선택사항)

단일 머신:
```bash
ray start --head --num-gpus=2
```

멀티 노드:
```bash
# Head 노드
ray start --head --port=6379

# Worker 노드
ray start --address='HEAD_IP:6379'
```

---

## 🚀 사용 방법

### Step 1: RetrievalService 시작

터미널 1에서 실행:

```bash
# GPU 0에서 실행
./runs/run_retrieval.sh

# 또는 GPU 1에서 실행
CUDA_VISIBLE_DEVICES=1 python3 src/retrieval_service.py \
    --emb_model_name "mixedbread-ai/mxbai-embed-large-v1" \
    --emb_type "review_description" \
    --datasets beauty \
    --actor_name "RetrievalService" \
    --namespace "rl4rec" \
    --num_gpus 1.0
```

출력 예시:
```
✓ Loaded 1 dataset(s) on cuda
  Loaded dataset 'beauty': torch.Size([20000, 1024])
✓ DB Server is up and running. Waiting for learners...
```

### Step 2: RL Training 시작

터미널 2에서 실행:

```bash
# 기본 학습
./runs/run_rl_train.sh

# 빠른 테스트 (Dummy 모드)
./runs/run_rl_train_dummy.sh

# Yelp 데이터셋
./runs/run_rl_train_yelp.sh
```

출력 예시:
```
================================================================================
🚀 Starting RL Training
================================================================================
Step      0 | Loss: 0.5234 | Reward: 0.6543 (max: 0.8912, min: 0.4321)
Step     10 | Loss: 0.4876 | Reward: 0.6721 (max: 0.9034, min: 0.4567)
Step     20 | Loss: 0.4532 | Reward: 0.6892 (max: 0.9123, min: 0.4789)
...
✓ Checkpoint saved: checkpoints/beauty_rl/checkpoint_step_100.pt
```

### Step 3: 체크포인트 확인

```bash
ls -lh checkpoints/beauty_rl/
```

출력:
```
checkpoint_step_100.pt
checkpoint_step_200.pt
checkpoint_final.pt
training_log.jsonl
```

---

## 📁 파일 설명

### 코어 파일

#### `src/rec_model.py`

Policy Model 정의:

- **RecPolicy**: LLM 기반 Policy Model
  - `build_prompts()`: 사용자 히스토리 → 프롬프트 변환
  - `generate()`: 프롬프트 → 추천 텍스트 생성
  - `compute_loss()`: Policy Gradient Loss 계산
  - `update()`: 모델 파라미터 업데이트

- **DummyRecPolicy**: 테스트용 간단한 모델

사용 예시:
```python
from rec_model import RecPolicy

policy = RecPolicy(model_name="gpt2", device="cuda")

# 프롬프트 생성
user_histories = [["Product A", "Product B"], ["Product C"]]
prompts = policy.build_prompts(user_histories)

# 텍스트 생성
texts = policy.generate(prompts)

# 업데이트
rewards = torch.tensor([0.8, 0.6])
stats = policy.update(prompts, texts, rewards)
```

#### `src/rl_train.py`

학습 스크립트:

- **RLTrainer**: 전체 학습 프로세스 관리
  - `train_step()`: 한 스텝 학습
  - `compute_rewards()`: RetrievalService 통해 보상 계산
  - `train()`: 전체 학습 루프

직접 실행:
```bash
python3 src/rl_train.py \
    --dataset_name beauty \
    --num_steps 1000 \
    --batch_size 16 \
    --learning_rate 1e-5
```

#### `src/retrieval_service.py`

임베딩 및 스코어 계산 서비스:

- **RetrievalService**: Ray Actor
  - `calculate_reward(texts, dataset_name)`: 전체 스코어 배열 반환
  - Shape: `[len(texts), num_items]`

### 실행 스크립트

| 스크립트 | 설명 |
|---------|------|
| `runs/run_retrieval.sh` | RetrievalService 시작 (Beauty 데이터셋) |
| `runs/run_rl_train.sh` | RL 학습 시작 (Beauty, 1000 steps) |
| `runs/run_rl_train_dummy.sh` | 빠른 테스트용 Dummy 모드 |
| `runs/run_rl_train_yelp.sh` | Yelp 데이터셋 학습 |

---

## 🎨 커스터마이징

### 1. 새로운 데이터셋 추가

```bash
# 1. 아이템 임베딩 생성 및 저장
python3 scripts/generate_embeddings.py \
    --dataset toys \
    --emb_model "mixedbread-ai/mxbai-embed-large-v1"

# 2. RetrievalService 시작
python3 src/retrieval_service.py \
    --datasets toys \
    --emb_type item

# 3. 학습 시작
python3 src/rl_train.py --dataset_name toys
```

### 2. 다른 Policy Model 사용

```python
# src/rl_train.py 실행 시
python3 src/rl_train.py \
    --policy_model "meta-llama/Llama-2-7b-hf" \
    --learning_rate 1e-6
```

### 3. 보상 함수 변경

`src/rl_train.py`의 `compute_rewards()` 메서드 수정:

```python
def compute_rewards(self, generated_texts, dataset_name):
    scores = ray.get(
        self.retrieval_service.calculate_reward.remote(
            generated_texts, dataset_name
        )
    )
    
    # 커스텀 보상 계산
    # 예: Top-5 평균 + Diversity penalty
    top_5 = torch.topk(scores, k=5, dim=1).values
    avg_score = top_5.mean(dim=1)
    diversity = top_5.std(dim=1)
    rewards = avg_score + 0.1 * diversity
    
    return rewards
```

### 4. 프롬프트 템플릿 변경

`src/rec_model.py`의 `build_prompts()` 메서드 수정:

```python
def build_prompts(self, user_histories):
    prompts = []
    for history in user_histories:
        prompt = f"""
        User Profile:
        - Recent purchases: {', '.join(history[-5:])}
        - Preferences: High-quality products
        
        Generate a compelling description for the next recommended product:
        """
        prompts.append(prompt)
    return prompts
```

---

## 📊 로그 분석

학습 로그는 JSONL 형식으로 저장:

```python
import json
import pandas as pd

# 로그 로드
logs = []
with open("checkpoints/beauty_rl/training_log.jsonl") as f:
    for line in f:
        logs.append(json.loads(line))

df = pd.DataFrame(logs)

# 분석
print(df[["step", "loss", "avg_reward"]].describe())

# 시각화
import matplotlib.pyplot as plt
plt.plot(df["step"], df["avg_reward"])
plt.xlabel("Step")
plt.ylabel("Average Reward")
plt.show()
```

---

## 🐛 트러블슈팅

### RetrievalService를 찾을 수 없음

```
❌ Error: RetrievalService not found!
```

**해결**: 터미널 1에서 RetrievalService가 실행 중인지 확인

```bash
ray list actors --namespace rl4rec
```

### GPU 메모리 부족

```
CUDA out of memory
```

**해결**:
1. Batch size 감소: `--batch_size 8`
2. GPU memory utilization 조정: `gpu_memory_utilization=0.6`
3. 모델 크기 감소: `--policy_model gpt2` (대신 gpt2-medium 사용 안 함)

### 학습이 너무 느림

**해결**:
1. Dummy 모드로 테스트: `--use_dummy`
2. 더 작은 모델 사용
3. GPU 개수 확인: `ray status`

---

## 📚 참고 자료

- [Ray Documentation](https://docs.ray.io/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)

---

## 📝 TODO

- [ ] 실제 사용자 데이터 로더 구현
- [ ] Wandb/TensorBoard 로깅 추가
- [ ] Multi-task learning 지원
- [ ] Distributed training 지원
- [ ] A/B Testing 프레임워크

---

**Happy Training! 🚀**

