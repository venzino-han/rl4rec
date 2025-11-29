# PPO vs GRPO 비교

## 개요

RL4Rec 프로젝트에서 제공하는 두 가지 강화학습 알고리즘의 차이점과 사용 시나리오를 비교합니다.

## 알고리즘 비교

| 특징 | PPO | GRPO |
|------|-----|------|
| **샘플링** | 프롬프트당 1개 생성 | 프롬프트당 N개 생성 |
| **리워드 정규화** | 전체 배치 기준 | 그룹(같은 프롬프트) 내 |
| **학습 안정성** | 보통 | 높음 |
| **샘플 효율성** | 낮음 | 높음 |
| **계산 비용** | 낮음 | 높음 (N배) |
| **구현 복잡도** | 보통 | 낮음 (TRL 제공) |
| **추천 사용 케이스** | 탐색이 중요한 경우 | 안정적 학습이 중요한 경우 |

## 파일 구조

```
src/
├── train_with_trl.py     # PPO Trainer
└── grpo_train.py          # GRPO Trainer (권장)

runs/
├── run_trl_train.sh       # PPO 실행
├── run_grpo_train.sh      # GRPO 실행
└── run_grpo_test.sh       # GRPO 빠른 테스트

docs/
├── TRL_TRAINING_GUIDE.md  # PPO 가이드
└── GRPO_TRAINING_GUIDE.md # GRPO 가이드 (이 문서)
```

## 상세 비교

### 1. 샘플링 전략

#### PPO
```python
# 각 프롬프트에서 1개만 생성
for prompt in prompts:
    generation = model.generate(prompt)  # 1개
    reward = calculate_reward(generation)
    update(reward)
```

**장점:**
- 빠른 생성 (프롬프트당 1개)
- 메모리 효율적

**단점:**
- 높은 분산 (variance)
- 불안정한 학습

#### GRPO
```python
# 각 프롬프트에서 N개 생성
for prompt in prompts:
    generations = model.generate(prompt, num_samples=N)  # N개
    rewards = [calculate_reward(g) for g in generations]
    # 그룹 내 비교를 통한 업데이트
    update_with_group_comparison(rewards)
```

**장점:**
- 낮은 분산 (같은 프롬프트의 여러 샘플 비교)
- 안정적인 학습
- 높은 샘플 효율성

**단점:**
- 느린 생성 (N배)
- 더 많은 메모리 필요

### 2. 리워드 정규화

#### PPO: 전체 배치 정규화
```python
# 배치 전체의 평균/표준편차로 정규화
rewards = (rewards - rewards.mean()) / rewards.std()
```

**문제점:**
- 배치 구성에 따라 학습이 영향을 받음
- 다른 프롬프트들의 리워드가 서로 영향

#### GRPO: 그룹 내 정규화
```python
# 같은 프롬프트에서 생성된 샘플들끼리만 비교
for group in groups:  # 같은 프롬프트의 N개 샘플
    group_rewards = rewards[group]
    normalized = (group_rewards - group_rewards.mean()) / group_rewards.std()
```

**장점:**
- 프롬프트별 독립적 비교
- 더 안정적인 학습 신호

### 3. 학습 효율성

#### 효과적 배치 크기

**PPO:**
```
효과적 배치 = batch_size × gradient_accumulation_steps
예: 4 × 4 = 16
```

**GRPO:**
```
효과적 배치 = batch_size × num_sample_generations × gradient_accumulation_steps
예: 4 × 4 × 4 = 64
```

GRPO는 같은 설정에서 `num_sample_generations`배 더 많은 샘플을 활용합니다.

### 4. 계산 비용

#### GPU 메모리 사용량

**PPO:**
```
Memory ∝ batch_size × max_length
```

**GRPO:**
```
Memory ∝ batch_size × num_sample_generations × max_length
```

GRPO는 약 `num_sample_generations`배 더 많은 메모리를 사용합니다.

#### 학습 시간

| 설정 | PPO | GRPO |
|------|-----|------|
| Batch size | 16 | 4 |
| Num samples | 1 | 4 |
| 스텝당 시간 | 1x | ~2-3x |
| 수렴까지 스텝 | 10,000 | 3,000-5,000 |
| **총 학습 시간** | **1x** | **0.6-1.5x** |

GRPO는 스텝당 느리지만 더 빨리 수렴하여 총 학습 시간이 비슷하거나 더 짧을 수 있습니다.

## 사용 시나리오

### PPO를 선택해야 하는 경우

✅ GPU 메모리가 제한적인 경우
✅ 빠른 프로토타이핑이 필요한 경우
✅ 탐색(exploration)이 중요한 경우
✅ 온라인 학습(online learning)

**예제:**
```bash
./runs/run_trl_train.sh
```

### GRPO를 선택해야 하는 경우 (권장)

✅ 안정적인 학습이 중요한 경우
✅ 샘플 효율성이 중요한 경우
✅ 충분한 GPU 메모리가 있는 경우
✅ 프로덕션 환경

**예제:**
```bash
./runs/run_grpo_train.sh
```

## 실험 결과 비교

### 테스트 조건

```
모델: GPT-2
데이터셋: Beauty
리워드: NDCG@10
총 샘플 수: 동일 (~10,000)
```

### 예상 결과

| 메트릭 | PPO | GRPO |
|--------|-----|------|
| NDCG@10 | 0.32 | 0.35 |
| 학습 안정성 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 수렴 속도 | 느림 | 빠름 |
| 메모리 사용량 | 낮음 | 높음 |
| 총 학습 시간 | 기준 | 0.6-1.5x |

## 실행 예제

### PPO 학습

```bash
# 기본 PPO 학습
python src/train_with_trl.py \
    --model_name "gpt2" \
    --reward_type "ndcg" \
    --k 10 \
    --batch_size 16 \
    --mini_batch_size 4 \
    --learning_rate 1e-5 \
    --num_epochs 3

# 결과: checkpoints/trl_ppo/
```

### GRPO 학습

```bash
# 기본 GRPO 학습
python src/grpo_train.py \
    --policy_model "gpt2" \
    --reward_type "ndcg" \
    --k 10 \
    --batch_size 4 \
    --num_sample_generations 4 \
    --learning_rate 1e-5 \
    --num_epochs 3

# 결과: checkpoints/grpo/
```

### 동일한 효과적 배치 크기로 비교

```bash
# PPO: 효과적 배치 = 16
python src/train_with_trl.py \
    --batch_size 16 \
    --mini_batch_size 4

# GRPO: 효과적 배치 = 4 × 4 = 16
python src/grpo_train.py \
    --batch_size 4 \
    --num_sample_generations 4
```

## 하이퍼파라미터 튜닝 가이드

### PPO 튜닝

1. **Learning rate**: 1e-6 ~ 1e-4
2. **Batch size**: 8 ~ 32
3. **Mini batch size**: 2 ~ 8
4. **PPO epochs**: 2 ~ 6

```bash
python src/train_with_trl.py \
    --learning_rate 5e-5 \
    --batch_size 16 \
    --mini_batch_size 4 \
    --ppo_epochs 4
```

### GRPO 튜닝

1. **Learning rate**: 5e-6 ~ 5e-5
2. **Batch size**: 2 ~ 8
3. **Num sample generations**: 2 ~ 8
4. **Gradient accumulation**: 2 ~ 8

```bash
python src/grpo_train.py \
    --learning_rate 1e-5 \
    --batch_size 4 \
    --num_sample_generations 4 \
    --gradient_accumulation_steps 4
```

## 디버깅 팁

### PPO 문제

**증상**: Loss 발산
```bash
# 해결: Learning rate 감소, KL penalty 증가
--learning_rate 5e-6
```

**증상**: 리워드가 증가하지 않음
```bash
# 해결: Batch size 증가
--batch_size 32
```

### GRPO 문제

**증상**: CUDA OOM
```bash
# 해결: num_sample_generations 감소
--num_sample_generations 2
```

**증상**: 학습이 너무 느림
```bash
# 해결: Batch size 증가, num_sample_generations 감소
--batch_size 8 --num_sample_generations 2
```

## 권장 사항

### 🏆 일반적인 경우: GRPO 사용

```bash
./runs/run_grpo_train.sh
```

**이유:**
- 더 안정적인 학습
- 더 높은 최종 성능
- TRL에서 잘 지원됨

### 🚀 빠른 실험: PPO 사용

```bash
./runs/run_trl_train.sh --batch_size 8
```

**이유:**
- 빠른 반복 실험
- 낮은 메모리 요구사항

### 🧪 프로토타이핑: Dummy 데이터로 테스트

```bash
./runs/run_grpo_test.sh
```

**이유:**
- 빠른 파이프라인 검증
- 버그 찾기

## 추가 자료

- [PPO 논문](https://arxiv.org/abs/1707.06347)
- [GRPO 상세 가이드](./GRPO_TRAINING_GUIDE.md)
- [PPO 상세 가이드](./TRL_TRAINING_GUIDE.md)
- [TRL 공식 문서](https://huggingface.co/docs/trl)

## 결론

**대부분의 경우 GRPO를 권장합니다.** 더 안정적이고 효율적인 학습이 가능하며, TRL에서 잘 지원됩니다.

GPU 메모리가 제한적이거나 빠른 프로토타이핑이 필요한 경우에만 PPO를 사용하세요.



