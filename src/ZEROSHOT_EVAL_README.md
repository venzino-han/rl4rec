# Zeroshot Evaluation with Multiple Rollouts

이 문서는 `zeroshot_eval.py` 스크립트의 사용법을 설명합니다.

## 개요

`zeroshot_eval.py`는 모델 학습 없이 제로샷 평가를 수행하는 스크립트입니다. 주요 특징은 다음과 같습니다:

- ✅ **Multiple Rollouts**: 각 프롬프트에 대해 k개의 다른 추론 결과 생성
- ✅ **Trigger Items**: 유저의 핵심 아이템을 강조하여 프롬프트 품질 향상
- ✅ **Temperature Control**: 샘플링 temperature 조절 (기본값: 0.6)
- ✅ **CSV Export**: 평가 결과를 CSV 형태로 저장 (rollout_index 포함)
- ✅ **Multiple Evaluations**: 각 rollout별로 독립적인 평가 수행

## 주요 기능

### 1. Multiple Rollouts
각 프롬프트에 대해 여러 개의 다른 응답을 생성하여 모델의 다양성과 안정성을 평가할 수 있습니다.

```bash
--num_rollouts 5  # 프롬프트당 5개의 응답 생성
```

### 2. Trigger Items
`trigger_items_from_sequential`에서 로드한 트리거 아이템을 프롬프트에 강조하여 추가합니다.

```bash
--use_trigger_items
--trigger_items_dir "sasrec_results/trigger_items_from_sequential"
--trigger_emphasis_text "This item was particularly influential in shaping the user's preferences."
```

프롬프트에서 트리거 아이템 다음에 강조 텍스트가 추가됩니다:
```
Item Title: Product Name
Brand: Brand Name
**Note:** This item was particularly influential in shaping the user's preferences.
```

### 3. Sampling Parameters
생성 다양성을 제어하는 파라미터들:

```bash
--temperature 0.6  # 샘플링 온도 (높을수록 다양함)
--top_p 0.9        # Top-p (nucleus) 샘플링
--top_k -1         # Top-k 샘플링 (-1이면 비활성화)
```

## 사용법

### 기본 사용

```bash
python3 src/zeroshot_eval.py \
    --data_name beauty \
    --model_name google/gemma-3-1b-it \
    --num_rollouts 3 \
    --temperature 0.6 \
    --use_trigger_items \
    --output_dir results
```

### 스크립트 사용

미리 만들어진 실행 스크립트를 사용할 수 있습니다:

```bash
# Beauty dataset (3 rollouts)
bash runs/4_zeroshot_eval_rollouts.sh

# Toys dataset (5 rollouts)
bash runs/4_zeroshot_eval_rollouts_toys.sh
```

### 체크포인트 사용

학습된 모델 체크포인트를 평가하려면:

```bash
python3 src/zeroshot_eval.py \
    --data_name beauty \
    --checkpoint_dir "models/sft_beauty_beauty_gemma-3-1b-it" \
    --num_rollouts 5 \
    --use_trigger_items
```

## 출력 파일

### 1. Generation Results CSV
`{output_dir}/zeroshot_{data_name}_trigger_generations.csv`

각 생성 결과를 포함합니다:

| Column | Description |
|--------|-------------|
| `user_id` | 유저 ID |
| `rollout_index` | Rollout 인덱스 (0부터 시작) |
| `generated_text` | 생성된 텍스트 |
| `prompt` | 입력 프롬프트 |

**예시:**
```csv
user_id,rollout_index,generated_text,prompt
1,0,"Product A is recommended...","<start_of_turn>user\n..."
1,1,"I suggest Product B...","<start_of_turn>user\n..."
1,2,"Consider Product C...","<start_of_turn>user\n..."
```

### 2. Evaluation Results CSV
`{output_dir}/zeroshot_{data_name}_trigger_eval.csv`

각 rollout별 평가 메트릭을 포함합니다:

| Column | Description |
|--------|-------------|
| `rollout_index` | Rollout 인덱스 |
| `hit@5` | Hit@5 메트릭 |
| `hit@10` | Hit@10 메트릭 |
| `hit@20` | Hit@20 메트릭 |
| `ndcg@5` | NDCG@5 메트릭 |
| `ndcg@10` | NDCG@10 메트릭 |
| `ndcg@20` | NDCG@20 메트릭 |
| `mrr` | Mean Reciprocal Rank |

**예시:**
```csv
rollout_index,hit@5,hit@10,hit@20,ndcg@5,ndcg@10,ndcg@20,mrr
0,0.1234,0.2345,0.3456,0.0876,0.1234,0.1567,0.0987
1,0.1345,0.2456,0.3567,0.0987,0.1345,0.1678,0.1098
2,0.1456,0.2567,0.3678,0.1098,0.1456,0.1789,0.1209
```

## 주요 파라미터

### Dataset & Model
- `--data_name`: 데이터셋 이름 (beauty, toys, sports, yelp)
- `--model_name`: 베이스 모델 이름
- `--checkpoint_dir`: 학습된 모델 체크포인트 경로 (선택사항)

### Rollout Settings
- `--num_rollouts`: 프롬프트당 생성할 rollout 개수 (기본: 1)
- `--temperature`: 샘플링 온도 (기본: 0.6)
- `--top_p`: Top-p 샘플링 파라미터 (기본: 0.9)
- `--top_k`: Top-k 샘플링 파라미터 (기본: -1, 비활성화)
- `--max_tokens`: 최대 생성 토큰 수 (기본: 128)

### Prompt Settings
- `--prompt_type`: 프롬프트 템플릿 타입 (기본: seq_rec)
- `--max_history_len`: 최대 히스토리 길이 (기본: 8)
- `--history_text_max_length`: 히스토리 텍스트 최대 단어 수 (기본: 128)
- `--use_brand`: 브랜드 포함 여부
- `--use_category`: 카테고리 포함 여부
- `--use_date`: 날짜 포함 여부
- `--use_last_item`: 마지막 아이템 강조 여부

### Trigger Items
- `--use_trigger_items`: 트리거 아이템 사용 여부
- `--trigger_items_dir`: 트리거 아이템 디렉토리
- `--trigger_emphasis_text`: 트리거 아이템 강조 텍스트

### Evaluation
- `--emb_model_name`: 임베딩 모델 이름
- `--emb_type`: 임베딩 타입
- `--eval_emb_max_length`: 임베딩 최대 길이
- `--eval_emb_batch_size`: 임베딩 배치 크기
- `--eval_samples`: 평가 샘플 수

### Output
- `--output_dir`: 결과 저장 디렉토리 (기본: results)
- `--seed`: 랜덤 시드 (기본: 42)

## 출력 예시

### Console Output

```
============================================================
🎯 Zeroshot Evaluation with Multiple Rollouts
============================================================
...
============================================================
🎲 Generating 5 rollout(s) per prompt
============================================================
  Total prompts: 22363
  Temperature: 0.6
  Max tokens: 128
  Top-p: 0.9
  Top-k: -1
============================================================

🔄 Generating responses...
Processing outputs: 100%|████████████| 22363/22363
✓ Generated 111815 total outputs (22363 prompts × 5 rollouts)

============================================================
📊 Evaluating Rollouts
============================================================

--- Evaluating Rollout 0 ---
  Evaluating 22363 samples for rollout 0...
  Rollout 0 results:
    hit@5: 0.1234
    hit@10: 0.2345
    ndcg@10: 0.1234
    mrr: 0.0987

--- Evaluating Rollout 1 ---
...

============================================================
📈 Summary Statistics Across Rollouts
============================================================
hit@5               : 0.1300 ± 0.0050
hit@10              : 0.2400 ± 0.0080
hit@20              : 0.3500 ± 0.0100
ndcg@5              : 0.0950 ± 0.0030
ndcg@10             : 0.1300 ± 0.0040
ndcg@20             : 0.1650 ± 0.0050
mrr                 : 0.1050 ± 0.0035
============================================================

💾 Saved generation results to: results/zeroshot_beauty_trigger_generations.csv
💾 Saved evaluation results to: results/zeroshot_beauty_trigger_eval.csv
```

## 활용 예시

### 1. 다양성 분석
여러 rollout을 생성하여 모델의 생성 다양성을 분석:

```bash
python3 src/zeroshot_eval.py \
    --data_name beauty \
    --num_rollouts 10 \
    --temperature 0.8 \
    --use_trigger_items
```

### 2. Temperature 실험
다양한 temperature 값으로 실험:

```bash
for temp in 0.3 0.6 0.9; do
    python3 src/zeroshot_eval.py \
        --data_name beauty \
        --num_rollouts 5 \
        --temperature $temp \
        --use_trigger_items \
        --output_dir "results/temp_${temp}"
done
```

### 3. Trigger Items 효과 비교
Trigger items 사용 여부에 따른 성능 비교:

```bash
# Without trigger items
python3 src/zeroshot_eval.py \
    --data_name beauty \
    --num_rollouts 5 \
    --output_dir results/no_trigger

# With trigger items
python3 src/zeroshot_eval.py \
    --data_name beauty \
    --num_rollouts 5 \
    --use_trigger_items \
    --output_dir results/with_trigger
```

## 참고 사항

1. **메모리 사용량**: num_rollouts가 클수록 메모리 사용량이 증가합니다
2. **실행 시간**: 각 rollout은 독립적으로 평가되므로 num_rollouts에 비례하여 시간이 증가합니다
3. **Deterministic vs Stochastic**: temperature=0.0으로 설정하면 deterministic 생성이 가능합니다
4. **CSV 파일**: 날짜가 파일명에 포함되지 않으므로 덮어쓰기에 주의하세요

## 문제 해결

### OOM (Out of Memory) 에러
```bash
--gpu_memory_utilization 0.8  # GPU 메모리 사용률 줄이기
--num_rollouts 3              # Rollout 개수 줄이기
```

### 평가 속도 개선
```bash
--eval_emb_batch_size 1024    # 임베딩 배치 크기 증가
--eval_samples 10000          # 평가 샘플 수 제한
```

## 관련 파일

- `src/zeroshot_eval.py`: 메인 스크립트
- `runs/4_zeroshot_eval_rollouts.sh`: Beauty dataset 실행 스크립트
- `runs/4_zeroshot_eval_rollouts_toys.sh`: Toys dataset 실행 스크립트
- `src/find_trigger_items_from_sequential.py`: Trigger items 생성 스크립트
