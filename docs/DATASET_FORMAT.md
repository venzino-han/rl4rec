# 데이터셋 형식 가이드

## 개요

GRPO 학습을 위한 데이터셋 형식과 준비 방법을 설명합니다.

## 필요한 파일

### 1. Sequential Data (`sequential_data.txt`)

사용자의 아이템 시퀀스 및 타겟 아이템 정보

**경로:** `data/{data_name}/sequential_data.txt`

**형식:**
```
user_id item1 item2 item3 ... itemN target_item
```

**예제:**
```
1 101 102 103 104 105
2 201 202 203 204 205
3 301 302 303 304 305
```

- 첫 번째 숫자: 사용자 ID
- 중간 숫자들: 사용자의 구매 히스토리 (아이템 ID)
- 마지막 숫자: 타겟 아이템 (예측할 아이템)

### 2. Item Metadata (선택적)

아이템의 메타데이터 정보

**경로 (우선순위 순):**
1. `data/{data_name}/item_metadata.json`
2. `data/{data_name}/item_metadata.pkl`
3. `data/{data_name}/metadata.json`
4. `data_processed/{data_name}_item_metadata.json`

**형식 (JSON):**
```json
{
  "101": {
    "title": "Wireless Bluetooth Headphones",
    "brand": "TechBrand",
    "category": "Electronics > Audio > Headphones",
    "description": "High-quality wireless headphones with noise cancellation...",
    "features": "- Bluetooth 5.0\n- 30-hour battery life\n- Active noise cancellation"
  },
  "102": {
    "title": "Running Shoes",
    "brand": "SportsBrand",
    "category": "Sports > Footwear > Running",
    "description": "Lightweight running shoes for marathon training...",
    "features": "- Breathable mesh\n- Cushioned sole\n- Arch support"
  }
}
```

**필수 필드:**
- `title`: 아이템 제목
- `brand`: 브랜드명
- `category`: 카테고리 (계층적 가능)

**선택적 필드:**
- `description`: 아이템 설명
- `features`: 아이템 특징 (줄바꿈으로 구분)

## 프롬프트 생성

### PromptGenerator 설정

```python
prompt_generator = PromptGenerator(
    item_metadata=item_metadata,
    use_brand=True,           # 브랜드 포함
    use_category=True,        # 카테고리 포함
    use_description=False,    # 설명 제외
    use_features=False,       # 특징 제외
    use_last_item=True,       # 마지막 아이템 강조
    max_history_len=5,        # 최대 5개 히스토리
)
```

### 생성된 프롬프트 예제

```
# User Purchase History

1. **Title:** `Wireless Bluetooth Headphones`
**Brand:** TechBrand
**Categories:** Electronics > Audio > Headphones
---
2. **Title:** `Running Shoes`
**Brand:** SportsBrand
**Categories:** Sports > Footwear > Running
---
3. **Title:** `Coffee Maker`
**Brand:** HomeBrand
**Categories:** Home & Kitchen > Appliances > Coffee Makers

`Coffee Maker` is the most recently purchased item.

# Task
Based on this user's purchase history, describe an ideal next product for this user:
```

## 데이터 분할

### 자동 분할

데이터셋은 자동으로 train/valid/test로 분할됩니다:

- **Train**: 80% (기본값)
- **Valid**: 10% (기본값)
- **Test**: 10% (나머지)

### 분할 설정

```bash
python src/grpo_train.py \
    --train_ratio 0.8 \
    --valid_ratio 0.1 \
    --seed 42
```

### 분할 로직

1. 모든 사용자 ID를 로드
2. 시드를 사용하여 셔플
3. 비율에 따라 분할
   - Train: 0 ~ train_ratio
   - Valid: train_ratio ~ train_ratio + valid_ratio
   - Test: train_ratio + valid_ratio ~ 1.0

## Beauty 데이터셋 예제

### 디렉토리 구조

```
data/
└── beauty/
    ├── sequential_data.txt           # 시퀀셜 데이터
    └── item_metadata.json            # 아이템 메타데이터

data_processed/
└── beauty_item_metadata.json         # 처리된 메타데이터 (대체 경로)
```

### sequential_data.txt

```
1 234 456 789 1234 5678
2 111 222 333 444 555
3 999 888 777 666 555
...
```

- 사용자 1: 234, 456, 789, 1234를 구매했고, 다음 타겟은 5678
- 사용자 2: 111, 222, 333, 444를 구매했고, 다음 타겟은 555

### item_metadata.json

```json
{
  "234": {
    "title": "Maybelline New York Fit Me Foundation",
    "brand": "Maybelline",
    "category": "Beauty > Makeup > Face > Foundation"
  },
  "456": {
    "title": "L'Oreal Paris Voluminous Mascara",
    "brand": "L'Oreal",
    "category": "Beauty > Makeup > Eyes > Mascara"
  }
}
```

## 메타데이터가 없는 경우

메타데이터 파일이 없어도 학습은 가능합니다:

1. **경고 메시지**: "Item metadata file not found. Using dummy metadata."
2. **Dummy 프롬프트 생성**: 간단한 아이템 ID 기반 프롬프트 사용
3. **학습 진행**: NDCG 리워드는 정상적으로 계산됨

## Dummy 데이터로 테스트

메타데이터 없이 빠르게 테스트:

```bash
python src/grpo_train.py \
    --use_dummy \
    --dummy_size 100
```

- 100개의 랜덤 사용자 생성
- 랜덤 아이템 시퀀스 생성
- 간단한 프롬프트 사용

## 데이터 준비 체크리스트

- [ ] `sequential_data.txt` 파일 준비
  - [ ] 각 줄: `user_id item1 item2 ... target`
  - [ ] 공백으로 구분
  - [ ] 모든 값은 정수
  
- [ ] `item_metadata.json` 파일 준비 (선택적)
  - [ ] 각 아이템 ID를 키로 사용
  - [ ] 최소한 `title`, `brand`, `category` 포함
  
- [ ] 파일 경로 확인
  - [ ] `data/{data_name}/` 디렉토리 존재
  - [ ] 파일 권한 확인

## 커스텀 데이터셋 사용

### 1. 데이터 준비

```bash
mkdir -p data/my_dataset
# sequential_data.txt와 item_metadata.json 준비
```

### 2. 학습 실행

```bash
python src/grpo_train.py \
    --data_name "my_dataset" \
    --data_dir "data" \
    --sequential_file "data/my_dataset/sequential_data.txt"
```

### 3. 메타데이터 위치 지정

메타데이터가 다른 위치에 있는 경우, `load_item_metadata()` 함수 수정:

```python
# grpo_train.py의 load_item_metadata 함수
possible_paths = [
    f"{data_dir}/{data_name}/item_metadata.json",
    f"my_custom_path/metadata.json",  # 추가
    ...
]
```

## 트러블슈팅

### 문제: "Item ID not found in item features"

**원인**: 메타데이터에 해당 아이템 ID가 없음

**해결**:
1. 메타데이터 파일에 모든 아이템 포함 확인
2. 또는 `_create_dummy_data`에서 에러를 스킵하도록 수정

### 문제: "No module named 'pickle'"

**원인**: pickle 모듈 import 실패

**해결**:
```python
import pickle  # grpo_train.py 상단에 추가
```

### 문제: 프롬프트가 너무 김

**해결**:
```bash
--max_history_len 3 \
--max_length 256
```

## 참고 자료

- [GRPO Training Guide](./GRPO_TRAINING_GUIDE.md)
- [Reward Function Guide](./TRL_TRAINING_GUIDE.md)



