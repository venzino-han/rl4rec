#!/bin/bash
# 
# 기존 스크립트에 Device Tracker를 추가하는 템플릿
# 
# 사용법: 이 템플릿을 참고하여 기존 스크립트를 수정하세요.
#

# ============================================================
# STEP 1: 스크립트 시작 부분에 추가
# ============================================================

# Device 설정 (기존에 있으면 그대로 사용)
device=0  

# Tracker 변수 정의 (이 한 줄만 추가)
TRACKER="python3 utils/device_tracker.py"

# 선택사항: 시작 시 현재 상태 표시
echo "=========================================="
echo "Starting experiment"
echo "=========================================="
$TRACKER show-simple


# ============================================================
# STEP 2: 각 실험/학습 시작 직전에 추가
# ============================================================

# RUN_NAME 정의 (기존에 있으면 그대로 사용)
RUN_NAME="your_experiment_name"

# Device 할당 (이 한 줄만 추가)
$TRACKER allocate $device "$RUN_NAME"

# 여기부터 실제 학습 코드
CUDA_VISIBLE_DEVICES=$device python3 src/train.py \
    --run_name $RUN_NAME \
    ...


# ============================================================
# STEP 3: 각 실험/학습 종료 직후에 추가
# ============================================================

# Device 해제 (이 한 줄만 추가)
$TRACKER free $device

# 선택사항: 완료 메시지와 함께 상태 표시
echo "✅ Completed: $RUN_NAME"
$TRACKER show-simple


# ============================================================
# STEP 4: 스크립트 종료 부분에 추가 (선택사항)
# ============================================================

echo "=========================================="
echo "✅ All experiments completed!"
echo "=========================================="
$TRACKER show


# ============================================================
# 고급: 에러 발생 시에도 자동으로 device 해제
# ============================================================

# 스크립트 시작 부분에 trap 추가 (권장)
trap '$TRACKER free $device' EXIT ERR INT TERM

# 이렇게 하면 스크립트가 어떤 이유로든 종료될 때
# 자동으로 device를 해제합니다.


# ============================================================
# 실제 적용 예시
# ============================================================

#!/bin/bash

# 기본 설정
device=0
dataset_names=(beauty toys sports yelp)
TRACKER="python3 utils/device_tracker.py"

# 에러 시 자동 해제
trap '$TRACKER free $device' EXIT

# 시작
$TRACKER show-simple

# 실험 루프
for dataset in ${dataset_names[@]}; do
for lr in 1e-6 2e-6 5e-6; do
    RUN_NAME="my_exp_${dataset}_lr${lr}"
    
    # === 추가: Device 할당 ===
    $TRACKER allocate $device "$RUN_NAME"
    
    # 기존 학습 코드
    CUDA_VISIBLE_DEVICES=$device python3 src/train.py \
        --run_name $RUN_NAME \
        --data_name $dataset \
        --learning_rate $lr \
        ...
    
    # === 추가: Device 해제 ===
    $TRACKER free $device
    
    echo "✅ Completed: $RUN_NAME"
done
done

# 종료
$TRACKER show


