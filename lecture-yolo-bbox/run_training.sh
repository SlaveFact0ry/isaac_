#!/bin/bash

echo "======================================"
echo "실습 5: YOLOv11 바운딩박스 전이학습"
echo "Isaac Sim 합성 데이터 활용"
echo "======================================"
echo ""
echo "실습 목표:"
echo "- Isaac Sim 합성 데이터를 YOLO 형식으로 변환"
echo "- YOLOv11n 모델 전이학습"
echo "- 학습 성능 평가 및 시각화"
echo ""
echo "======================================"

# 현재 디렉토리 저장
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 기본값 설정
CONVERT_ONLY=""
TRAIN_ONLY=""
INFERENCE_ONLY=""
EPOCHS=50
BATCH_SIZE=16
IMGSZ=640
WEIGHTS=""

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --convert-only)
            CONVERT_ONLY="--convert-only"
            echo "데이터 변환만 수행합니다."
            shift
            ;;
        --train-only)
            TRAIN_ONLY="--train-only"
            echo "학습만 수행합니다."
            shift
            ;;
        --inference-only)
            INFERENCE_ONLY="--inference-only"
            echo "추론만 수행합니다."
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            echo "에폭 수: $EPOCHS"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            echo "배치 크기: $BATCH_SIZE"
            shift 2
            ;;
        --imgsz)
            IMGSZ="$2"
            echo "이미지 크기: $IMGSZ"
            shift 2
            ;;
        --weights)
            WEIGHTS="--weights $2"
            echo "가중치 파일: $2"
            shift 2
            ;;
        --help)
            echo ""
            echo "사용법: $0 [옵션]"
            echo ""
            echo "옵션:"
            echo "  --convert-only      데이터 변환만 수행"
            echo "  --train-only        학습만 수행 (데이터 변환 건너뜀)"
            echo "  --inference-only    추론만 수행"
            echo "  --epochs N          학습 에폭 수 (기본: 50)"
            echo "  --batch-size N      배치 크기 (기본: 16)"
            echo "  --imgsz N           이미지 크기 (기본: 640)"
            echo "  --weights PATH      추론용 가중치 파일 경로"
            echo "  --help              도움말 표시"
            echo ""
            echo "예시:"
            echo "  $0                          # 전체 파이프라인 실행"
            echo "  $0 --convert-only           # 데이터 변환만"
            echo "  $0 --train-only --epochs 30 # 학습만 (30 에폭)"
            echo "  $0 --inference-only --weights runs/detect/yolo11_isaac_*/weights/best.pt"
            echo ""
            exit 0
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            echo "도움말을 보려면: $0 --help"
            exit 1
            ;;
    esac
done

# Python 환경 확인
echo ""
echo "Python 환경 확인 중..."
python3 -c "import ultralytics; print(f'✓ Ultralytics YOLO v{ultralytics.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠ ultralytics가 설치되지 않았습니다."
    echo "설치 명령: pip install ultralytics torch torchvision"
    exit 1
fi

python3 -c "import cv2; print(f'✓ OpenCV v{cv2.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠ OpenCV가 설치되지 않았습니다."
    echo "설치 명령: pip install opencv-python"
    exit 1
fi

# 실행 명령 구성
CMD="python3 yolo_bbox_training.py"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --imgsz $IMGSZ"
CMD="$CMD $CONVERT_ONLY $TRAIN_ONLY $INFERENCE_ONLY $WEIGHTS"

# 실행
echo ""
echo "실행 명령: $CMD"
echo ""
echo "======================================"
echo ""

# 시간 측정 시작
START_TIME=$(date +%s)

# 스크립트 실행
$CMD

# 실행 결과 확인
RESULT=$?

# 시간 측정 종료
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

echo ""
echo "======================================"
if [ $RESULT -eq 0 ]; then
    echo "✓ 실습 완료!"
    echo "소요 시간: $((ELAPSED_TIME / 60))분 $((ELAPSED_TIME % 60))초"
    
    # 결과 파일 확인
    if [ -f "detection_result.jpg" ]; then
        echo ""
        echo "생성된 파일:"
        echo "  - detection_result.jpg (추론 결과)"
    fi
    
    if [ -d "yolo_dataset" ]; then
        echo "  - yolo_dataset/ (변환된 데이터셋)"
    fi
    
    if [ -d "runs/detect" ]; then
        echo "  - runs/detect/ (학습 결과)"
    fi
else
    echo "⚠ 실습 중 오류가 발생했습니다."
    echo "위의 오류 메시지를 확인하세요."
fi
echo "======================================" 