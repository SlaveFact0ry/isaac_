#!/bin/bash

# ============================================================================
# Isaac Sim Standalone 실습 2 실행 스크립트
# 인터랙티브 환경과 카메라 제어 실습
# ============================================================================

echo "=============================================================="
echo "   Isaac Sim Standalone 실습 2: 인터랙티브 환경과 카메라 제어"
echo "=============================================================="
echo ""
echo "실습 내용:"
echo "  1. 복잡한 씬 구성 (Sphere, Cylinder, Cone, Capsule)"
echo "  2. 재질 적용 (Metal, Wood, Plastic, Glass)"
echo "  3. 다중 카메라 (메인 카메라, 탑뷰 카메라)"
echo "  4. 애니메이션 (궤도, 바운싱, 카메라 회전)"
echo "  5. 물리 이벤트 (충격, 중력 반전)"
echo ""
echo "특별 이벤트:"
echo "  - Step 150: 충격 이벤트"
echo "  - Step 300: 중력 반전"
echo "  - Step 400: 중력 복원"
echo "  - Step 500-550: 슬로우 모션"
echo ""
echo "=============================================================="
echo ""

# 스크립트 경로 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_SIM_PATH="/home/wj/isaac-sim"  # 본인의 isaac-sim 경로로 변경 필요

# 스크립트 존재 확인
if [ ! -f "$SCRIPT_DIR/lecture_02_interactive_scene.py" ]; then
    echo "에러: Python 스크립트를 찾을 수 없습니다!"
    echo "경로: $SCRIPT_DIR/lecture_02_interactive_scene.py"
    exit 1
fi

# Isaac Sim Python 실행 파일 확인
if [ ! -f "$ISAAC_SIM_PATH/python.sh" ]; then
    echo "에러: Isaac Sim Python 실행 파일을 찾을 수 없습니다!"
    echo "경로: $ISAAC_SIM_PATH/python.sh"
    exit 1
fi

# 실행 옵션 파싱
HEADLESS_MODE=""
SIM_STEPS="600"
SAVE_IMAGES=""

# 커맨드 라인 인자 처리
while [[ $# -gt 0 ]]; do
    case $1 in
        --headless)
            HEADLESS_MODE="--headless"
            echo "► 헤드리스 모드로 실행합니다."
            shift
            ;;
        --steps)
            SIM_STEPS="$2"
            echo "► 시뮬레이션 스텝: $SIM_STEPS"
            shift 2
            ;;
        --save-images)
            SAVE_IMAGES="--save_images"
            echo "► 카메라 이미지를 저장합니다."
            shift
            ;;
        --help)
            echo "사용법: $0 [옵션]"
            echo ""
            echo "옵션:"
            echo "  --headless        GUI 없이 실행 (서버 환경)"
            echo "  --steps N         시뮬레이션 스텝 수 (기본값: 600)"
            echo "  --save-images     카메라 이미지 저장 (OpenCV 필요)"
            echo "  --help            도움말 표시"
            echo ""
            echo "예제:"
            echo "  $0                           # 기본 설정으로 실행"
            echo "  $0 --headless                # 헤드리스 모드"
            echo "  $0 --steps 1000              # 1000 스텝 실행"
            echo "  $0 --save-images             # 이미지 캡처 활성화"
            echo "  $0 --steps 300 --save-images # 300 스텝, 이미지 저장"
            exit 0
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            echo "--help 옵션으로 사용법을 확인하세요."
            exit 1
            ;;
    esac
done

# 디렉토리 이동
cd "$SCRIPT_DIR"

# 이미지 저장 디렉토리 생성 (필요한 경우)
if [ ! -z "$SAVE_IMAGES" ]; then
    IMAGE_DIR="$SCRIPT_DIR/camera_images"
    mkdir -p "$IMAGE_DIR"
    echo "► 이미지 저장 디렉토리: $IMAGE_DIR"
    
    # 기존 이미지 정리 (선택사항)
    if [ -d "$IMAGE_DIR" ] && [ "$(ls -A $IMAGE_DIR 2>/dev/null)" ]; then
        echo "  기존 이미지 파일을 정리하시겠습니까? (y/N)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            rm -f "$IMAGE_DIR"/*.png
            echo "  ✓ 기존 이미지 파일 삭제 완료"
        fi
    fi
fi


# OpenCV 확인 (이미지 저장 옵션이 있을 때만)
if [ ! -z "$SAVE_IMAGES" ]; then
    echo "OpenCV 라이브러리 확인 중..."
    $ISAAC_SIM_PATH/python.sh -c "import cv2; print('  ✓ OpenCV 버전:', cv2.__version__)" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "  ⚠ OpenCV를 찾을 수 없습니다. 이미지 저장이 비활성화됩니다."
        echo "  설치하려면: $ISAAC_SIM_PATH/python.sh -m pip install opencv-python"
    fi
    echo ""
fi

echo "실행 중... (Ctrl+C로 중단 가능)"
echo "=============================================================="
echo ""

# Isaac Sim 실행
$ISAAC_SIM_PATH/python.sh lecture_02_interactive_scene.py \
    $HEADLESS_MODE \
    --simulation_steps $SIM_STEPS \
    $SAVE_IMAGES \


# 종료 상태 확인
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=============================================================="
    echo "✓ 실습이 성공적으로 완료되었습니다!"
    
    # 이미지 저장 확인
    if [ ! -z "$SAVE_IMAGES" ] && [ -d "$IMAGE_DIR" ]; then
        IMAGE_COUNT=$(find "$IMAGE_DIR" -name "*.png" 2>/dev/null | wc -l)
        if [ $IMAGE_COUNT -gt 0 ]; then
            echo ""
            echo "📸 저장된 이미지: $IMAGE_COUNT 개"
            echo "   위치: $IMAGE_DIR"
        fi
    fi
    
    echo "=============================================================="
else
    echo ""
    echo "=============================================================="
    echo "✗ 실습 중 오류가 발생했습니다. (종료 코드: $EXIT_CODE)"
    echo "=============================================================="
    exit $EXIT_CODE
fi