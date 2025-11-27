#!/bin/bash

echo "======================================"
echo "Isaac Sim Replicator Lecture 04"
echo "Advanced Robot Vision Dataset"
echo "======================================"
echo ""
echo "실습 목표:"
echo "- YCB 객체를 활용한 로봇 비전 데이터셋 생성"
echo "- 다중 카메라 시스템 구축"
echo "- 고급 도메인 랜덤화 구현"
echo "- 3D 어노테이션 생성"
echo ""
echo "======================================"

# 현재 디렉토리 저장
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


# Isaac Sim 경로 확인
ISAAC_SIM_PATH="/home/int/isaacsim/_build/linux-x86_64/release/"  # 본인의 isaac-sim 경로로 변경 필요
if [ ! -d "$ISAAC_SIM_PATH" ]; then
    echo "Error: Isaac Sim이 $ISAAC_SIM_PATH 에 설치되어 있지 않습니다."
    exit 1
fi

# Python 스크립트 실행
echo "실습을 시작합니다..."
echo "GUI 모드로 실행 중... (헤드리스 모드: --headless 추가)"
echo ""

cd "$SCRIPT_DIR"

# 기본값 설정
HEADLESS=""
NUM_FRAMES=100

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --headless)
            HEADLESS="--headless"
            echo "헤드리스 모드로 실행합니다."
            shift
            ;;
        --num_frames)
            NUM_FRAMES="$2"
            echo "프레임 수: $NUM_FRAMES"
            shift 2
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            echo "사용법: $0 [--headless] [--num_frames N]"
            exit 1
            ;;
    esac
done

# 실행
echo "실행 중... (프레임: $NUM_FRAMES)"
$ISAAC_SIM_PATH/python.sh lecture_04_replicator_advanced.py $HEADLESS --num_frames $NUM_FRAMES

echo ""
echo "======================================"
echo "실습 완료!"
echo "생성된 데이터: ${SCRIPT_DIR}/replicator_output/advanced_dataset"
echo "======================================"