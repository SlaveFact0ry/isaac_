#!/bin/bash

echo "======================================"
echo "Isaac Sim Replicator Lecture 03"
echo "Basic Synthetic Data Generation"
echo "======================================"
echo ""
echo "실습 목표:"
echo "- Replicator 기본 개념 이해"
echo "- 도메인 랜덤화 구현"
echo "- 합성 데이터 생성 파이프라인 구축"
echo ""
echo "======================================"

# 현재 디렉토리 저장
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


# Isaac Sim 경로 확인
ISAAC_SIM_PATH="/home/wj/isaac-sim"  # 본인의 isaac-sim 경로로 변경 필요
if [ ! -d "$ISAAC_SIM_PATH" ]; then
    echo "Error: Isaac Sim이 $ISAAC_SIM_PATH 에 설치되어 있지 않습니다."
    exit 1
fi

# Python 스크립트 실행
echo "실습을 시작합니다..."
echo "GUI 모드로 실행 중... (헤드리스 모드: --headless 추가)"
echo ""

cd "$SCRIPT_DIR"

# 실행 옵션 설정
if [ "$1" == "--headless" ]; then
    echo "헤드리스 모드로 실행합니다."
    $ISAAC_SIM_PATH/python.sh lecture_03_replicator_basics.py --headless --num_frames 100
else
    echo "GUI 모드로 실행합니다."
    $ISAAC_SIM_PATH/python.sh lecture_03_replicator_basics.py --num_frames 100
fi

echo ""
echo "======================================"
echo "실습 완료!"
echo "생성된 데이터: ${SCRIPT_DIR}/replicator_output/basic_dataset"
echo "======================================"