#!/bin/bash

# ============================================================================
# Isaac Sim Standalone 실습 1 실행 스크립트
# 기본 환경 구성 실습
# ============================================================================

echo "=============================================================="
echo "   Isaac Sim Standalone 실습 1: 기본 환경 구성"
echo "=============================================================="
echo ""
echo "실습 내용:"
echo "  1. SimulationApp 초기화"
echo "  2. 기본 객체 생성 및 배치"
echo "  3. 조명 설정 (Dome, Point, Distant Light)"
echo "  4. 물리 속성 설정"
echo "  5. 시뮬레이션 제어"
echo ""
echo "특별 이벤트:"
echo "  - Step 100: 도미노 효과 시작"
echo "  - Step 200: 랜덤 충격 이벤트"
echo "  - 50 스텝마다: Visual Cube 색상 변경"
echo ""
echo "=============================================================="
echo ""

# 스크립트 경로 설정 (현재 스크립트가 있는 디렉토리를 기준으로)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_SIM_PATH="/home/wj/isaac-sim"  # 본인의 isaac-sim 경로로 변경 필요

# 스크립트 존재 확인
if [ ! -f "$SCRIPT_DIR/lecture_01_basic_environment.py" ]; then
    echo "에러: Python 스크립트를 찾을 수 없습니다!"
    echo "경로: $SCRIPT_DIR/lecture_01_basic_environment.py"
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
NUM_CUBES="5"
SIM_STEPS="500"

# 커맨드 라인 인자 처리
while [[ $# -gt 0 ]]; do
    case $1 in
        --headless)
            HEADLESS_MODE="--headless"
            echo "► 헤드리스 모드로 실행합니다."
            shift
            ;;
        --num-cubes)
            NUM_CUBES="$2"
            echo "► 큐브 개수: $NUM_CUBES"
            shift 2
            ;;
        --steps)
            SIM_STEPS="$2"
            echo "► 시뮬레이션 스텝: $SIM_STEPS"
            shift 2
            ;;
        --help)
            echo "사용법: $0 [옵션]"
            echo ""
            echo "옵션:"
            echo "  --headless        GUI 없이 실행 (서버 환경)"
            echo "  --num-cubes N     생성할 큐브 개수 (기본값: 5)"
            echo "  --steps N         시뮬레이션 스텝 수 (기본값: 500)"
            echo "  --help            도움말 표시"
            echo ""
            echo "예제:"
            echo "  $0                           # 기본 설정으로 실행"
            echo "  $0 --headless                # 헤드리스 모드"
            echo "  $0 --num-cubes 10 --steps 1000  # 10개 큐브, 1000 스텝"
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

echo ""
echo "실행 중... (Ctrl+C로 중단 가능)"
echo "=============================================================="
echo ""

# Isaac Sim 실행
$ISAAC_SIM_PATH/python.sh lecture_01_basic_environment.py \
    $HEADLESS_MODE \
    --num_cubes $NUM_CUBES \
    --simulation_steps $SIM_STEPS

# 종료 상태 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================================="
    echo "✓ 실습이 성공적으로 완료되었습니다!"
    echo "=============================================================="
else
    echo ""
    echo "=============================================================="
    echo "✗ 실습 중 오류가 발생했습니다."
    echo "=============================================================="
    exit 1
fi