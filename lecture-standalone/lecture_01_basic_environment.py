#!/usr/bin/env python3
"""
Isaac Sim Standalone 실습 1: 기본 환경 구성
이 스크립트는 Isaac Sim의 기본적인 환경 구성 방법을 학습합니다.

주요 학습 내용:
1. SimulationApp 초기화
2. 기본 객체 생성 및 배치
3. 조명 설정
4. 물리 속성 설정
5. 시뮬레이션 제어
"""

import numpy as np
import time
import argparse

# 커맨드 라인 인자 파싱
parser = argparse.ArgumentParser(description='Isaac Sim 기본 환경 구성 실습')
parser.add_argument('--headless', action='store_true', help='헤드리스 모드로 실행')
parser.add_argument('--num_cubes', type=int, default=5, help='생성할 큐브 개수')
parser.add_argument('--simulation_steps', type=int, default=500, help='시뮬레이션 스텝 수')
args = parser.parse_args()

# ============================================================================
# 1. SimulationApp 초기화
# ============================================================================
print("=" * 60)
print("Isaac Sim 실습 1: 기본 환경 구성")
print("=" * 60)
print(f"설정: headless={args.headless}, 큐브 개수={args.num_cubes}")
print("-" * 60)

from isaacsim import SimulationApp

# SimulationApp 생성 - 이것이 가장 먼저 와야 합니다!
simulation_app = SimulationApp({
    "headless": args.headless,  # GUI 표시 여부
    "width": 1280,              # 뷰포트 너비
    "height": 720,              # 뷰포트 높이
    "window_width": 1920,       # 창 너비
    "window_height": 1080,      # 창 높이
})

# SimulationApp이 생성된 후에 Isaac Sim 모듈들을 import 합니다
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, VisualCuboid, FixedCuboid
from omni.isaac.core.prims import XFormPrim
from pxr import Sdf, UsdLux, Gf, UsdGeom
import omni

# ============================================================================
# 2. World 생성 및 기본 설정
# ============================================================================
print("\n[1단계] World 생성 중...")

# World 객체 생성 - 시뮬레이션의 기본 컨테이너
my_world = World(stage_units_in_meters=1.0)
print(f"  ✓ World 생성 완료")

# ============================================================================
# 3. 조명 설정
# ============================================================================
print("\n[2단계] 조명 설정 중...")

# Stage 접근 (USD 씬 그래프)
stage = omni.usd.get_context().get_stage()

# Dome Light (환경광) 생성
dome_light = stage.DefinePrim("/World/DomeLight", "DomeLight")
dome_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(1000.0)
dome_light.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.98, 0.95))
print("  ✓ Dome Light 생성 완료")

# Point Light (점광원) 생성
point_light = stage.DefinePrim("/World/PointLight", "SphereLight")
xform = UsdGeom.Xformable(point_light)
xform.AddTranslateOp().Set(Gf.Vec3f(0, 0, 5))
point_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(50000.0)
point_light.CreateAttribute("inputs:radius", Sdf.ValueTypeNames.Float).Set(0.5)
point_light.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.9, 0.7))

print("  ✓ Point Light 생성 완료")

# Distant Light (방향광) 생성 - 태양광 시뮬레이션
distant_light = stage.DefinePrim("/World/DistantLight", "DistantLight")
xform = UsdGeom.Xformable(distant_light)
xform.AddRotateXYZOp().Set(Gf.Vec3f(-45, -45, 0))
distant_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(3000.0)
print("  ✓ Distant Light 생성 완료")

# ============================================================================
# 4. 지면 생성
# ============================================================================
print("\n[3단계] 지면 생성 중...")

# 기본 지면 추가 (물리 충돌 포함)
my_world.scene.add_default_ground_plane(
    z_position=0,
    name="ground_plane",
    prim_path="/World/ground",
    static_friction=0.5,
    dynamic_friction=0.5,
    restitution=0.2,  # 반발 계수
)
print("  ✓ 지면 생성 완료")

# ============================================================================
# 5. 객체 생성 - 다양한 타입의 큐브들
# ============================================================================
print(f"\n[4단계] {args.num_cubes}개의 큐브 생성 중...")

created_cubes = []
colors = [
    [1.0, 0.2, 0.2],  # 빨강
    [0.2, 1.0, 0.2],  # 초록
    [0.2, 0.2, 1.0],  # 파랑
    [1.0, 1.0, 0.2],  # 노랑
    [1.0, 0.2, 1.0],  # 마젠타
    [0.2, 1.0, 1.0],  # 시안
]

# 5-1. Visual Cube (물리 없음, 장식용)
visual_cube = my_world.scene.add(
    VisualCuboid(
        prim_path="/World/visual_cube",
        name="visual_cube",
        position=np.array([-3, 0, 0.5]),
        size=0.5,
        color=np.array([0, 0, 0])  # 검정색
    )
)

print("  ✓ Visual Cube 생성 (물리 없음)")

# 5-2. Fixed Cube (고정된 큐브, 충돌은 있지만 움직이지 않음)
fixed_cube = my_world.scene.add(
    FixedCuboid(
        prim_path="/World/fixed_cube",
        name="fixed_cube",
        position=np.array([0, 0, 0.25]),
        size=0.5,
        color=np.array([128, 128, 128])  # 회색
    )
)
print("  ✓ Fixed Cube 생성 (고정됨)")

# 5-3. Dynamic Cubes (물리 시뮬레이션 적용)
for i in range(args.num_cubes):
    # 위치 계산 - 원형으로 배치
    angle = (2 * np.pi * i) / args.num_cubes
    x = 2.0 * np.cos(angle)
    y = 2.0 * np.sin(angle)
    z = 1.0 + i * 0.5  # 높이를 다르게 설정
    
    # 크기 변화
    size = 0.3 + (i * 0.05)
    
    # 색상 선택
    color = np.array(colors[i % len(colors)])
    
    # 동적 큐브 생성
    cube = my_world.scene.add(
        DynamicCuboid(
            prim_path=f"/World/dynamic_cube_{i}",
            name=f"cube_{i}",
            position=np.array([x, y, z]),
            size=size,
            color=color,
            mass=1.0 + i * 0.2,  # 질량 변화
            linear_velocity=np.array([0, 0, 0]),  # 초기 속도
        )
    )
    
    created_cubes.append(cube)
    print(f"  ✓ Dynamic Cube {i+1}/{args.num_cubes} 생성 (위치: [{x:.2f}, {y:.2f}, {z:.2f}])")

# ============================================================================
# 6. 추가 객체 - 도미노 효과를 위한 큐브들
# ============================================================================
print("\n[5단계] 도미노 큐브 생성 중...")

domino_cubes = []
for i in range(10):
    domino = my_world.scene.add(
        DynamicCuboid(
            prim_path=f"/World/domino_{i}",
            name=f"domino_{i}",
            position=np.array([i * 0.6 - 3, 3, 0.5]),
            scale=np.array([0.1, 0.4, 1.0]),  # 얇고 높은 형태
            size=0.5,
            color=np.array([200, 150, 100]),  # 나무색
            mass=0.5,
        )
    )

    domino_cubes.append(domino)
    print(f"  ✓ {len(domino_cubes)}개의 도미노 큐브 생성 완료")

# ============================================================================
# 7. 시뮬레이션 실행
# ============================================================================
print(f"\n{'=' * 60}")
print("시뮬레이션 시작")
print("=" * 60)

# World 초기화 및 물리 설정
my_world.reset()

# 물리 컨텍스트를 통한 중력 설정
physics_context = my_world.get_physics_context()
if physics_context:
    physics_context.set_gravity(value=-9.81)
    print(f"중력 설정: -9.81 m/s²")

print("World 초기화 완료\n")

# 시뮬레이션 이벤트 상수 정의
STATUS_PRINT_INTERVAL = 10  # 상태 출력 주기
COLOR_CHANGE_INTERVAL = 50  # 색상 변경 주기
PUSH_DOMINO_STEP = 100      # 도미노 밀기 시점
RANDOM_IMPACT_STEP = 200    # 랜덤 충격 이벤트 시점
GRAVITY_CHANGE_STEP = 300   # 중력 변경 시점
GRAVITY_RESTORE_STEP = 400  # 중력 복원 시점

# 물리 상수
DEFAULT_GRAVITY = -9.81     # 기본 중력 (m/s²)
MODIFIED_GRAVITY = -20.0    # 변경된 중력 (m/s²)

# 시뮬레이션 변수
step_count = 0

# 메인 시뮬레이션 루프
while simulation_app.is_running() and step_count < args.simulation_steps:
    # 시뮬레이션 한 스텝 실행
    my_world.step(render=True)
    
    # 정기적으로 상태 출력
    if step_count % STATUS_PRINT_INTERVAL == 0:
        # 첫 번째 동적 큐브의 상태 출력
        if created_cubes:
            pos, orient = created_cubes[0].get_world_pose()
            vel = created_cubes[0].get_linear_velocity()
            print(f"[Step {step_count:4d}] Cube_0 위치: [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}], "
                  f"속도: [{vel[0]:6.3f}, {vel[1]:6.3f}, {vel[2]:6.3f}]")

    # 도미노 효과 트리거
    if step_count == PUSH_DOMINO_STEP:
        print(f"\n{'!' * 60}")
        print("도미노 효과 시작! 첫 번째 도미노를 밉니다...")
        print(f"{'!' * 60}\n")
        # 첫 번째 도미노에 힘 가하기
        if domino_cubes:
            domino_cubes[0].set_linear_velocity(np.array([2.0, 0, 0]))
    
    # 랜덤 충격 이벤트
    if step_count == RANDOM_IMPACT_STEP:
        print(f"\n{'*' * 60}")
        print("랜덤 충격 이벤트! 모든 동적 큐브에 힘을 가합니다...")
        print(f"{'*' * 60}\n")
        for cube in created_cubes:
            random_force = np.array([
                np.random.uniform(-3, 3),
                np.random.uniform(-3, 3),
                np.random.uniform(5, 10)
            ])
            cube.set_linear_velocity(random_force)
    
    # 물리 파라미터 변경
    if step_count == GRAVITY_CHANGE_STEP:
        print(f"\n중력 변경: {MODIFIED_GRAVITY} m/s²")
        if physics_context:
            physics_context.set_gravity(value=MODIFIED_GRAVITY)
    
    # 중력 복원
    if step_count == GRAVITY_RESTORE_STEP:
        print(f"\n중력 복원: {DEFAULT_GRAVITY} m/s²")
        if physics_context:
            physics_context.set_gravity(value=DEFAULT_GRAVITY)
    
    step_count += 1
    
    # ESC 키로 종료 (headless 모드가 아닐 때)
    if not args.headless:
        # 시뮬레이션 시간 제어
        time.sleep(0.01)  # 약 100 FPS

# ============================================================================
# 8. 시뮬레이션 종료
# ============================================================================
print(f"\n{'=' * 60}")
print("시뮬레이션 완료!")
print(f"총 실행 스텝: {step_count}")
print("=" * 60)

# 최종 상태 출력
print("\n최종 객체 상태:")
for i, cube in enumerate(created_cubes[:3]):  # 처음 3개만 출력
    pos, _ = cube.get_world_pose()
    vel = cube.get_linear_velocity()
    print(f"  Cube_{i}: 위치={pos}, 속도={vel}")

# 정리
simulation_app.close()
print("\nIsaac Sim 종료 완료")