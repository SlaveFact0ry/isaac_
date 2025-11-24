#!/usr/bin/env python3
"""
Isaac Sim Standalone 실습 2: 인터랙티브 환경과 카메라 제어
이 스크립트는 복잡한 씬 구성, 재질 적용, 카메라 제어를 학습합니다.

주요 학습 내용:
1. 복잡한 씬 구성 (다양한 형태의 객체)
2. 재질(Material) 적용
3. 카메라 설정 및 제어
4. 사용자 인터랙션
5. 애니메이션과 경로
"""

import numpy as np
import math
import time
import argparse
import os

# 커맨드 라인 인자 파싱
parser = argparse.ArgumentParser(description='Isaac Sim 인터랙티브 환경 실습')
parser.add_argument('--headless', action='store_true', help='헤드리스 모드로 실행')
parser.add_argument('--simulation_steps', type=int, default=600, help='시뮬레이션 스텝 수')
parser.add_argument('--save_images', action='store_true', help='카메라 이미지 저장')
args = parser.parse_args()

# ============================================================================
# 1. SimulationApp 초기화
# ============================================================================
print("=" * 60)
print("Isaac Sim 실습 2: 인터랙티브 환경과 카메라 제어")
print("=" * 60)
print(f"설정: headless={args.headless}, 스텝={args.simulation_steps}")
print("-" * 60)

from isaacsim import SimulationApp

# SimulationApp 생성
simulation_app = SimulationApp({
    "headless": args.headless,
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1080,
})

# SimulationApp이 생성된 후에 Isaac Sim 모듈들을 import
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, VisualCuboid, FixedCuboid
from omni.isaac.core.objects import DynamicSphere, VisualSphere, FixedSphere
from omni.isaac.core.objects import DynamicCylinder, VisualCylinder, FixedCylinder
from omni.isaac.core.objects import DynamicCone, VisualCone, FixedCone
from omni.isaac.core.objects import DynamicCapsule, VisualCapsule, FixedCapsule
from omni.isaac.core.prims import XFormPrim, GeometryPrim
from omni.isaac.core.utils.prims import create_prim, get_prim_at_path, define_prim
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.numpy.rotations as rot_utils
from pxr import Sdf, UsdLux, Gf, UsdGeom, UsdShade

# 이미지 저장을 위한 라이브러리
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    if args.save_images:
        print("WARNING: OpenCV not available. Image saving disabled.")
        args.save_images = False

# ============================================================================
# 2. World 생성
# ============================================================================
print("\n[1단계] World 생성 중...")

my_world = World(stage_units_in_meters=1.0)
stage = get_current_stage()

print("  ✓ World 생성 완료")

# ============================================================================
# 3. 조명 설정 - 다양한 조명 효과
# ============================================================================
print("\n[2단계] 조명 설정 중...")

# Dome Light (환경광)
dome_light = stage.DefinePrim("/World/DomeLight", "DomeLight")
dome_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(800.0)
dome_light.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.9, 0.95, 1.0))
print("  ✓ Dome Light 생성")

# Rect Light (면광원) - 창문 효과
rect_light = stage.DefinePrim("/World/RectLight", "RectLight")
xform = UsdGeom.Xformable(rect_light)
xform.AddTranslateOp().Set(Gf.Vec3f(-5, 0, 3))
xform.AddRotateXYZOp().Set(Gf.Vec3f(0, 90, 0))
rect_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(10000.0)
rect_light.CreateAttribute("inputs:width", Sdf.ValueTypeNames.Float).Set(3.0)
rect_light.CreateAttribute("inputs:height", Sdf.ValueTypeNames.Float).Set(2.0)
rect_light.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.95, 0.8))
print("  ✓ Rect Light 생성 (창문 효과)")

# Spot Light (스포트라이트)
spot_light = stage.DefinePrim("/World/SpotLight", "SphereLight")
xform = UsdGeom.Xformable(spot_light)
xform.AddTranslateOp().Set(Gf.Vec3f(2, 2, 4))
spot_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(30000.0)
spot_light.CreateAttribute("inputs:radius", Sdf.ValueTypeNames.Float).Set(0.3)
spot_light.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 0.8))
print("  ✓ Spot Light 생성")

# ============================================================================
# 4. 지면 및 벽 생성
# ============================================================================
print("\n[3단계] 환경 구성 중...")

# 지면
my_world.scene.add_default_ground_plane(
    z_position=0,
    name="ground_plane",
    prim_path="/World/ground",
    static_friction=0.5,
    dynamic_friction=0.5,
    restitution=0.1,
)
print("  ✓ 지면 생성")

# 벽 생성 (룸 효과) - 카메라를 가리지 않도록 멀리 배치
wall_thickness = 0.2
wall_height = 5.0
room_size = 15.0  # 방 크기를 더 크게

# 뒷벽 (더 멀리 배치)
back_wall = my_world.scene.add(
    FixedCuboid(
        prim_path="/World/back_wall",
        name="back_wall",
        position=np.array([0, room_size/2, wall_height/2]),
        scale=np.array([room_size, wall_thickness, wall_height]),
        size=1.0,
        color=np.array([220, 220, 220])  # 연한 회색
    )
)

# 왼쪽 벽 (더 멀리 배치)
left_wall = my_world.scene.add(
    FixedCuboid(
        prim_path="/World/left_wall",
        name="left_wall",
        position=np.array([-room_size/2, 0, wall_height/2]),
        scale=np.array([wall_thickness, room_size, wall_height]),
        size=1.0,
        color=np.array([220, 220, 220])  # 연한 회색
    )
)

# 오른쪽 벽 (추가 - 공간감을 위해)
right_wall = my_world.scene.add(
    FixedCuboid(
        prim_path="/World/right_wall",
        name="right_wall",
        position=np.array([room_size/2, 0, wall_height/2]),
        scale=np.array([wall_thickness, room_size, wall_height]),
        size=1.0,
        color=np.array([220, 220, 220])  # 연한 회색
    )
)
print("  ✓ 벽 생성 완료")

# ============================================================================
# 5. 재질(Material) 생성 - 간단하게 색상만 사용
# ============================================================================
print("\n[4단계] 재질 설정 건너뛰기 (API 호환성 문제)")
print("  ℹ 색상으로 재질 효과를 시뮬레이션합니다.")

# 재질 대신 None을 사용 (나중에 색상으로 구분)
metal_material = None
wood_material = None
plastic_material = None
glass_material = None

# ============================================================================
# 6. 다양한 형태의 객체 생성
# ============================================================================
print("\n[5단계] 다양한 객체 생성 중...")

# 테이블 (고정된 큐브)
table = my_world.scene.add(
    FixedCuboid(
        prim_path="/World/table",
        name="table",
        position=np.array([0, 0, 0.5]),
        scale=np.array([3.0, 2.0, 0.1]), # 넓고 얇게
        size=1.0,
        color=np.array([139, 69, 19])  # 갈색
    )
)
# table.apply_visual_material(wood_material)  # 재질 적용 건너뛰기
print("  ✓ 테이블 생성")

# 구 (금속)
metal_sphere = my_world.scene.add(
    DynamicSphere(
        prim_path="/World/metal_sphere",
        name="metal_sphere",
        position=np.array([0, 0, 1.0]),
        radius=0.2,
        color=np.array([192, 192, 192]),  # 은색
        mass=2.0,
    )
)
# metal_sphere.apply_visual_material(metal_material)  # 재질 적용 건너뛰기
print("  ✓ 금속 구 생성")

# 원기둥 (플라스틱)
plastic_cylinder = my_world.scene.add(
    DynamicCylinder(
        prim_path="/World/plastic_cylinder",
        name="plastic_cylinder",
        position=np.array([1, 0, 1.0]),
        radius=0.15,
        height=0.4,
        color=np.array([255, 0, 0]),  # 빨강
        mass=0.5,
    )
)
# plastic_cylinder.apply_visual_material(plastic_material)  # 재질 적용 건너뛰기
print("  ✓ 플라스틱 원기둥 생성")

# 유리 큐브
glass_cube = my_world.scene.add(
    DynamicCuboid(
        prim_path="/World/glass_cube",
        name="glass_cube",
        position=np.array([-1, 0, 1.0]),
        size=0.3,
        color=np.array([200, 200, 255]),  # 연한 파랑
        mass=1.0,
    )
)
# glass_cube.apply_visual_material(glass_material)  # 재질 적용 건너뛰기
print("  ✓ 유리 큐브 생성")

# 원뿔
cone = my_world.scene.add(
    DynamicCone(
        prim_path="/World/cone",
        name="cone",
        position=np.array([0, 1, 1.0]),
        radius=0.2,
        height=0.4,
        color=np.array([255, 255, 0]),  # 노랑
        mass=0.3,
    )
)
print("  ✓ 원뿔 생성")

# 캡슐
capsule = my_world.scene.add(
    DynamicCapsule(
        prim_path="/World/capsule",
        name="capsule",
        position=np.array([0, -1, 1.0]),
        radius=0.1,
        height=0.3,
        color=np.array([0, 255, 0]),  # 초록
        mass=0.2,
    )
)
print("  ✓ 캡슐 생성")

# ============================================================================
# 7. 카메라 설정
# ============================================================================
print("\n[6단계] 카메라 설정 중...")

# 메인 카메라
main_camera = Camera(
    prim_path="/World/MainCamera",
    name="main_camera",
    position=np.array([5.0, 5.0, 3.0]),
    frequency=20,
    resolution=(512, 512),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 30, -45]), degrees=True),
)
print("  ✓ 메인 카메라 생성")

# 탑뷰 카메라
top_camera = Camera(
    prim_path="/World/TopCamera",
    name="top_camera",
    position=np.array([0.0, 0.0, 8.0]),
    frequency=20,
    resolution=(256, 256),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
)
print("  ✓ 탑뷰 카메라 생성")

# 이미지 저장 디렉토리 생성
if args.save_images:
    image_dir = "~/isaac_/lecture-standalone/camera_images"
    os.makedirs(image_dir, exist_ok=True)
    print(f"  ✓ 이미지 저장 디렉토리 생성: {image_dir}")

# ============================================================================
# 8. 애니메이션용 객체 생성
# ============================================================================
print("\n[7단계] 애니메이션 객체 생성 중...")

# 궤도를 도는 구
orbiting_sphere = my_world.scene.add(
    VisualSphere(
        prim_path="/World/orbiting_sphere",
        name="orbiting_sphere",
        position=np.array([3, 0, 2]),
        radius=0.15,
        color=np.array([255, 100, 255]),  # 핑크
    )
)
print("  ✓ 궤도 구 생성")

# 위아래로 움직이는 큐브
bouncing_cube = my_world.scene.add(
    VisualCuboid(
        prim_path="/World/bouncing_cube",
        name="bouncing_cube",
        position=np.array([-3, 0, 2]),
        size=0.3,
        color=np.array([100, 255, 100]),  # 연두
    )
)
print("  ✓ 바운싱 큐브 생성")

# ============================================================================
# 9. 시뮬레이션 실행
# ============================================================================
print(f"\n{'=' * 60}")
print("시뮬레이션 시작")
print("키보드 컨트롤: 구현 예정")
print("=" * 60)

# World 초기화
my_world.reset()

# 초기 뷰포트 카메라 위치 설정 (물체가 잘 보이도록)
# Isaac Sim의 기본 뷰포트 카메라 제어
from omni.isaac.core.utils.viewports import set_camera_view

# 초기 카메라 위치: 앙사각에서 전체 씬을 볼 수 있도록
initial_camera_position = np.array([7.0, -7.0, 5.0])  # 앙사각 위치
initial_camera_target = np.array([0.0, 0.0, 1.0])  # 테이블 위를 바라보기

# 뷰포트 카메라 설정
set_camera_view(
    eye=initial_camera_position,
    target=initial_camera_target,
    camera_prim_path="/OmniverseKit_Persp"
)
print(f"초기 뷰포트 카메라 위치: {initial_camera_position}")
print(f"초기 뷰포트 카메라 타겟: {initial_camera_target}")

# 카메라 초기화
main_camera.initialize()
top_camera.initialize()

# 물리 설정
physics_context = my_world.get_physics_context()
if physics_context:
    physics_context.set_gravity(value=-9.81)
    print(f"중력 설정: -9.81 m/s²")

print("World 초기화 완료\n")

# 애니메이션 상수
ORBIT_RADIUS = 3.0           # 궤도 반경
ORBIT_HEIGHT = 2.0           # 궤도 높이
BOUNCE_AMPLITUDE = 1.0       # 바운스 진폭
BOUNCE_BASE_HEIGHT = 2.0     # 바운스 기준 높이
CAMERA_ORBIT_RADIUS = 8.0    # 카메라 궤도 반경
CAMERA_HEIGHT = 4.5          # 카메라 높이

# 시뮬레이션 이벤트 상수
STATUS_PRINT_INTERVAL = 20   # 상태 출력 주기
IMAGE_CAPTURE_INTERVAL = 100 # 이미지 캐처 주기
IMPACT_EVENT_STEP = 150      # 충격 이벤트 시점
GRAVITY_INVERT_STEP = 300    # 중력 반전 시점
GRAVITY_RESTORE_STEP = 400   # 중력 복원 시점
SLOW_MOTION_START = 500      # 슬로우 모션 시작
SLOW_MOTION_END = 550        # 슬로우 모션 종료

# 물리 상수
DEFAULT_GRAVITY = -9.81      # 기본 중력 (m/s²)
INVERTED_GRAVITY = 5.0       # 반전된 중력 (m/s²)

# 시뮬레이션 변수
step_count = 0

# 애니메이션 함수들
def circular_path(t, radius=ORBIT_RADIUS, height=ORBIT_HEIGHT):
    """원형 경로 계산"""
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = height
    return np.array([x, y, z])

def lissajous_path(t, a=3, b=2, delta=np.pi/2):
    """리사주 곡선 경로"""
    x = a * np.sin(t * 3)  # 주파수 3
    y = b * np.sin(t * 5 + delta)  # 주파수 5
    z = 2.0 + 0.3 * np.sin(t * 2)  # z축도 약간 움직임 (추가)
    return np.array([x, y, z])

def bounce_path(t, amplitude=BOUNCE_AMPLITUDE, base_height=BOUNCE_BASE_HEIGHT):
    """위아래 바운싱 경로 계산"""
    z = base_height + amplitude * abs(np.sin(t * 2))
    return z

# 메인 시뮬레이션 루프
print("\n실행 중인 애니메이션:")
print("  - 궤도를 도는 구 (원형 경로)")
print("  - 위아래로 움직이는 큐브")
print("  - 회전하는 메인 카메라")
print("  - 물리 시뮬레이션 객체들\n")

while simulation_app.is_running() and step_count < args.simulation_steps:
    # 시뮬레이션 스텝
    my_world.step(render=True)
    
    # 애니메이션 시간 계산
    t = step_count * 0.05  # 시간 스케일
    
    # 궤도 구 애니메이션 (리사주 곡선)
    orbit_pos = lissajous_path(t, a=3, b=2)
    orbiting_sphere.set_world_pose(position=orbit_pos)
    
    # 바운싱 큐브 애니메이션
    bounce_z = bounce_path(t, BOUNCE_AMPLITUDE, BOUNCE_BASE_HEIGHT)
    bouncing_cube.set_world_pose(position=np.array([-3, 0, bounce_z]))
    
    # 카메라 궤도 애니메이션
    camera_angle = t * 0.5  # 천천히 회전
    camera_x = CAMERA_ORBIT_RADIUS * np.cos(camera_angle)
    camera_y = CAMERA_ORBIT_RADIUS * np.sin(camera_angle)
    camera_pos = np.array([camera_x, camera_y, CAMERA_HEIGHT])
    
    # 카메라가 항상 원점을 바라보도록 방향 설정
    look_at_target = np.array([0, 0, 1])  # 테이블 위를 바라봄
    camera_forward = look_at_target - camera_pos
    camera_forward = camera_forward / np.linalg.norm(camera_forward)

    # 2. Right 벡터 계산
    world_up = np.array([0, 0, 1])
    camera_right = np.cross(camera_forward, world_up)
    camera_right = camera_right / np.linalg.norm(camera_right)

    # 3. Up 벡터 계산
    camera_up = np.cross(camera_right, camera_forward)

    # 4. 회전 행렬 구성
    rotation_matrix = np.array([
        camera_right,
        camera_forward,
        camera_up
    ]).T

    # 5. 쿼터니언으로 변환
    import omni.isaac.core.utils.numpy.rotations as rot_utils
    # Isaac Sim의 변환 함수
    quaternion = rot_utils.rot_matrices_to_quats(rotation_matrix)

    # 카메라 위치 업데이트
    main_camera.set_world_pose(position=camera_pos, orientation=quaternion)
    
    # 정기적으로 상태 출력
    if step_count % STATUS_PRINT_INTERVAL == 0:
        # 동적 객체들의 상태 확인
        sphere_pos, _ = metal_sphere.get_world_pose()
        cylinder_pos, _ = plastic_cylinder.get_world_pose()
        
        print(f"[Step {step_count:4d}] 카메라 각도: {math.degrees(camera_angle):.1f}°, "
              f"구 위치: [{sphere_pos[0]:.2f}, {sphere_pos[1]:.2f}, {sphere_pos[2]:.2f}]")
    
    # 정기적으로 이미지 캡처 (옵션)
    if args.save_images and step_count % IMAGE_CAPTURE_INTERVAL == 0 and CV2_AVAILABLE:
        # 메인 카메라 이미지 캡처
        # main_rgb = main_camera.get_rgba()[:, :, :3]
        rgba = main_camera.get_rgba()
        if rgba is not None and rgba.ndim == 3:
            main_rgb = rgba[:, :, :3]
        else:
            print("Warning: Camera image data not ready or invalid shape", rgba)
        main_depth = main_camera.get_depth()
        
        # 탑뷰 카메라 이미지 캡처
        top_rgb = top_camera.get_rgba()[:, :, :3]
        
        # 이미지 저장
        image_path = f"{image_dir}/main_camera_step_{step_count:04d}.png"
        cv2.imwrite(image_path, cv2.cvtColor((main_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        depth_path = f"{image_dir}/main_depth_step_{step_count:04d}.png"
        # Depth를 시각화용으로 정규화
        depth_normalized = (depth / np.max(depth) * 255).astype(np.uint8) if np.max(depth) > 0 else depth
        cv2.imwrite(depth_path, depth_normalized)
        
        top_path = f"{image_dir}/top_camera_step_{step_count:04d}.png"
        cv2.imwrite(top_path, cv2.cvtColor((top_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        print(f"  📸 이미지 저장: step {step_count}")
    
    # 특별 이벤트들
    
    # 충격 이벤트
    if step_count == IMPACT_EVENT_STEP:
        print(f"\n{'!' * 60}")
        print("충격 이벤트! 모든 동적 객체에 힘을 가합니다...")
        print(f"{'!' * 60}\n")
        
        # 각 객체에 랜덤 충격
        metal_sphere.set_linear_velocity(np.array([2, 1, 3]))
        plastic_cylinder.set_linear_velocity(np.array([-1, 2, 4]))
        glass_cube.set_linear_velocity(np.array([1, -1, 3]))
        cone.set_linear_velocity(np.array([0, 2, 5]))
        capsule.set_linear_velocity(np.array([2, 0, 4]))
    
    # 중력 반전
    if step_count == GRAVITY_INVERT_STEP:
        print(f"\n{'*' * 60}")
        print("중력 반전! 물체들이 위로 떠오릅니다...")
        print(f"{'*' * 60}\n")
        if physics_context:
            physics_context.set_gravity(value=INVERTED_GRAVITY)
    
    # 중력 복원
    if step_count == GRAVITY_RESTORE_STEP:
        print(f"\n중력 복원: {DEFAULT_GRAVITY} m/s²")
        if physics_context:
            physics_context.set_gravity(value=DEFAULT_GRAVITY)
    
    # 슬로우 모션 효과
    if SLOW_MOTION_START <= step_count < SLOW_MOTION_END:
        time.sleep(0.05)  # 슬로우 모션 효과
        if step_count == SLOW_MOTION_START:
            print(f"\n🎬 슬로우 모션 구간 ({SLOW_MOTION_END - SLOW_MOTION_START} 스텝)")
    
    step_count += 1
    
    # 일반 속도
    if not args.headless and step_count < SLOW_MOTION_START:
        time.sleep(0.01)

# ============================================================================
# 10. 시뮬레이션 종료
# ============================================================================
print(f"\n{'=' * 60}")
print("시뮬레이션 완료!")
print(f"총 실행 스텝: {step_count}")
print("=" * 60)

# 최종 상태 출력
print("\n최종 객체 상태:")
objects = [
    ("금속 구", metal_sphere),
    ("플라스틱 원기둥", plastic_cylinder),
    ("유리 큐브", glass_cube),
    ("원뿔", cone),
    ("캡슐", capsule)
]

for name, obj in objects:
    pos, _ = obj.get_world_pose()
    vel = obj.get_linear_velocity()
    print(f"  {name}: 위치={pos}, 속도={vel}")

if args.save_images:
    print(f"\n📁 이미지 저장 위치: {image_dir}")

# 정리
simulation_app.close()
print("\nIsaac Sim 종료 완료")