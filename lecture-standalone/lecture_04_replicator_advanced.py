#!/usr/bin/env python3
"""
실습 4: Isaac Sim Replicator 고급 - 로봇 비전 데이터셋 구축
YCB 객체와 다중 뷰포인트를 활용한 복잡한 합성 데이터 생성
"""

import numpy as np
import argparse
import os
import time

# 커맨드 라인 인자
parser = argparse.ArgumentParser(description='Isaac Sim Replicator 고급 실습')
parser.add_argument('--headless', action='store_true', help='헤드리스 모드')
parser.add_argument('--num_frames', type=int, default=100, help='생성할 프레임 수')
args = parser.parse_args()

print("=" * 50)
print("Isaac Sim Replicator 고급 실습")
print("로봇 비전 데이터셋 구축")
print("=" * 50)
print(f"프레임 수: {args.num_frames}")
print(f"헤드리스 모드: {args.headless}")
print()

# ====================================
# 1. SimulationApp 초기화
# ====================================
print("[Step 1] SimulationApp 초기화 중...")
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": args.headless,
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1080,
})
print("✓ SimulationApp 초기화 완료")

# ====================================
# 2. 필요한 모듈 임포트
# ====================================
print("[Step 2] 필요한 모듈 임포트 중...")

# Isaac Sim 모듈
from omni.isaac.core import World
from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.core.prims import XFormPrim

# Replicator 모듈
import omni.replicator.core as rep

# USD 모듈
from pxr import Gf, UsdGeom, UsdLux, Sdf, UsdPhysics, PhysxSchema
import omni

print("✓ 모든 모듈 임포트 완료")

# ====================================
# 3. World 생성 및 기본 설정
# ====================================
print("[Step 3] 시뮬레이션 월드 생성 중...")

my_world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

# 물리 설정
physics_context = my_world.get_physics_context()
if physics_context:
    physics_context.set_gravity(value=-9.81)

print("✓ World 생성 완료")

# ====================================
# 4. 조명 설정 (고급)
# ====================================
print("[Step 4] 고급 조명 시스템 구성 중...")

# Dome Light (HDR 환경광)
dome_light = stage.DefinePrim("/World/DomeLight", "DomeLight")
dome_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(1000.0)

# Key Light (주 조명)
key_light = create_prim(
    prim_path="/World/KeyLight",
    prim_type="DistantLight"
)
UsdLux.DistantLight(key_light).CreateIntensityAttr(5000)
UsdLux.DistantLight(key_light).CreateAngleAttr(0.53)
xform = UsdGeom.Xformable(key_light)
xform.AddRotateXYZOp().Set(Gf.Vec3f(-45, -45, 0))

# Fill Light (보조 조명)
fill_light = create_prim(
    prim_path="/World/FillLight",
    prim_type="DistantLight"
)
UsdLux.DistantLight(fill_light).CreateIntensityAttr(2000)
UsdLux.DistantLight(fill_light).CreateAngleAttr(1.0)
xform = UsdGeom.Xformable(fill_light)
xform.AddRotateXYZOp().Set(Gf.Vec3f(-35, 45, 0))

print("✓ 3-point 조명 시스템 완료")

# ====================================
# 5. 환경 구성
# ====================================
print("[Step 5] 환경 구성 중...")

# 지면
my_world.scene.add_default_ground_plane(
    z_position=0,
    name="ground_plane",
    prim_path="/World/GroundPlane",
    static_friction=0.5,
    dynamic_friction=0.5
)

# 테이블 (작업 공간)
table = my_world.scene.add(
    FixedCuboid(
        prim_path="/World/Table",
        name="table",
        position=np.array([0.5, 0.0, 0.35]),
        scale=np.array([0.8, 0.8, 0.05]),
        size=1.0,
        color=np.array([0.5, 0.5, 0.5])
    )
)

print("✓ 환경 구성 완료")

# ====================================
# 6. YCB 객체 로드
# ====================================
print("[Step 6] YCB 객체 로드 중...")

# YCB 객체 정의
ycb_objects = [
    {
        "name": "PottedMeatCan",
        "usd": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned/010_potted_meat_can.usd",
        "position": [0.5, 0.0, 0.45],
        "rotation": [0.707, -0.707, 0.0, 0.0],  # X축 -90도 회전
        "label": "meat_can"
    },
    {
        "name": "Banana",
        "usd": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned/011_banana.usd",
        "position": [0.4, 0.15, 0.45],
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "label": "banana"
    },
    {
        "name": "LargeMarker",
        "usd": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned/040_large_marker.usd",
        "position": [0.6, -0.15, 0.45],
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "label": "marker"
    },
    {
        "name": "TomatoSoupCan",
        "usd": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned/005_tomato_soup_can.usd",
        "position": [0.35, -0.1, 0.45],
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "label": "soup_can"
    }
]

# YCB 객체 생성
from omni.isaac.core.utils.semantics import add_update_semantics

for obj_info in ycb_objects:
    obj_path = f"/World/{obj_info['name']}"
    
    # Prim 생성
    create_prim(
        prim_path=obj_path,
        prim_type="Xform",
        position=obj_info["position"],
        orientation=obj_info["rotation"]
    )
    
    # USD 참조 추가
    add_reference_to_stage(usd_path=obj_info["usd"], prim_path=obj_path)
    
    # Semantic label 추가
    obj_prim = stage.GetPrimAtPath(obj_path)
    if obj_prim:
        add_update_semantics(obj_prim, obj_info["label"])
        
        # 물리 속성 추가
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(obj_prim)
        mass_api = UsdPhysics.MassAPI.Apply(obj_prim)
        mass_api.CreateMassAttr(0.1)  # 100g
        
        collision_api = UsdPhysics.CollisionAPI.Apply(obj_prim)
        
        print(f"  ✓ {obj_info['name']} 로드 및 설정 완료")

print("✓ YCB 객체 로드 완료")

# ====================================
# 7. 다중 카메라 설정
# ====================================
print("[Step 7] 다중 카메라 시스템 구성 중...")

# 메인 카메라 (움직이는 뷰)
main_camera = Camera(
    prim_path="/World/MainCamera",
    name="main_camera",
    position=np.array([1.2, 1.2, 1.0]),
    frequency=30,
    resolution=(512, 512),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 30, -45]), degrees=True)
)
main_camera.initialize()

# 탑뷰 카메라 (고정)
top_camera = Camera(
    prim_path="/World/TopCamera",
    name="top_camera",
    position=np.array([0.5, 0.0, 1.5]),
    frequency=30,
    resolution=(512, 512),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True)
)
top_camera.initialize()

# 사이드 카메라 (고정)
side_camera = Camera(
    prim_path="/World/SideCamera",
    name="side_camera",
    position=np.array([1.5, 0.0, 0.6]),
    frequency=30,
    resolution=(512, 512),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 10, -90]), degrees=True)
)
side_camera.initialize()

print("✓ 3개 카메라 시스템 구성 완료")

# ====================================
# 8. Replicator 고급 설정
# ====================================
print(f"\n{'='*50}")
print("[Replicator 고급 설정]")
print("="*50)

# 출력 디렉토리
output_dir = os.path.join(os.path.dirname(__file__), "replicator_output", "advanced_dataset")
os.makedirs(output_dir, exist_ok=True)

print("[Step 8] Replicator 고급 기능 설정 중...")

# Replicator graph 정의 (고급 도메인 랜덤화)
with rep.new_layer():
    
    # 카메라 render products
    main_cam_rp = rep.create.render_product("/World/MainCamera", (512, 512))
    top_cam_rp = rep.create.render_product("/World/TopCamera", (512, 512))
    side_cam_rp = rep.create.render_product("/World/SideCamera", (512, 512))
    
    # YCB 객체들 그룹화
    ycb_prims = []
    for obj_info in ycb_objects:
        ycb_prims.append(rep.get.prim_at_path(f"/World/{obj_info['name']}"))
    
    all_objects = rep.create.group(ycb_prims)
    
    # 텍스처 랜덤화를 위한 재질 생성
    def random_color():
        return rep.distribution.uniform((0.2, 0.2, 0.2), (0.9, 0.9, 0.9))
    
    # 도메인 랜덤화 트리거
    with rep.trigger.on_frame(num_frames=args.num_frames):
        
        # 1. 객체 위치와 회전 랜덤화
        with all_objects:
            rep.modify.pose(
                position=rep.distribution.uniform((0.3, -0.25, 0.42), (0.7, 0.25, 0.5)),
                rotation=rep.distribution.uniform((0, 0, 0), (360, 360, 360))
            )
        
        # 2. 조명 랜덤화 (강도와 방향)
        with rep.get.prim_at_path("/World/DomeLight"):
            rep.modify.attribute(
                "inputs:intensity",
                rep.distribution.uniform(500, 2000)
            )
        
        with rep.get.prim_at_path("/World/KeyLight"):
            rep.modify.attribute(
                "inputs:intensity",
                rep.distribution.uniform(3000, 7000)
            )
            # 조명 방향도 약간 변경
            rep.modify.attribute(
                "xformOp:rotateXYZ",
                rep.distribution.uniform((-50, -50, -10), (-40, -40, 10))
            )
        
        # 3. 메인 카메라 위치 랜덤화 (다양한 뷰포인트)
        with rep.get.prim_at_path("/World/MainCamera"):
            # 구면 좌표계를 사용한 카메라 위치
            rep.modify.pose(
                position=rep.distribution.uniform((0.8, 0.8, 0.6), (1.4, 1.4, 1.2)),
                look_at=(0.5, 0.0, 0.45)  # 테이블 중심을 바라봄
            )
        
        # 4. 테이블 색상 랜덤화 (배경 다양성)
        with rep.get.prim_at_path("/World/Table/geometry"):
            rep.modify.attribute(
                "primvars:displayColor",
                rep.distribution.uniform((0.3, 0.3, 0.3), (0.8, 0.8, 0.8))
            )
        
        # 5. 물리 속성 랜덤화 (선택적)
        # 각 객체의 질량을 약간씩 변경
        for i, obj_prim in enumerate(ycb_prims):
            with obj_prim:
                rep.modify.attribute(
                    "physics:mass",
                    rep.distribution.uniform(0.05, 0.2)
                )

print("✓ 고급 도메인 랜덤화 설정 완료")

# Writer 설정 (다중 카메라, 고급 어노테이션)
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir=output_dir,
    rgb=True,  # RGB 이미지
    bounding_box_2d_tight=True,  # 2D 바운딩 박스
    bounding_box_3d=True,  # 3D 바운딩 박스
    semantic_segmentation=True,  # 시맨틱 세그멘테이션
    instance_segmentation=True,  # 인스턴스 세그멘테이션
    distance_to_image_plane=True,  # 깊이 정보
    distance_to_camera=True,  # 카메라까지의 거리
    camera_params=True,  # 카메라 파라미터
    occlusion=True,  # 가림 정보
    normals=False,  # 법선 벡터 (비활성화)
    pointcloud=False  # 포인트 클라우드 (비활성화)
)

# 모든 카메라에 writer 연결
writer.attach([main_cam_rp, top_cam_rp, side_cam_rp])

print(f"✓ 다중 카메라 Writer 설정 완료")
print(f"✓ 출력 디렉토리: {output_dir}")

# ====================================
# 9. World 초기화
# ====================================
print("[Step 9] 월드 초기화 중...")
my_world.reset()

# 초기 안정화
for i in range(20):
    my_world.step(render=True)

print("✓ 월드 초기화 완료")

# ====================================
# 10. 데이터 생성 실행
# ====================================
print(f"\n{'='*50}")
print("[고급 데이터셋 생성 시작]")
print("="*50)

print(f"총 {args.num_frames}개의 프레임을 3개 카메라에서 생성합니다...")
print(f"예상 출력: {args.num_frames * 3}개의 이미지 세트")

# Replicator 실행
start_time = time.time()
rep.orchestrator.run_until_complete()
end_time = time.time()

total_time = end_time - start_time
fps = args.num_frames / total_time if total_time > 0 else 0

# ====================================
# 11. 결과 요약
# ====================================
print(f"\n{'='*50}")
print("[데이터 생성 완료]")
print("="*50)

print(f"✓ 생성된 프레임: {args.num_frames} x 3 카메라")
print(f"✓ 소요 시간: {total_time:.2f}초")
print(f"✓ 평균 FPS: {fps:.2f}")
print(f"✓ 저장 위치: {output_dir}")

# 생성된 파일 통계
if os.path.exists(output_dir):
    import glob
    
    # 각 카메라별 파일 수 계산
    for cam_name in ["MainCamera", "TopCamera", "SideCamera"]:
        cam_dir = os.path.join(output_dir, cam_name)
        if os.path.exists(cam_dir):
            rgb_files = glob.glob(os.path.join(cam_dir, "*rgb*.png"))
            seg_files = glob.glob(os.path.join(cam_dir, "*semantic*.png"))
            bbox_files = glob.glob(os.path.join(cam_dir, "*bounding_box_2d*.npy"))
            
            print(f"\n[{cam_name}]")
            print(f"  - RGB 이미지: {len(rgb_files)}개")
            print(f"  - Segmentation: {len(seg_files)}개")
            print(f"  - Bounding Box: {len(bbox_files)}개")

# ====================================
# 12. 학습 포인트 요약
# ====================================
print(f"\n{'='*50}")
print("[고급 학습 포인트 요약]")
print("="*50)

print("""
✓ 고급 Replicator 기능:
  1. 다중 카메라 시스템 (3개 뷰포인트)
  2. YCB 객체 활용 (실제 데이터셋)
  3. 3D Bounding Box 생성
  4. Occlusion 정보 추가
  5. 복잡한 조명 시스템 (3-point lighting)

✓ 고급 도메인 랜덤화:
  - 다중 객체 동시 랜덤화
  - 조명 강도와 방향 변화
  - 배경(테이블) 색상 변화
  - 물리 속성 랜덤화
  - 카메라 궤도 랜덤화

✓ 생성된 고급 어노테이션:
  - 다중 뷰 RGB 이미지
  - 2D/3D Bounding Box
  - Semantic/Instance Segmentation
  - Depth & Distance 정보
  - Occlusion 마스크
  - Camera Intrinsics/Extrinsics

✓ 실무 활용:
  - 로봇 그래스핑 학습
  - 6-DoF 포즈 추정
  - 다중 뷰 3D 재구성
  - Sim-to-Real 전이 학습
""")


# ====================================
# 13. 종료
# ====================================
print("\n시뮬레이션을 종료합니다...")
simulation_app.close()
print("✓ 프로그램 종료")
