#!/usr/bin/env python3
"""
실습 3: Isaac Sim Replicator 기초 - 합성 데이터 생성 입문
Replicator의 핵심 개념과 도메인 랜덤화를 학습합니다.
"""

import argparse
import os

# 커맨드 라인 인자
parser = argparse.ArgumentParser(description='Isaac Sim Replicator 기초 실습')
parser.add_argument('--headless', action='store_true', help='헤드리스 모드')
parser.add_argument('--num_frames', type=int, default=10, help='생성할 프레임 수')
args = parser.parse_args()

print("=" * 50)
print("Isaac Sim Replicator 기초 실습")
print("합성 데이터 생성 파이프라인 구축")
print("=" * 50)
print(f"프레임 수: {args.num_frames}")
print(f"헤드리스 모드: {args.headless}")
print()

# ====================================
# 1. SimulationApp 초기화
# ====================================
print("[Step 1] SimulationApp 초기화 중...")
from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={
    "headless": args.headless,
    "width": 1280,
    "height": 720
})
print("✓ SimulationApp 초기화 완료")

# ====================================
# 2. 필요한 모듈 임포트
# ====================================
print("[Step 2] 필요한 모듈 임포트 중...")

import omni.replicator.core as rep
import omni.usd
from isaacsim.core.utils.semantics import add_update_semantics
from pxr import Sdf, UsdGeom

print("✓ 모든 모듈 임포트 완료")

# ====================================
# 3. 스테이지 생성 및 기본 설정
# ====================================
print("[Step 3] 새로운 스테이지 생성 중...")

# 새 스테이지 생성
omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()

# Replicator capture on play 비활성화 (수동 제어)
rep.orchestrator.set_capture_on_play(False)

print("✓ 스테이지 생성 완료")

# ====================================
# 4. 조명 설정
# ====================================
print("[Step 4] 조명 시스템 구성 중...")

# Dome Light 생성
dome_light = stage.DefinePrim("/World/DomeLight", "DomeLight")
dome_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(1000.0)

# Distant Light 추가
distant_light = stage.DefinePrim("/World/DistantLight", "DistantLight")
distant_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(3000.0)
UsdGeom.Xformable(distant_light).AddRotateXYZOp().Set((-45.0, -45.0, 0.0))

print("✓ 조명 설정 완료")

# ====================================
# 5. 환경 구성 (지면과 테이블)
# ====================================
print("[Step 5] 기본 환경 구성 중...")

# Ground Plane
ground = stage.DefinePrim("/World/GroundPlane", "Xform")
ground_mesh = stage.DefinePrim("/World/GroundPlane/Mesh", "Mesh")
UsdGeom.Mesh(ground_mesh).CreatePointsAttr([
    (-10, -10, 0), (10, -10, 0), (10, 10, 0), (-10, 10, 0)
])
UsdGeom.Mesh(ground_mesh).CreateFaceVertexCountsAttr([4])
UsdGeom.Mesh(ground_mesh).CreateFaceVertexIndicesAttr([0, 1, 2, 3])
UsdGeom.Mesh(ground_mesh).CreateDisplayColorAttr([(0.5, 0.5, 0.5)])

# 테이블 (큐브로 표현)
table = stage.DefinePrim("/World/Table", "Cube")
UsdGeom.Xformable(table).AddTranslateOp().Set((0.0, 0.0, 0.4))
UsdGeom.Xformable(table).AddScaleOp().Set((0.8, 0.8, 0.05))
UsdGeom.Gprim(table).CreateDisplayColorAttr([(0.6, 0.4, 0.2)])
add_update_semantics(table, "table")

print("✓ 환경 구성 완료")

# ====================================
# 6. 학습용 객체 생성
# ====================================
print("[Step 6] 학습용 객체 생성 중...")

# 객체 1: 빨간 큐브
cube1 = stage.DefinePrim("/World/Cube1", "Cube")
UsdGeom.Xformable(cube1).AddTranslateOp().Set((0.0, 0.0, 0.5))
UsdGeom.Xformable(cube1).AddScaleOp().Set((0.05, 0.05, 0.05))
UsdGeom.Gprim(cube1).CreateDisplayColorAttr([(1.0, 0.0, 0.0)])
add_update_semantics(cube1, "red_cube")

# 객체 2: 초록 구
sphere1 = stage.DefinePrim("/World/Sphere1", "Sphere")
UsdGeom.Xformable(sphere1).AddTranslateOp().Set((0.15, 0.0, 0.5))
UsdGeom.Xformable(sphere1).AddScaleOp().Set((0.05, 0.05, 0.05))
UsdGeom.Gprim(sphere1).CreateDisplayColorAttr([(0.0, 1.0, 0.0)])
add_update_semantics(sphere1, "green_sphere")

# 객체 3: 파란 실린더
cylinder1 = stage.DefinePrim("/World/Cylinder1", "Cylinder")
UsdGeom.Xformable(cylinder1).AddTranslateOp().Set((-0.15, 0.0, 0.5))
UsdGeom.Xformable(cylinder1).AddScaleOp().Set((0.05, 0.05, 0.1))
UsdGeom.Gprim(cylinder1).CreateDisplayColorAttr([(0.0, 0.0, 1.0)])
add_update_semantics(cylinder1, "blue_cylinder")

print("✓ 3개의 학습용 객체 생성 완료")

# ====================================
# 7. 카메라 생성
# ====================================
print("[Step 7] 카메라 설정 중...")

# 메인 카메라 생성 - 탑다운 뷰
camera = stage.DefinePrim("/World/Camera", "Camera")
UsdGeom.Xformable(camera).AddTranslateOp().Set((0.0593, 0.11424, 1.59549))
UsdGeom.Xformable(camera).AddRotateXYZOp().Set((0.82228, 2.12303, 93.2228))

# 카메라 속성 설정
camera_geom = UsdGeom.Camera(camera)
camera_geom.CreateFocalLengthAttr(24.0)
camera_geom.CreateFocusDistanceAttr(1.5)

print("✓ 카메라 설정 완료")

# ====================================
# 8. Replicator 도메인 랜덤화 설정
# ====================================
print("\n[Replicator 설정]")
print("[Step 8] 도메인 랜덤화 설정 중...")

# 색상 랜덤화 함수
def randomize_colors():
    """객체들의 색상을 랜덤하게 변경"""
    cube1_prim = rep.get.prim_at_path("/World/Cube1")
    sphere1_prim = rep.get.prim_at_path("/World/Sphere1")
    cylinder1_prim = rep.get.prim_at_path("/World/Cylinder1")

    with cube1_prim:
        rep.randomizer.color(colors=rep.distribution.uniform((0.5, 0, 0), (1, 0.3, 0.3)))
    with sphere1_prim:
        rep.randomizer.color(colors=rep.distribution.uniform((0, 0.5, 0), (0.3, 1, 0.3)))
    with cylinder1_prim:
        rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0.5), (0.3, 0.3, 1)))

    return cube1_prim.node

# 위치 랜덤화 함수
def randomize_positions():
    """객체들의 위치를 테이블 위에서 랜덤하게 변경"""
    objects = rep.get.prims(path_pattern="/World/(Cube1|Sphere1|Cylinder1)")
    with objects:
        rep.modify.pose(
            position=rep.distribution.uniform((-0.25, -0.25, 0.45), (0.25, 0.25, 0.52)),
            rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360))
        )
    return objects.node

# 랜덤화 함수 등록
rep.randomizer.register(randomize_colors)
rep.randomizer.register(randomize_positions)

# 매 프레임마다 랜덤화 실행
with rep.trigger.on_frame():
    rep.randomizer.randomize_colors()
    rep.randomizer.randomize_positions()

print("✓ 도메인 랜덤화 설정 완료")

# ====================================
# 9. Render Product 및 Writer 설정
# ====================================
print("[Step 9] 데이터 저장 설정 중...")

# Render Product 생성 (카메라와 해상도 연결)
render_product = rep.create.render_product("/World/Camera", (512, 512))

# 출력 디렉토리 설정
output_dir = os.path.join(os.path.dirname(__file__), "replicator_output", "lecture03_basic")
os.makedirs(output_dir, exist_ok=True)

# Writer 설정 (문서와 일치하는 어노테이션만 생성)
writer = rep.writers.get("BasicWriter")
writer.initialize(
    output_dir=output_dir,
    rgb=True,  # RGB 이미지 캡처
    bounding_box_2d_tight=True,  # 2D Bounding Box
    semantic_segmentation=True,  # Semantic Segmentation 기초
    distance_to_image_plane=True  # Depth 맵 생성
)
writer.attach(render_product)

print(f"✓ Writer 설정 완료")
print(f"✓ 출력 디렉토리: {output_dir}")

# ====================================
# 10. 데이터 생성 실행
# ====================================
print(f"\n{'='*50}")
print("[데이터 생성 시작]")
print("="*50)

print(f"총 {args.num_frames}개의 프레임을 생성합니다...")

# 데이터 캡처 실행
for i in range(args.num_frames):
    print(f"프레임 {i+1}/{args.num_frames} 생성 중...", end='\r')
    rep.orchestrator.step()

print(f"\n✓ {args.num_frames}개 프레임 생성 완료")

# Writer 정리
writer.detach()

# 데이터 쓰기 완료 대기
print("데이터 저장 중...")
rep.orchestrator.wait_until_complete()

# ====================================
# 11. 결과 확인
# ====================================
print(f"\n{'='*50}")
print("[결과 요약]")
print("="*50)

# 생성된 파일 확인
if os.path.exists(output_dir):
    import glob
    rgb_files = glob.glob(os.path.join(output_dir, "*rgb*.png"))
    seg_files = glob.glob(os.path.join(output_dir, "*semantic*.png"))
    bbox_files = glob.glob(os.path.join(output_dir, "*bounding*.npy"))

    print(f"✓ RGB 이미지: {len(rgb_files)}개")
    print(f"✓ Segmentation 마스크: {len(seg_files)}개")
    print(f"✓ Bounding Box: {len(bbox_files)}개")
    print(f"✓ 저장 위치: {output_dir}")

# ====================================
# 12. 학습 포인트 요약
# ====================================
print(f"\n{'='*50}")
print("[학습 포인트 요약]")
print("="*50)

print("""
✓ Replicator 핵심 개념:
  1. Render Product: 카메라와 해상도 연결
  2. Writer: 데이터 저장 설정
  3. Randomizer: 도메인 랜덤화 함수
  4. Trigger: 실행 타이밍 제어
  5. Orchestrator: 실행 관리

✓ 구현한 도메인 랜덤화:
  - 객체 색상 랜덤화
  - 위치/회전 랜덤화
  
✓ 생성된 어노테이션:
  - RGB 이미지 캡처
  - Depth 맵 생성
  - 2D Bounding Box
  - Semantic Segmentation 기초
""")

print("[다음 단계]")
print(f"1. 생성된 데이터 확인: {output_dir}")
print("2. 실습 4에서 고급 기능 학습")

# ====================================
# 13. 종료
# ====================================
print("\n시뮬레이션을 종료합니다...")
simulation_app.close()
print("✓ 프로그램 종료")