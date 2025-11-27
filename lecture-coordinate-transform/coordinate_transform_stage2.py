#!/usr/bin/env python3
"""
Stage 2: 3D 변환과 카메라 좌표계
- Stage 1 기능 포함
- Pinhole 카메라 모델로 3D 변환 추가
- 카메라 좌표계에서 포인트 클라우드 시각화

새로운 학습 포인트:
- 카메라 내부 파라미터 (fx, fy, cx, cy)
- 2D 픽셀 → 3D 카메라 좌표 변환 공식
- 포인트 클라우드 생성과 시각화
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import cv2
from typing import Tuple

# Qt 플랫폼 충돌 방지
import matplotlib
if 'DISPLAY' in os.environ:
    try:
        matplotlib.use('TkAgg')
    except (ImportError, RuntimeError) as e:
        matplotlib.use('Agg')
else:
    matplotlib.use('Agg')

# Open3D import
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    print("Warning: Open3D not installed. 3D visualization will be limited.")
    HAS_OPEN3D = False

# utils를 위한 상위 경로 추가
sys.path.append(str(Path(__file__).parent.parent))
from utils import camera_utils

# ====================================
# 데이터 경로 설정
# ====================================
DATA_BASE_PATH = Path(__file__).parent.parent / "lecture-standalone" / "replicator_output" / "advanced_dataset" / "Replicator_04"

def print_step(step_num: int, description: str) -> float:
    """단계별 출력 헬퍼"""
    print(f"\n[Step {step_num}] {description}...")
    return time.time()

def print_done(start_time: float, details: str = "") -> None:
    """완료 메시지 출력"""
    elapsed = time.time() - start_time
    print(f"✓ 완료 ({elapsed:.3f}초) {details}")

# ====================================
# Step 1: 데이터 로드 (Stage 1에서 복사)
# ====================================
def load_data(frame_id: int = 53) -> Tuple[np.ndarray, np.ndarray]:
    """RGB-D 데이터 로드"""
    
    # 파일 경로 생성
    rgb_path = DATA_BASE_PATH / "rgb" / f"rgb_{frame_id:04d}.png"
    depth_path = DATA_BASE_PATH / "distance_to_image_plane" / f"distance_to_image_plane_{frame_id:04d}.npy"
    
    # 파일 존재 확인
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB 파일을 찾을 수 없습니다: {rgb_path}")
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth 파일을 찾을 수 없습니다: {depth_path}")
    
    # RGB 이미지 로드
    rgb_image = cv2.imread(str(rgb_path))
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    
    # Depth 맵 로드
    depth_map = np.load(str(depth_path))
    
    print(f"  RGB shape: {rgb_image.shape}")
    print(f"  Depth shape: {depth_map.shape}")
    print(f"  Depth range: {depth_map.min():.3f} - {depth_map.max():.3f} meters")
    
    return rgb_image, depth_map

# ====================================
# Step 2: 2D 시각화 (Stage 1에서 복사)
# ====================================
def visualize_data_2d(rgb_image: np.ndarray, depth_map: np.ndarray, show: bool = True) -> None:
    """2D 데이터 시각화"""
    
    if not show:
        return
    
    # Depth 맵 정규화 및 컬러맵 적용
    depth_norm = ((depth_map - depth_map.min()) / 
                  (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_VIRIDIS)
    
    # RGB를 BGR로 변환
    vis_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
    # 두 이미지 합치기
    combined = np.hstack([vis_bgr, depth_colored])
    
    # 제목 추가
    titled = np.zeros((combined.shape[0] + 40, combined.shape[1], 3), dtype=np.uint8)
    titled[40:] = combined
    titled[:40] = (50, 50, 50)
    
    cv2.putText(titled, 'RGB Image', (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(titled, 'Depth Map', (rgb_image.shape[1] + 10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    try:
        cv2.imshow('Stage 2: RGB-D Data', titled)
        print("  → 아무 키나 누르면 계속됩니다...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except (RuntimeError, ImportError) as e:
        save_path = Path("output") / "stage2_2d_visualization.png"
        save_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(save_path), titled)
        print(f"  → 이미지 저장: {save_path}")

# ====================================
# ★ NEW: Step 3 - Pinhole 카메라 모델로 3D 변환
# ====================================
def depth_to_pointcloud(depth_map: np.ndarray, rgb_image: np.ndarray, intrinsics) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D 픽셀 + 깊이 → 3D 카메라 좌표 변환
    
    핵심 공식 (Pinhole Camera Model):
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = depth
    
    여기서:
    - (u, v): 픽셀 좌표
    - (cx, cy): 주점 (principal point) - 이미지 중심
    - (fx, fy): 초점 거리 (focal length) - 픽셀 단위
    - Z: 깊이값
    - (X, Y, Z): 3D 카메라 좌표
    
    학습 포인트:
    - 카메라 중심이 원점
    - Z축이 카메라 전방
    - X축이 오른쪽, Y축이 아래
    """
    
    print("\n[3D 변환 과정]")
    print(f"  카메라 내부 파라미터:")
    print(f"    fx = {intrinsics.fx:.2f} pixels (초점거리 X)")
    print(f"    fy = {intrinsics.fy:.2f} pixels (초점거리 Y)")
    print(f"    cx = {intrinsics.cx:.2f} pixels (주점 X)")
    print(f"    cy = {intrinsics.cy:.2f} pixels (주점 Y)")
    
    height, width = depth_map.shape
    
    # 픽셀 좌표 메시그리드 생성
    u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))
    
    # Pinhole 카메라 모델 적용
    z = depth_map  # 깊이값 그대로 사용
    x = (u_grid - intrinsics.cx) * z / intrinsics.fx
    y = (v_grid - intrinsics.cy) * z / intrinsics.fy
    
    # 3D 포인트 생성 (N×3 배열로 변환)
    points_3d = np.stack([x, y, z], axis=-1)
    points_3d = points_3d.reshape(-1, 3)
    
    # 색상 정보
    colors = rgb_image.reshape(-1, 3)
    
    # 유효한 depth만 선택 (깊이가 0인 픽셀 제외)
    valid_mask = depth_map.flatten() > 0
    points_3d = points_3d[valid_mask]
    colors = colors[valid_mask]
    
    print(f"\n  변환 결과:")
    print(f"    입력 픽셀: {width}×{height} = {width*height:,}")
    print(f"    유효 포인트: {len(points_3d):,}")
    print(f"    제외된 포인트: {width*height - len(points_3d):,} (깊이=0)")
    
    return points_3d, colors

# ====================================
# ★ NEW: Step 4 - 포인트 클라우드 분석
# ====================================
def analyze_pointcloud(points_3d: np.ndarray) -> None:
    """
    3D 포인트 클라우드 분석
    
    학습 포인트:
    - 카메라 좌표계의 특성
    - 3D 포인트 분포
    - 중심점과 범위
    """
    
    print("\n[포인트 클라우드 분석]")
    print(f"  총 포인트 수: {len(points_3d):,}")
    
    # 중심점 계산
    centroid = points_3d.mean(axis=0)
    print(f"\n  중심점 (카메라 좌표계):")
    print(f"    X: {centroid[0]:7.3f}m (오른쪽+)")
    print(f"    Y: {centroid[1]:7.3f}m (아래+)")
    print(f"    Z: {centroid[2]:7.3f}m (전방+)")
    
    # 범위 계산
    print(f"\n  좌표 범위:")
    print(f"    X: [{points_3d[:,0].min():6.3f}, {points_3d[:,0].max():6.3f}]m")
    print(f"    Y: [{points_3d[:,1].min():6.3f}, {points_3d[:,1].max():6.3f}]m")
    print(f"    Z: [{points_3d[:,2].min():6.3f}, {points_3d[:,2].max():6.3f}]m (깊이)")
    
    # 가장 가까운/먼 점
    distances = np.linalg.norm(points_3d, axis=1)
    print(f"\n  카메라로부터의 거리:")
    print(f"    최소: {distances.min():.3f}m")
    print(f"    최대: {distances.max():.3f}m")
    print(f"    평균: {distances.mean():.3f}m")

# ====================================
# ★ NEW: Step 5 - 3D 시각화
# ====================================
def visualize_3d_camera(points_3d: np.ndarray, colors: np.ndarray, title: str = "Camera Coordinate System") -> None:
    """
    카메라 좌표계에서 3D 포인트 클라우드 시각화
    
    학습 포인트:
    - 빨간색 축: X (오른쪽)
    - 초록색 축: Y (아래)
    - 파란색 축: Z (전방)
    - 카메라는 원점(0,0,0)에 위치
    """
    
    if not HAS_OPEN3D:
        print("  Open3D가 설치되지 않아 3D 시각화를 건너뜁니다.")
        return
    
    print("\n[3D 시각화]")
    print("  카메라 좌표계 표시")
    print("  • 빨간색 축: X (오른쪽)")
    print("  • 초록색 축: Y (아래)")
    print("  • 파란색 축: Z (전방)")
    
    # Point cloud 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    # 색상 정규화 (0-255 → 0-1)
    if colors is not None:
        if colors.max() > 1:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 좌표축 추가 (크기 0.5m)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    
    # 시각화
    o3d.visualization.draw_geometries(
        [pcd, axes],
        window_name=f"Stage 2: {title}",
        width=1024, height=768,
        point_show_normal=False
    )

# ====================================
# 메인 함수
# ====================================
def main() -> None:
    """Stage 2 메인 실행 함수"""
    
    print("=" * 60)
    print("Stage 2: 3D 변환과 카메라 좌표계")
    print("=" * 60)
    print("\n학습 목표:")
    print("  1. Pinhole 카메라 모델 이해")
    print("  2. 2D 픽셀을 3D 카메라 좌표로 변환")
    print("  3. 포인트 클라우드 생성과 시각화")
    
    # Step 1: 데이터 로드
    t = print_step(1, "RGB-D 데이터 로드 중")
    frame_id = 53
    rgb_image, depth_map = load_data(frame_id)
    print_done(t)
    
    # Step 2: 2D 시각화 (간단히)
    t = print_step(2, "2D 데이터 확인")
    visualize_data_2d(rgb_image, depth_map, show=True)
    print_done(t)
    
    # ★ Step 3: 3D 변환 (NEW)
    t = print_step(3, "Pinhole 카메라 모델로 3D 변환")
    
    # 카메라 내부 파라미터 생성
    intrinsics = camera_utils.CameraIntrinsics()
    
    # 2D → 3D 변환
    points_3d, colors = depth_to_pointcloud(depth_map, rgb_image, intrinsics)
    print_done(t)
    
    # ★ Step 4: 포인트 클라우드 분석 (NEW)
    t = print_step(4, "포인트 클라우드 분석")
    analyze_pointcloud(points_3d)
    print_done(t)
    
    # ★ Step 5: 3D 시각화 (NEW)
    if HAS_OPEN3D:
        t = print_step(5, "3D 포인트 클라우드 시각화")
        visualize_3d_camera(points_3d, colors)
        print_done(t)
    
    # 완료
    print(f"\n{'=' * 60}")
    print("✓ Stage 2 완료!")
    print("=" * 60)
    print("\n핵심 공식 (Pinhole Camera Model):")
    print("  X = (u - cx) * Z / fx")
    print("  Y = (v - cy) * Z / fy")
    print("  Z = depth")
    print("\n생성된 결과:")
    print(f"  • 3D 포인트: {len(points_3d):,}개")
    print(f"  • 카메라 좌표계: 원점(0,0,0)이 카메라 위치")
    print(f"  • Z축이 전방, X축이 오른쪽, Y축이 아래")
    print("\n다음 단계: Stage 3에서 월드 좌표와 ROS 좌표로 변환합니다.")

if __name__ == "__main__":
    main()