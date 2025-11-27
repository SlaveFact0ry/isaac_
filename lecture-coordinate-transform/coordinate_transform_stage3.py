#!/usr/bin/env python3
"""
Stage 3: 좌표계 변환 (카메라 → 월드 → ROS)
- Stage 2 기능 포함
- 카메라 → 월드 좌표 변환 추가
- Isaac Sim → ROS 좌표 변환 추가

새로운 학습 포인트:
- 4x4 변환 매트릭스 (회전 + 평행이동)
- 쿼터니언과 회전 매트릭스
- Isaac Sim (Y-up) vs ROS (Z-up) 좌표계
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import cv2
from typing import Tuple, Dict, Any

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
from utils import camera_utils, transform_utils

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
# Stage 1-2 기능들 (간소화)
# ====================================
def load_data(frame_id: int = 53) -> Tuple[np.ndarray, np.ndarray]:
    """RGB-D 데이터 로드"""
    rgb_path = DATA_BASE_PATH / "rgb" / f"rgb_{frame_id:04d}.png"
    depth_path = DATA_BASE_PATH / "distance_to_image_plane" / f"distance_to_image_plane_{frame_id:04d}.npy"
    
    rgb_image = cv2.imread(str(rgb_path))
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    depth_map = np.load(str(depth_path))
    
    print(f"  RGB shape: {rgb_image.shape}")
    print(f"  Depth shape: {depth_map.shape}")
    print(f"  Depth range: {depth_map.min():.3f} - {depth_map.max():.3f} meters")
    
    return rgb_image, depth_map

def depth_to_pointcloud(depth_map: np.ndarray, rgb_image: np.ndarray, intrinsics) -> Tuple[np.ndarray, np.ndarray]:
    """2D → 3D 카메라 좌표 변환"""
    height, width = depth_map.shape
    
    # 픽셀 좌표 메시그리드
    u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))
    
    # Pinhole 카메라 모델
    z = depth_map
    x = (u_grid - intrinsics.cx) * z / intrinsics.fx
    y = (v_grid - intrinsics.cy) * z / intrinsics.fy
    
    # 3D 포인트 생성
    points_3d = np.stack([x, y, z], axis=-1)
    points_3d = points_3d.reshape(-1, 3)
    colors = rgb_image.reshape(-1, 3)
    
    # 유효한 depth만 선택
    valid_mask = depth_map.flatten() > 0
    points_3d = points_3d[valid_mask]
    colors = colors[valid_mask]
    
    return points_3d, colors

# ====================================
# ★ NEW: Step 4 - 카메라 → 월드 좌표 변환
# ====================================
def get_camera_pose() -> Dict[str, Any]:
    """
    카메라 포즈 정의
    
    학습 포인트:
    - 카메라 위치: [0.5, 1.5, 0.0] (Y=1.5m 높이)
    - 카메라 방향: 아래를 보도록 회전
    - 쿼터니언으로 회전 표현
    """
    
    print("\n[카메라 포즈 설정]")
    
    # 카메라가 아래를 보려면:
    # 1. X축 기준 -90도 회전 (Y축이 -Z축으로)
    cam_rotation_x = np.array([
        [1, 0, 0],
        [0, 0, -1],  # Y → -Z
        [0, 1, 0]    # Z → Y
    ])
    
    # 2. Z축 기준 90도 회전 (카메라 정렬)
    cam_rotation_z = np.array([
        [0, -1, 0],  # X → -Y
        [1, 0, 0],   # Y → X
        [0, 0, 1]    # Z는 유지
    ])
    
    # 전체 회전 = Z 회전 후 X 회전
    cam_rotation = cam_rotation_x @ cam_rotation_z
    
    # Quaternion으로 변환
    r = Rotation.from_matrix(cam_rotation)
    camera_quaternion = r.as_quat()  # [x, y, z, w]
    
    camera_pose = {
        'position': np.array([0.5, 1.5, 0.0]),
        'orientation': camera_quaternion
    }
    
    print(f"  위치: {camera_pose['position']}")
    print(f"  방향 (quaternion): [{camera_quaternion[0]:.3f}, {camera_quaternion[1]:.3f}, "
          f"{camera_quaternion[2]:.3f}, {camera_quaternion[3]:.3f}]")
    
    return camera_pose

def transform_camera_to_world(points_camera: np.ndarray, camera_pose: Dict[str, Any]) -> np.ndarray:
    """
    카메라 좌표 → 월드 좌표 변환
    
    변환 공식:
    P_world = R * P_camera + t
    
    학습 포인트:
    - R: 3x3 회전 매트릭스
    - t: 3x1 평행이동 벡터
    - 4x4 동차 변환 매트릭스 사용
    """
    
    print("\n[카메라 → 월드 변환]")
    print("  변환 매트릭스 구성:")
    print("  T = [R  t]")
    print("      [0  1]")
    
    # 변환 매트릭스 생성
    T = transform_utils.create_transformation_matrix(
        camera_pose['orientation'],
        camera_pose['position']
    )
    
    # 동차 좌표로 변환
    points_homogeneous = np.hstack([
        points_camera,
        np.ones((len(points_camera), 1))
    ])
    
    # 변환 적용
    points_world_homogeneous = points_homogeneous @ T.T
    points_world = points_world_homogeneous[:, :3]
    
    print(f"  변환된 포인트: {len(points_world):,}개")
    
    return points_world

# ====================================
# ★ NEW: Step 5 - Isaac Sim → ROS 좌표 변환
# ====================================
def isaac_to_ros_points(points_isaac: np.ndarray) -> np.ndarray:
    """
    Isaac Sim (Y-up) → ROS (Z-up) 좌표 변환
    
    좌표계 차이:
    Isaac Sim: Y=위, X=오른쪽, Z=전방
    ROS:       Z=위, X=전방, Y=왼쪽
    
    변환 규칙:
    ROS_X = Isaac_X
    ROS_Y = -Isaac_Z
    ROS_Z = Isaac_Y
    """
    
    print("\n[Isaac Sim → ROS 변환]")
    print("  Isaac Sim (Y-up): Y=위, X=오른쪽, Z=전방")
    print("  ROS (Z-up):       Z=위, X=전방, Y=왼쪽")
    print("\n  변환 규칙:")
    print("  ROS_X = Isaac_X")
    print("  ROS_Y = -Isaac_Z")
    print("  ROS_Z = Isaac_Y")
    
    points_ros = np.zeros_like(points_isaac)
    points_ros[:, 0] = points_isaac[:, 0]   # X는 유지
    points_ros[:, 1] = -points_isaac[:, 2]  # Z → -Y
    points_ros[:, 2] = points_isaac[:, 1]   # Y → Z
    
    return points_ros

# ====================================
# ★ NEW: Step 6 - 좌표 분석
# ====================================
def analyze_coordinates(points_camera: np.ndarray, points_world: np.ndarray, points_ros: np.ndarray) -> None:
    """
    각 좌표계에서의 포인트 분석
    
    학습 포인트:
    - 각 좌표계의 중심점 비교
    - 좌표 범위 변화
    - 변환의 일관성 확인
    """
    
    print("\n[좌표계별 분석]")
    
    # 카메라 좌표계
    centroid_cam = points_camera.mean(axis=0)
    print(f"\n  카메라 좌표계 중심점:")
    print(f"    X: {centroid_cam[0]:7.3f}m (오른쪽+)")
    print(f"    Y: {centroid_cam[1]:7.3f}m (아래+)")
    print(f"    Z: {centroid_cam[2]:7.3f}m (전방+)")
    
    # 월드 좌표계 (Isaac Sim)
    centroid_world = points_world.mean(axis=0)
    print(f"\n  월드 좌표계 중심점 (Isaac Sim Y-up):")
    print(f"    X: {centroid_world[0]:7.3f}m")
    print(f"    Y: {centroid_world[1]:7.3f}m (위+)")
    print(f"    Z: {centroid_world[2]:7.3f}m")
    
    # ROS 좌표계
    centroid_ros = points_ros.mean(axis=0)
    print(f"\n  ROS 좌표계 중심점 (Z-up):")
    print(f"    X: {centroid_ros[0]:7.3f}m (전방+)")
    print(f"    Y: {centroid_ros[1]:7.3f}m (왼쪽+)")
    print(f"    Z: {centroid_ros[2]:7.3f}m (위+)")
    
    # 테이블 높이 확인 (ROS Z 좌표)
    z_values = points_ros[:, 2]
    table_points = z_values[(z_values > 0.35) & (z_values < 0.40)]
    if len(table_points) > 0:
        print(f"\n  테이블 높이 (ROS Z): {table_points.mean():.3f}m")

# ====================================
# ★ NEW: Step 7 - ROS 좌표계 시각화
# ====================================
def visualize_3d_ros(points_ros: np.ndarray, colors: np.ndarray, title: str = "ROS Coordinate System") -> None:
    """
    ROS 좌표계에서 포인트 클라우드 시각화
    
    학습 포인트:
    - 빨간색 축: X (전방)
    - 초록색 축: Y (왼쪽)
    - 파란색 축: Z (위)
    """
    
    if not HAS_OPEN3D:
        print("  Open3D가 설치되지 않아 3D 시각화를 건너뜁니다.")
        return
    
    print("\n[ROS 좌표계 시각화]")
    print("  • 빨간색 축: X (전방)")
    print("  • 초록색 축: Y (왼쪽)")
    print("  • 파란색 축: Z (위)")
    
    # Point cloud 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_ros)
    
    if colors is not None:
        if colors.max() > 1:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 좌표축 추가
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    
    # 시각화
    o3d.visualization.draw_geometries(
        [pcd, axes],
        window_name=f"Stage 3: {title}",
        width=1024, height=768,
        point_show_normal=False
    )

# ====================================
# 메인 함수
# ====================================
def main() -> None:
    """Stage 3 메인 실행 함수"""
    
    print("=" * 60)
    print("Stage 3: 좌표계 변환 (카메라 → 월드 → ROS)")
    print("=" * 60)
    print("\n학습 목표:")
    print("  1. 4x4 변환 매트릭스 이해")
    print("  2. 카메라 → 월드 좌표 변환")
    print("  3. Isaac Sim (Y-up) → ROS (Z-up) 변환")
    
    # Step 1: 데이터 로드
    t = print_step(1, "RGB-D 데이터 로드 중")
    frame_id = 53
    rgb_image, depth_map = load_data(frame_id)
    print_done(t)
    
    # Step 2: 3D 변환
    t = print_step(2, "3D 변환 중")
    intrinsics = camera_utils.CameraIntrinsics()
    points_3d_camera, colors = depth_to_pointcloud(depth_map, rgb_image, intrinsics)
    print(f"  생성된 포인트: {len(points_3d_camera):,}개")
    print_done(t)
    
    # Step 3: 카메라 좌표계 시각화
    if HAS_OPEN3D:
        t = print_step(3, "카메라 좌표계 확인")
        print("  시각화를 건너뜁니다 (Stage 2에서 확인)")
        print_done(t)
    
    # ★ Step 4: 카메라 → 월드 변환 (NEW)
    t = print_step(4, "카메라 → 월드 좌표 변환")
    camera_pose = get_camera_pose()
    points_3d_world = transform_camera_to_world(points_3d_camera, camera_pose)
    print_done(t)
    
    # ★ Step 5: Isaac Sim → ROS 변환 (NEW)
    t = print_step(5, "Isaac Sim → ROS 좌표 변환")
    points_3d_ros = isaac_to_ros_points(points_3d_world)
    print_done(t)
    
    # ★ Step 6: 좌표 분석 (NEW)
    t = print_step(6, "좌표계별 분석")
    analyze_coordinates(points_3d_camera, points_3d_world, points_3d_ros)
    print_done(t)
    
    # ★ Step 7: ROS 좌표계 시각화 (NEW)
    if HAS_OPEN3D:
        t = print_step(7, "ROS 좌표계 시각화")
        visualize_3d_ros(points_3d_ros, colors)
        print_done(t)
    
    # 완료
    print(f"\n{'=' * 60}")
    print("✓ Stage 3 완료!")
    print("=" * 60)
    print("\n변환 체인:")
    print("  픽셀 (u,v) + 깊이")
    print("      ↓ Pinhole model")
    print("  카메라 좌표 (카메라 중심)")
    print("      ↓ 4x4 변환 매트릭스")
    print("  월드 좌표 (Isaac Sim Y-up)")
    print("      ↓ 축 재배열")
    print("  ROS 좌표 (Z-up)")
    print("\n핵심 확인:")
    print("  • 테이블이 ROS Z=0.38m 높이에 위치")
    print("  • 모든 축이 올바른 방향")
    print("\n다음 단계: Stage 4에서 3개 좌표계를 동시에 비교합니다.")

if __name__ == "__main__":
    main()