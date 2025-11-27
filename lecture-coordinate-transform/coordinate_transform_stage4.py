#!/usr/bin/env python3
"""
Stage 4: 좌표계 비교 시각화와 결과 저장
- Stage 3 기능 포함
- 3개 좌표계 동시 비교 시각화 추가
- 결과 저장 기능 추가

새로운 학습 포인트:
- 좌표계 변환 검증
- 3개 좌표계 동시 비교
- 포인트 클라우드 저장 (PLY 형식)
"""

import os
import sys
import time
import json
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
    print(f"✓ 완룼 ({elapsed:.3f}초) {details}")

# ====================================
# Stage 1-3 기능들 (재사용)
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
    
    return rgb_image, depth_map

def depth_to_pointcloud(depth_map: np.ndarray, rgb_image: np.ndarray, intrinsics) -> Tuple[np.ndarray, np.ndarray]:
    """2D → 3D 카메라 좌표 변환"""
    height, width = depth_map.shape
    u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))
    
    z = depth_map
    x = (u_grid - intrinsics.cx) * z / intrinsics.fx
    y = (v_grid - intrinsics.cy) * z / intrinsics.fy
    
    points_3d = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = rgb_image.reshape(-1, 3)
    
    valid_mask = depth_map.flatten() > 0
    return points_3d[valid_mask], colors[valid_mask]

def get_camera_pose() -> Dict[str, Any]:
    """카메라 포즈 정의"""
    cam_rotation_x = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    cam_rotation_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    cam_rotation = cam_rotation_x @ cam_rotation_z
    
    r = Rotation.from_matrix(cam_rotation)
    camera_quaternion = r.as_quat()
    
    return {
        'position': np.array([0.5, 1.5, 0.0]),
        'orientation': camera_quaternion
    }

def transform_camera_to_world(points_camera: np.ndarray, camera_pose: Dict[str, Any]) -> np.ndarray:
    """카메라 → 월드 좌표 변환"""
    T = transform_utils.create_transformation_matrix(
        camera_pose['orientation'],
        camera_pose['position']
    )
    
    points_homogeneous = np.hstack([
        points_camera,
        np.ones((len(points_camera), 1))
    ])
    
    points_world_homogeneous = points_homogeneous @ T.T
    return points_world_homogeneous[:, :3]

def isaac_to_ros_points(points_isaac: np.ndarray) -> np.ndarray:
    """Isaac Sim → ROS 좌표 변환"""
    points_ros = np.zeros_like(points_isaac)
    points_ros[:, 0] = points_isaac[:, 0]
    points_ros[:, 1] = -points_isaac[:, 2]
    points_ros[:, 2] = points_isaac[:, 1]
    return points_ros

# ====================================
# ★ Step 8 - 좌표계 비교 시각화
# ====================================
def visualize_comparison(points_camera: np.ndarray, points_world: np.ndarray, points_ros: np.ndarray) -> None:
    """
    3개 좌표계 동시 비교 시각화
    
    학습 포인트:
    - 빨간색: 카메라 좌표계
    - 초록색: 월드 좌표계 (Isaac Sim)
    - 파란색: ROS 좌표계
    - 각 좌표계를 옆으로 배치하여 비교
    """
    
    if not HAS_OPEN3D:
        print("  Open3D가 설치되지 않아 비교 시각화를 건너뜁니다.")
        return
    
    print("\n[좌표계 비교 시각화]")
    print("  빨간색: 카메라 좌표계 (원점이 카메라)")
    print("  초록색: 월드 좌표계 (Isaac Sim Y-up)")
    print("  파란색: ROS 좌표계 (Z-up)")
    print("\n  각 좌표계를 1m씩 옆으로 배치하여 비교")
    
    geometries = []
    
    # 카메라 좌표계 (빨간색) - 위치 그대로
    pcd_camera = o3d.geometry.PointCloud()
    pcd_camera.points = o3d.utility.Vector3dVector(points_camera)
    pcd_camera.paint_uniform_color([1, 0.2, 0.2])  # 빨간색
    geometries.append(pcd_camera)
    
    # 카메라 좌표축
    axes_camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    geometries.append(axes_camera)
    
    # 월드 좌표계 (초록색) - 오른쪽으로 2m 이동
    pcd_world = o3d.geometry.PointCloud()
    pcd_world.points = o3d.utility.Vector3dVector(points_world + np.array([2, 0, 0]))
    pcd_world.paint_uniform_color([0.2, 1, 0.2])  # 초록색
    geometries.append(pcd_world)
    
    # 월드 좌표축
    axes_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    axes_world.translate([2, 0, 0])
    geometries.append(axes_world)
    
    # ROS 좌표계 (파란색) - 오른쪽으로 4m 이동
    pcd_ros = o3d.geometry.PointCloud()
    pcd_ros.points = o3d.utility.Vector3dVector(points_ros + np.array([4, 0, 0]))
    pcd_ros.paint_uniform_color([0.2, 0.2, 1])  # 파란색
    geometries.append(pcd_ros)
    
    # ROS 좌표축
    axes_ros = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    axes_ros.translate([4, 0, 0])
    geometries.append(axes_ros)
    
    # 라벨 추가 (구체로 표시)
    sphere_camera = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    sphere_camera.translate([0, -0.5, 0])
    sphere_camera.paint_uniform_color([1, 0, 0])
    geometries.append(sphere_camera)
    
    sphere_world = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    sphere_world.translate([2, -0.5, 0])
    sphere_world.paint_uniform_color([0, 1, 0])
    geometries.append(sphere_world)
    
    sphere_ros = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    sphere_ros.translate([4, -0.5, 0])
    sphere_ros.paint_uniform_color([0, 0, 1])
    geometries.append(sphere_ros)
    
    # 시각화
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Stage 4: Coordinate Systems Comparison",
        width=1280, height=720,
        point_show_normal=False
    )

# ====================================
# ★ Step 9 - 결과 저장
# ====================================
def save_results(points_camera: np.ndarray, points_world: np.ndarray, points_ros: np.ndarray, colors: np.ndarray, camera_pose: Dict[str, Any], frame_id: int) -> None:
    """
    변환 결과 저장
    
    저장 내용:
    - PLY 파일: 3D 포인트 클라우드
    - NPY 파일: NumPy 배열
    - JSON 파일: 통계와 메타데이터
    """
    
    print("\n[결과 저장]")
    
    # 출력 디렉토리
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. PLY 파일 저장 (3D 포인트 클라우드)
    if HAS_OPEN3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_ros)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        
        ply_path = output_dir / f"pointcloud_frame{frame_id:04d}.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd)
        print(f"  ✓ PLY 저장: {ply_path}")
    
    # 2. NumPy 배열 저장
    np_path = output_dir / "points_ros.npy"
    np.save(str(np_path), points_ros)
    print(f"  ✓ NumPy 저장: {np_path}")
    
    # 3. 통계 JSON 저장
    stats = {
        'frame_id': frame_id,
        'num_points': len(points_camera),
        'centroid_camera': points_camera.mean(axis=0).tolist(),
        'centroid_world': points_world.mean(axis=0).tolist(),
        'centroid_ros': points_ros.mean(axis=0).tolist(),
        'camera_pose': {
            'position': camera_pose['position'].tolist(),
            'orientation': camera_pose['orientation'].tolist()
        },
        'coordinate_systems': {
            'camera': 'X=right, Y=down, Z=forward',
            'world': 'Isaac Sim Y-up: X=right, Y=up, Z=forward',
            'ros': 'Z-up: X=forward, Y=left, Z=up'
        }
    }
    
    stats_path = output_dir / "transform_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ 통계 저장: {stats_path}")
    
    return stats

# ====================================
# ★ Step 10 - 최종 요약
# ====================================
def print_summary(stats: Dict[str, Any]) -> None:
    """
    변환 결과 요약
    
    학습 포인트:
    - 각 좌표계의 특성 확인
    - 변환의 정확성 검증
    - 테이블 높이 등 주요 지표
    """
    
    print(f"\n{'=' * 60}")
    print("변환 결과 요약")
    print("=" * 60)
    
    print(f"\n총 포인트 수: {stats['num_points']:,}")
    
    print("\n중심점 비교:")
    print(f"  카메라: [{stats['centroid_camera'][0]:.3f}, "
          f"{stats['centroid_camera'][1]:.3f}, "
          f"{stats['centroid_camera'][2]:.3f}]")
    print(f"  월드:   [{stats['centroid_world'][0]:.3f}, "
          f"{stats['centroid_world'][1]:.3f}, "
          f"{stats['centroid_world'][2]:.3f}]")
    print(f"  ROS:    [{stats['centroid_ros'][0]:.3f}, "
          f"{stats['centroid_ros'][1]:.3f}, "
          f"{stats['centroid_ros'][2]:.3f}]")
    
    print("\n주요 확인 사항:")
    print(f"  ✓ 카메라 Z (깊이): {stats['centroid_camera'][2]:.3f}m")
    print(f"  ✓ 월드 Y (높이):   {stats['centroid_world'][1]:.3f}m")
    print(f"  ✓ ROS Z (높이):    {stats['centroid_ros'][2]:.3f}m")
    
    # 테이블 높이는 약 0.38m
    expected_table_height = 0.38
    ros_z = stats['centroid_ros'][2]
    error = abs(ros_z - expected_table_height)
    
    if error < 0.05:
        print(f"\n  ✅ 테이블 높이 검증: {ros_z:.3f}m (예상: {expected_table_height}m)")
    else:
        print(f"\n  ⚠️ 테이블 높이 확인 필요: {ros_z:.3f}m (예상: {expected_table_height}m)")

# ====================================
# 메인 함수
# ====================================
def main() -> None:
    """Stage 4 메인 실행 함수"""
    
    print("=" * 60)
    print("Stage 4: 좌표계 비교와 결과 저장")
    print("=" * 60)
    print("\n학습 목표:")
    print("  1. 3개 좌표계 동시 비교")
    print("  2. 변환 결과 검증")
    print("  3. 포인트 클라우드 저장")
    
    # Step 1: 데이터 로드
    t = print_step(1, "RGB-D 데이터 로드 중")
    frame_id = 53
    rgb_image, depth_map = load_data(frame_id)
    print_done(t)
    
    # Step 2: 3D 변환
    t = print_step(2, "3D 변환 중")
    intrinsics = camera_utils.CameraIntrinsics()
    points_camera, colors = depth_to_pointcloud(depth_map, rgb_image, intrinsics)
    print(f"  생성된 포인트: {len(points_camera):,}개")
    print_done(t)
    
    # Step 3: 카메라 → 월드 변환
    t = print_step(3, "카메라 → 월드 좌표 변환")
    camera_pose = get_camera_pose()
    points_world = transform_camera_to_world(points_camera, camera_pose)
    print_done(t)
    
    # Step 4: Isaac Sim → ROS 변환
    t = print_step(4, "Isaac Sim → ROS 좌표 변환")
    points_ros = isaac_to_ros_points(points_world)
    print_done(t)
    
    # ★ Step 5: 좌표계 비교 시각화 (NEW)
    if HAS_OPEN3D:
        t = print_step(5, "좌표계 비교 시각화")
        visualize_comparison(points_camera, points_world, points_ros)
        print_done(t)
    
    # ★ Step 6: 최종 ROS 좌표계 시각화
    if HAS_OPEN3D:
        t = print_step(6, "최종 ROS 좌표계 시각화")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_ros)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        
        o3d.visualization.draw_geometries(
            [pcd, axes],
            window_name="Stage 4: Final ROS Point Cloud",
            width=1024, height=768
        )
        print_done(t)
    
    # ★ Step 7: 결과 저장 (NEW)
    t = print_step(7, "결과 저장 중")
    stats = save_results(points_camera, points_world, points_ros, colors, camera_pose, frame_id)
    print_done(t)
    
    # ★ Step 8: 최종 요약 (NEW)
    print_summary(stats)
    
    # 완료
    print(f"\n{'=' * 60}")
    print("✓ Stage 4 완료! (전체 파이프라인)")
    print("=" * 60)
    print("\n전체 변환 체인:")
    print("  1. 픽셀 + 깊이 → 카메라 3D (Pinhole model)")
    print("  2. 카메라 3D → 월드 3D (4x4 변환)")
    print("  3. 월드 3D → ROS 3D (좌표계 변환)")
    print("\n저장된 파일:")
    print("  • output/stage4_pointcloud_frame0053.ply")
    print("  • output/stage4_points_ros.npy")
    print("  • output/stage4_transform_stats.json")
    print("\n🎉 좌표 변환 파이프라인 실습 완료!")

if __name__ == "__main__":
    main()