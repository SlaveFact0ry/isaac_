#!/usr/bin/env python3
"""
Stage 1: 데이터 로드와 기본 시각화
- 변환된 포인트 클라우드 데이터 로드
- 좌표축과 함께 시각화
- 기본 정보 출력

학습 포인트:
- 검증 데이터 구조 (NPY, PLY, JSON)
- Open3D 기본 시각화
- ROS 좌표계 확인
"""

import numpy as np
import open3d as o3d
import json
from pathlib import Path
import time
from typing import Tuple, Dict, Any

def print_step(step_num: int, description: str) -> float:
    """단계별 출력 헬퍼"""
    print(f"\n[Step {step_num}] {description}...")
    return time.time()

def print_done(start_time: float, details: str = "") -> None:
    """완료 메시지 출력"""
    elapsed = time.time() - start_time
    print(f"✓ 완료 ({elapsed:.3f}초) {details}")

# ====================================
# Step 1: 데이터 로드
# ====================================
def load_validation_data() -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    검증용 데이터 로드
    
    필요한 파일:
    - points_ros.npy: ROS 좌표계 포인트
    - pointcloud_frame0053.ply: 색상 포함 포인트 클라우드
    - transform_stats.json: 변환 통계
    """
    
    output_dir = Path(__file__).parent.parent / "lecture-coordinate-transform" / "output"
    
    # 파일 경로
    points_ros_path = output_dir / "points_ros.npy"
    pointcloud_path = output_dir / "pointcloud_frame0053.ply"
    stats_path = output_dir / "transform_stats.json"
    
    # 파일 존재 확인
    if not points_ros_path.exists():
        raise FileNotFoundError(f"ROS 포인트 파일을 찾을 수 없습니다: {points_ros_path}")
    if not pointcloud_path.exists():
        raise FileNotFoundError(f"PLY 파일을 찾을 수 없습니다: {pointcloud_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"통계 파일을 찾을 수 없습니다: {stats_path}")
    
    # 데이터 로드
    points_ros = np.load(points_ros_path)
    pcd = o3d.io.read_point_cloud(str(pointcloud_path))
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    print(f"  ROS 포인트: {len(points_ros):,}개")
    print(f"  PLY 포인트: {len(pcd.points):,}개")
    print(f"  카메라 위치: {stats['camera_pose']['position']}")
    
    return points_ros, pcd, stats

# ====================================
# Step 2: 좌표축 시각화
# ====================================
def visualize_with_axes(pcd) -> None:
    """
    포인트 클라우드와 좌표축 시각화
    
    학습 포인트:
    - ROS 좌표계: X=전방, Y=좌측, Z=상방
    - 축 색상: 빨강(X), 초록(Y), 파랑(Z)
    """
    
    print("\n[좌표축 설명]")
    print("  빨간색 축: X (전방)")
    print("  초록색 축: Y (좌측)")
    print("  파란색 축: Z (상방)")
    print("\n  마우스 조작:")
    print("  - 왼쪽 드래그: 회전")
    print("  - 스크롤: 확대/축소")
    print("  - 휠 클릭 드래그: 이동")
    
    # 좌표축 생성 (0.5m 크기)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0]
    )
    
    # 시각화
    o3d.visualization.draw_geometries(
        [pcd, coord_frame],
        window_name="Stage 1: 포인트 클라우드와 좌표축",
        width=1024, height=768,
        point_show_normal=False
    )

# ====================================
# Step 3: 기본 정보 출력
# ====================================
def print_basic_info(points_ros: np.ndarray, stats: Dict[str, Any]) -> None:
    """
    기본 통계 정보 출력
    
    학습 포인트:
    - 포인트 개수와 범위
    - 중심점 위치
    - 카메라 포즈 정보
    """
    
    print("\n[기본 정보]")
    print(f"  총 포인트 수: {len(points_ros):,}")
    
    # 좌표 범위
    print(f"\n  좌표 범위:")
    print(f"    X: [{points_ros[:,0].min():.3f}, {points_ros[:,0].max():.3f}]m")
    print(f"    Y: [{points_ros[:,1].min():.3f}, {points_ros[:,1].max():.3f}]m")
    print(f"    Z: [{points_ros[:,2].min():.3f}, {points_ros[:,2].max():.3f}]m")
    
    # 중심점
    centroid = points_ros.mean(axis=0)
    print(f"\n  중심점 (ROS):")
    print(f"    X: {centroid[0]:.3f}m (전방)")
    print(f"    Y: {centroid[1]:.3f}m (좌측)")
    print(f"    Z: {centroid[2]:.3f}m (상방)")
    
    # 카메라 정보
    cam_pos = stats['camera_pose']['position']
    print(f"\n  카메라 위치 (Isaac Sim):")
    print(f"    X: {cam_pos[0]:.1f}m")
    print(f"    Y: {cam_pos[1]:.1f}m (높이)")
    print(f"    Z: {cam_pos[2]:.1f}m")

# ====================================
# 메인 함수
# ====================================
def main() -> None:
    """Stage 1 메인 실행 함수"""
    
    print("=" * 60)
    print("Stage 1: 데이터 로드와 기본 시각화")
    print("=" * 60)
    print("\n학습 목표:")
    print("  1. 변환된 데이터 구조 이해")
    print("  2. ROS 좌표계 확인")
    print("  3. Open3D 기본 시각화")
    
    try:
        # Step 1: 데이터 로드
        t = print_step(1, "검증 데이터 로드 중")
        points_ros, pcd, stats = load_validation_data()
        print_done(t)
        
        # Step 2: 기본 정보
        t = print_step(2, "기본 정보 분석")
        print_basic_info(points_ros, stats)
        print_done(t)
        
        # Step 3: 시각화
        t = print_step(3, "좌표축과 함께 시각화")
        visualize_with_axes(pcd)
        print_done(t)
        
    except FileNotFoundError as e:
        print(f"\n❌ 오류: {e}")
        print("\n먼저 coordinate_transform_pipeline.py를 실행하여 데이터를 생성하세요:")
        print("  cd ..")
        print("  python3 coordinate_transform_pipeline.py")
        return
    
    # 완료
    print("\n" + "=" * 60)
    print("✓ Stage 1 완료!")
    print("=" * 60)
    print("\n핵심 포인트:")
    print("  • NPY: NumPy 배열 형식 (빠른 로드)")
    print("  • PLY: 포인트 클라우드 표준 형식 (색상 포함)")
    print("  • JSON: 메타데이터 (통계, 카메라 포즈)")
    print("  • ROS 좌표계: Z축이 위 (Z-up)")
    print("\n다음 단계: Stage 2에서 통계 분석을 추가합니다.")

if __name__ == "__main__":
    main()