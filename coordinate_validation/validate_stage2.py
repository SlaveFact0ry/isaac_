#!/usr/bin/env python3
"""
Stage 2: 통계 분석 추가
- Stage 1 기능 포함
- 좌표계별 중심점 비교
- Z 분포 히스토그램
- 상세 통계 분석

새로운 학습 포인트:
- 변환 결과 통계적 검증
- 분포 분석을 통한 이상치 탐지
- matplotlib 히스토그램
"""

import numpy as np
import open3d as o3d
import json
from pathlib import Path
import time
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt

def print_step(step_num: int, description: str) -> float:
    """단계별 출력 헬퍼"""
    print(f"\n[Step {step_num}] {description}...")
    return time.time()

def print_done(start_time: float, details: str = "") -> None:
    """완료 메시지 출력"""
    elapsed = time.time() - start_time
    print(f"✓ 완료 ({elapsed:.3f}초) {details}")

# ====================================
# Stage 1 기능들 (재사용)
# ====================================
def load_validation_data() -> Tuple[np.ndarray, Dict[str, Any]]:
    """검증용 데이터 로드"""
    output_dir = Path(__file__).parent.parent / "lecture-coordinate-transform" / "output"
    
    points_ros_path = output_dir / "points_ros.npy"
    pointcloud_path = output_dir / "pointcloud_frame0053.ply"
    stats_path = output_dir / "transform_stats.json"
    
    points_ros = np.load(points_ros_path)
    pcd = o3d.io.read_point_cloud(str(pointcloud_path))
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    print(f"  데이터 로드 완료: {len(points_ros):,}개 포인트")
    
    return points_ros, pcd, stats

def visualize_with_axes(pcd, show: bool = True) -> None:
    """포인트 클라우드와 좌표축 시각화"""
    if not show:
        return
        
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0]
    )
    
    o3d.visualization.draw_geometries(
        [pcd, coord_frame],
        window_name="Stage 2: 포인트 클라우드 시각화",
        width=1024, height=768,
        point_show_normal=False
    )

# ====================================
# ★ NEW: Step 4 - 좌표계별 중심점 비교
# ====================================
def compare_centroids(stats: Dict[str, Any]) -> None:
    """
    각 좌표계의 중심점 비교
    
    학습 포인트:
    - 카메라 → 월드 → ROS 변환 추적
    - 좌표계별 특성 확인
    """
    
    print("\n[좌표계별 중심점 비교]")
    print("-" * 40)
    
    # 카메라 좌표계
    centroid_cam = stats['centroid_camera']
    print(f"카메라 좌표계:")
    print(f"  X: {centroid_cam[0]:7.4f}m (오른쪽+)")
    print(f"  Y: {centroid_cam[1]:7.4f}m (아래+)")
    print(f"  Z: {centroid_cam[2]:7.4f}m (전방+)")
    
    # ROS 좌표계
    centroid_ros = stats['centroid_ros']
    print(f"\nROS 좌표계:")
    print(f"  X: {centroid_ros[0]:7.4f}m (전방+)")
    print(f"  Y: {centroid_ros[1]:7.4f}m (좌측+)")
    print(f"  Z: {centroid_ros[2]:7.4f}m (상방+)")
    
    # 주요 변환 확인
    print(f"\n변환 확인:")
    print(f"  카메라 Z (깊이) {centroid_cam[2]:.3f}m")
    print(f"  → ROS X (전방) {centroid_ros[0]:.3f}m 근처")
    print(f"  테이블 높이 (ROS Z): {centroid_ros[2]:.3f}m")

# ====================================
# ★ NEW: Step 5 - Z 분포 분석
# ====================================
def analyze_z_distribution(points_ros: np.ndarray) -> None:
    """
    Z 좌표 (높이) 분포 분석
    
    학습 포인트:
    - 높이 분포로 평면 탐지
    - 히스토그램을 통한 시각화
    - 통계적 이상치 탐지
    """
    
    print("\n[Z 좌표 분포 분석]")
    
    z_values = points_ros[:, 2]
    
    # 기본 통계
    print(f"  최소 높이: {z_values.min():.3f}m")
    print(f"  최대 높이: {z_values.max():.3f}m")
    print(f"  평균 높이: {z_values.mean():.3f}m")
    print(f"  중앙값: {np.median(z_values):.3f}m")
    print(f"  표준편차: {z_values.std():.3f}m")
    
    # 주요 높이 구간 분석
    print(f"\n  높이별 포인트 수:")
    height_ranges = [
        (0.0, 0.1, "바닥 근처"),
        (0.35, 0.41, "테이블 표면"),
        (0.41, 0.6, "테이블 위 물체"),
        (0.6, 1.0, "높은 물체")
    ]
    
    for z_min, z_max, description in height_ranges:
        count = np.sum((z_values >= z_min) & (z_values <= z_max))
        percentage = count / len(z_values) * 100
        if count > 0:
            print(f"    {description} ({z_min:.2f}~{z_max:.2f}m): {count:,}개 ({percentage:.1f}%)")
    
    return z_values

# ====================================
# ★ NEW: Step 6 - 히스토그램 시각화
# ====================================
def plot_z_histogram(z_values: np.ndarray) -> None:
    """
    Z 분포 히스토그램 그리기
    
    학습 포인트:
    - matplotlib 히스토그램
    - 테이블 높이 표시
    - 분포 패턴 해석
    """
    
    print("\n[히스토그램 생성]")
    
    plt.figure(figsize=(10, 6))
    
    # 히스토그램 그리기
    n, bins, patches = plt.hist(z_values, bins=50, 
                                edgecolor='black', alpha=0.7,
                                color='steelblue')
    
    # 테이블 높이 표시
    expected_table_height = 0.38
    plt.axvline(expected_table_height, color='red', linestyle='--', 
                linewidth=2, label=f'Expected Table Height ({expected_table_height}m)')
    
    # 실제 피크 위치 찾기
    peak_idx = np.argmax(n)
    peak_height = (bins[peak_idx] + bins[peak_idx + 1]) / 2
    plt.axvline(peak_height, color='green', linestyle=':', 
                linewidth=2, label=f'Actual Peak ({peak_height:.3f}m)')
    
    plt.xlabel('Z Coordinate (Height, m)', fontsize=12)
    plt.ylabel('Number of Points', fontsize=12)
    plt.title('ROS Coordinate System Z Distribution', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    stats_text = f'Mean: {z_values.mean():.3f}m\n'
    stats_text += f'Median: {np.median(z_values):.3f}m\n'
    stats_text += f'Std Dev: {z_values.std():.3f}m'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    print(f"  피크 높이: {peak_height:.3f}m")
    print(f"  예상 높이와 차이: {abs(peak_height - expected_table_height)*100:.1f}cm")

# ====================================
# ★ NEW: Step 7 - 상세 통계
# ====================================
def detailed_statistics(points_ros: np.ndarray, stats: Dict[str, Any]) -> None:
    """
    상세 통계 분석
    
    학습 포인트:
    - 변환 전후 비교
    - 포인트 밀도 분석
    - 데이터 품질 지표
    """
    
    print("\n[상세 통계 분석]")
    print("-" * 40)
    
    # 포인트 개수 비교
    print(f"포인트 수:")
    print(f"  총 포인트: {stats['num_points']:,}")
    
    # 공간 범위
    x_range = np.ptp(points_ros[:, 0])
    y_range = np.ptp(points_ros[:, 1])
    z_range = np.ptp(points_ros[:, 2])
    volume = x_range * y_range * z_range
    
    print(f"\n공간 범위:")
    print(f"  X 범위: {x_range:.3f}m")
    print(f"  Y 범위: {y_range:.3f}m")
    print(f"  Z 범위: {z_range:.3f}m")
    print(f"  부피: {volume:.3f}m³")
    
    # 포인트 밀도
    density = len(points_ros) / volume
    print(f"\n포인트 밀도:")
    print(f"  {density:.0f} points/m³")
    print(f"  평균 간격: {(volume/len(points_ros))**(1/3)*1000:.1f}mm")

# ====================================
# 메인 함수
# ====================================
def main() -> None:
    """Stage 2 메인 실행 함수"""
    
    print("=" * 60)
    print("Stage 2: 통계 분석 추가")
    print("=" * 60)
    print("\n학습 목표:")
    print("  1. 변환 결과 통계적 검증")
    print("  2. Z 분포 분석으로 평면 확인")
    print("  3. 좌표계 간 변환 추적")
    
    try:
        # Step 1: 데이터 로드
        t = print_step(1, "데이터 로드 중")
        points_ros, pcd, stats = load_validation_data()
        print_done(t)
        
        # Step 2: 기본 시각화
        t = print_step(2, "포인트 클라우드 확인")
        visualize_with_axes(pcd, show=True)
        print_done(t)
        
        # ★ Step 3: 좌표계 비교 (NEW)
        t = print_step(3, "좌표계별 중심점 비교")
        compare_centroids(stats)
        print_done(t)
        
        # ★ Step 4: Z 분포 분석 (NEW)
        t = print_step(4, "Z 좌표 분포 분석")
        z_values = analyze_z_distribution(points_ros)
        print_done(t)
        
        # ★ Step 5: 히스토그램 (NEW)
        t = print_step(5, "히스토그램 생성")
        plot_z_histogram(z_values)
        print_done(t)
        
        # ★ Step 6: 상세 통계 (NEW)
        t = print_step(6, "상세 통계 계산")
        detailed_statistics(points_ros, stats)
        print_done(t)
        
    except FileNotFoundError as e:
        print(f"\n❌ 오류: {e}")
        print("\n데이터를 먼저 생성하세요.")
        return
    
    # 완료
    print("\n" + "=" * 60)
    print("✓ Stage 2 완료!")
    print("=" * 60)
    print("\n핵심 발견:")
    print("  • 테이블 높이가 Z 히스토그램에서 명확한 피크")
    print("  • 카메라 Z축 깊이 → ROS Z축 높이로 정확히 변환")
    print("  • 포인트 밀도로 데이터 품질 확인 가능")
    print("\n다음 단계: Stage 3에서 RANSAC 평면 검출을 추가합니다.")

if __name__ == "__main__":
    main()