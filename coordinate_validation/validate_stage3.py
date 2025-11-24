#!/usr/bin/env python3
"""
Stage 3: RANSAC 평면 검출 추가
- Stage 2 기능 포함
- RANSAC을 통한 자동 평면 검출
- 법선 벡터 계산과 수평도 분석
- 인라이어/아웃라이어 분류

새로운 학습 포인트:
- RANSAC 알고리즘 원리
- 평면 방정식과 법선 벡터
- 수평도 검증 방법
"""

import numpy as np
import open3d as o3d
import json
from pathlib import Path
import time
from typing import Tuple, Dict, Any, List

from numpy import ndarray, dtype
from sklearn.linear_model import RANSACRegressor
import warnings
warnings.filterwarnings('ignore')

def print_step(step_num: int, description: str) -> float:
    """단계별 출력 헬퍼"""
    print(f"\n[Step {step_num}] {description}...")
    return time.time()

def print_done(start_time: float, details: str = "") -> None:
    """완료 메시지 출력"""
    elapsed = time.time() - start_time
    print(f"✓ 완료 ({elapsed:.3f}초) {details}")

# ====================================
# Stage 1-2 기능들 (재사용)
# ====================================
def load_validation_data() -> tuple[Any, Any, Any]:
    """검증용 데이터 로드"""
    output_dir = Path(__file__).parent.parent / "lecture-coordinate-transform" / "output"
    
    points_ros_path = output_dir / "points_ros.npy"
    pointcloud_path = output_dir / "pointcloud_frame0053.ply"
    stats_path = output_dir / "transform_stats.json"
    
    points_ros = np.load(points_ros_path)
    pcd = o3d.io.read_point_cloud(str(pointcloud_path))
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    return points_ros, pcd, stats

def analyze_z_distribution(points_ros: np.ndarray) -> None:
    """Z 좌표 분포 분석"""
    z_values = points_ros[:, 2]
    
    print(f"  Z 범위: {z_values.min():.3f} ~ {z_values.max():.3f}m")
    print(f"  평균: {z_values.mean():.3f}m, 중앙값: {np.median(z_values):.3f}m")

# ====================================
# ★ NEW: Step 5 - RANSAC 평면 검출
# ====================================
def detect_planes_ransac(points_ros: np.ndarray) -> tuple[dict[str, Any], ndarray[Any, dtype[Any] | Any]] | tuple[
    None, None]:
    """
    RANSAC을 사용한 자동 평면 검출
    
    학습 포인트:
    - RANSAC: Random Sample Consensus
    - 노이즈에 강건한 모델 피팅
    - 인라이어/아웃라이어 개념
    """
    
    print("\n[RANSAC 평면 검출]")
    print("  RANSAC 파라미터:")
    print("    - min_samples: 3 (평면 정의 최소 점)")
    print("    - residual_threshold: 0.01m (1cm)")
    print("    - max_trials: 1000")
    
    # 높이별 평면 검출
    z_values = points_ros[:, 2]
    
    # 테이블 높이 범위 (0.35 ~ 0.41m)
    table_mask = (z_values >= 0.35) & (z_values <= 0.41)
    table_points = points_ros[table_mask]
    
    if len(table_points) > 100:
        print(f"\n  테이블 영역 포인트: {len(table_points):,}개")
        plane_params = fit_plane_ransac(table_points, "테이블 표면")
        return plane_params, table_points
    else:
        print("  테이블 영역에 충분한 포인트가 없습니다.")
        return None, None

# ====================================
# ★ NEW: Step 6 - 평면 피팅
# ====================================
def fit_plane_ransac(points: np.ndarray, plane_name: str = "평면") -> Dict[str, Any]:
    """
    RANSAC으로 평면 피팅
    
    평면 방정식: ax + by + cz + d = 0
    법선 벡터: n = [a, b, c]
    
    학습 포인트:
    - 평면 방정식의 의미
    - 법선 벡터 계산
    - 인라이어 비율의 중요성
    """
    
    print(f"\n  [{plane_name} 피팅]")
    
    # X, Y를 입력으로 Z를 예측
    X = points[:, :2]  # X, Y 좌표
    y = points[:, 2]    # Z 좌표
    
    # RANSAC 회귀
    ransac = RANSACRegressor(
        random_state=42,
        min_samples=3,
        residual_threshold=0.01  # 1cm 임계값
    )
    
    try:
        ransac.fit(X, y)
        
        # 평면 파라미터 추출
        # Z = aX + bY + c → aX + bY - Z + c = 0
        a = ransac.estimator_.coef_[0]
        b = ransac.estimator_.coef_[1]
        c = -1.0
        d = ransac.estimator_.intercept_
        
        # 정규화
        norm = np.sqrt(a**2 + b**2 + c**2)
        normal = np.array([a, b, c]) / norm
        d = d / norm
        
        # 인라이어 계산
        inliers = ransac.inlier_mask_
        num_inliers = np.sum(inliers)
        inlier_ratio = num_inliers / len(points)
        
        # 평면 높이
        mean_z = np.mean(points[inliers, 2])
        std_z = np.std(points[inliers, 2])
        
        print(f"    평면 방정식: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
        print(f"    법선 벡터: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
        print(f"    평균 높이: {mean_z:.3f}m (표준편차: {std_z:.4f}m)")
        print(f"    인라이어: {num_inliers:,}/{len(points):,} ({inlier_ratio*100:.1f}%)")
        
        return {
            'normal': normal,
            'd': d,
            'mean_z': mean_z,
            'std_z': std_z,
            'inliers': inliers,
            'inlier_ratio': inlier_ratio
        }
        
    except Exception as e:
        print(f"    피팅 실패: {e}")
        return None

# ====================================
# ★ NEW: Step 7 - 수평도 검증
# ====================================
def verify_horizontality(plane_params: Dict[str, Any]) -> None:
    """
    평면의 수평도 검증
    
    학습 포인트:
    - 법선 벡터와 Z축의 각도
    - 수평 평면의 조건
    - 검증 기준값 설정
    """
    
    if plane_params is None:
        return
    
    print("\n[수평도 검증]")
    
    normal = plane_params['normal']
    
    # Z축과의 각도 계산
    z_axis = np.array([0, 0, 1])
    cos_angle = np.abs(np.dot(normal, z_axis))
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    print(f"  법선과 Z축 각도: {angle_deg:.2f}°")
    print(f"  수평도 판정:")
    
    if angle_deg < 1.0:
        print(f"    ✅ 완벽한 수평 (< 1°)")
    elif angle_deg < 5.0:
        print(f"    ✅ 양호한 수평 (< 5°)")
    elif angle_deg < 10.0:
        print(f"    ⚠️ 약간 기울어짐 (< 10°)")
    else:
        print(f"    ❌ 심하게 기울어짐 (≥ 10°)")
    
    # 높이 일관성 검사
    std_z = plane_params['std_z']
    print(f"\n  높이 일관성:")
    print(f"    표준편차: {std_z*1000:.1f}mm")
    
    if std_z < 0.005:
        print(f"    ✅ 매우 평평함 (< 5mm)")
    elif std_z < 0.01:
        print(f"    ✅ 평평함 (< 10mm)")
    else:
        print(f"    ⚠️ 굴곡 있음 (≥ 10mm)")
    
    return angle_deg

# ====================================
# ★ NEW: Step 8 - 평면 시각화
# ====================================
def visualize_plane_detection(points: np.ndarray, plane_params: List[Dict[str, Any]]) -> None:
    """
    검출된 평면 시각화
    
    학습 포인트:
    - 인라이어/아웃라이어 색상 구분
    - 법선 벡터 화살표 표시
    - Open3D 기하 객체 생성
    """
    
    if plane_params is None:
        return
    
    print("\n[평면 시각화]")
    print("  녹색: 평면 인라이어")
    print("  빨간색: 아웃라이어")
    print("  노란색 화살표: 법선 벡터")
    
    # 포인트 클라우드 생성
    pcd_plane = o3d.geometry.PointCloud()
    pcd_plane.points = o3d.utility.Vector3dVector(points)
    
    # 색상 설정
    colors = np.zeros((len(points), 3))
    inliers = plane_params['inliers']
    colors[inliers, 1] = 1.0   # 인라이어: 녹색
    colors[~inliers, 0] = 1.0  # 아웃라이어: 빨간색
    pcd_plane.colors = o3d.utility.Vector3dVector(colors)
    
    # 법선 벡터 화살표
    center = np.mean(points[inliers], axis=0)
    arrow = create_normal_arrow(center, plane_params['normal'])
    
    # 좌표축
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    
    # 시각화
    o3d.visualization.draw_geometries(
        [pcd_plane, arrow, axes],
        window_name="Stage 3: RANSAC 평면 검출 결과",
        width=1024, height=768
    )

def create_normal_arrow(center: np.ndarray, normal: np.ndarray):
    """법선 벡터 화살표 생성"""
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.01,
        cone_radius=0.02,
        cylinder_height=0.2,
        cone_height=0.05
    )
    
    # 법선 방향으로 회전
    if np.abs(normal[2]) < 0.999:
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, normal)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_angle = np.arccos(np.clip(np.dot(z_axis, normal), -1.0, 1.0))
        
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_rotvec(rotation_angle * rotation_axis)
        arrow.rotate(rot.as_matrix(), center=[0, 0, 0])
    
    arrow.translate(center)
    arrow.paint_uniform_color([1, 1, 0])  # 노란색
    
    return arrow

# ====================================
# 메인 함수
# ====================================
def main() -> None:
    """Stage 3 main execution function"""
    
    print("=" * 60)
    print("Stage 3: RANSAC 평면 검출")
    print("=" * 60)
    print("\n학습 목표:")
    print("  1. RANSAC 알고리즘으로 노이즈에 강건한 평면 검출")
    print("  2. 법선 벡터로 평면 방향 분석")
    print("  3. 수평도와 평탄도 검증")
    
    try:
        # Step 1: 데이터 로드
        t = print_step(1, "데이터 로드 중")
        points_ros, pcd, stats = load_validation_data()
        print_done(t)
        
        # Step 2: Z 분포 확인
        t = print_step(2, "Z 분포 분석")
        analyze_z_distribution(points_ros)
        print_done(t)
        
        # ★ Step 3: RANSAC 평면 검출 (NEW)
        t = print_step(3, "RANSAC 평면 검출")
        plane_params, table_points = detect_planes_ransac(points_ros)
        print_done(t)
        
        # ★ Step 4: 수평도 검증 (NEW)
        if plane_params:
            t = print_step(4, "수평도 검증")
            verify_horizontality(plane_params)
            print_done(t)
            
            # ★ Step 5: 시각화 (NEW)
            t = print_step(5, "평면 검출 결과 시각화")
            visualize_plane_detection(table_points, plane_params)
            print_done(t)
        
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        return
    
    # 완료
    print("\n" + "=" * 60)
    print("✓ Stage 3 완료!")
    print("=" * 60)
    print("\n핵심 학습:")
    print("  • RANSAC: 노이즈와 아웃라이어에 강건한 피팅")
    print("  • 평면 방정식: ax + by + cz + d = 0")
    print("  • 법선 벡터: 평면에 수직인 방향")
    print("  • 수평 검증: 법선이 Z축과 평행하면 수평")
    print("\n다음 단계: Stage 4에서 대화형 검증 시스템을 완성합니다.")

if __name__ == "__main__":
    main()