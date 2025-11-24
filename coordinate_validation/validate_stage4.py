#!/usr/bin/env python3
"""
Stage 4: 대화형 검증 시스템 (전체 통합)
- Stage 3 기능 포함
- 마우스로 평면 선택
- 메뉴 시스템
- 검증 클래스 통합

새로운 학습 포인트:
- VisualizerWithEditing 사용법
- 대화형 인터페이스 설계
- 객체지향 검증 시스템
"""

import numpy as np
import open3d as o3d
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from scipy.spatial.transform import Rotation
import warnings
warnings.filterwarnings('ignore')

# ====================================
# ★ NEW: 통합 검증 클래스
# ====================================
class PointCloudValidator:
    """
    포인트 클라우드 좌표 변환 검증 클래스
    
    학습 포인트:
    - 객체지향 설계
    - 상태 관리
    - 메서드 조직화
    """
    
    def __init__(self):
        """초기화"""
        self.points_ros = None
        self.pcd = None
        self.stats = None
        self.selected_points = []
        
        # 데이터 로드
        self._load_data()
    
    def _load_data(self):
        """데이터 로드"""
        output_dir = Path(__file__).parent.parent / "lecture-coordinate-transform" / "output"

        points_ros_path = output_dir / "points_ros.npy"
        pointcloud_path = output_dir / "pointcloud_frame0053.ply"
        stats_path = output_dir / "transform_stats.json"
        
        if not all([points_ros_path.exists(), 
                   pointcloud_path.exists(), 
                   stats_path.exists()]):
            raise FileNotFoundError("필요한 데이터 파일이 없습니다.")
        
        self.points_ros = np.load(points_ros_path)
        self.pcd = o3d.io.read_point_cloud(str(pointcloud_path))
        
        with open(stats_path, 'r') as f:
            self.stats = json.load(f)
        
        print(f"✓ 데이터 로드 완료")
        print(f"  - 포인트 수: {len(self.points_ros):,}")
        print(f"  - 카메라 위치: {self.stats['camera_pose']['position']}")
        print()
    
    # ====================================
    # ★ NEW: 메뉴 시스템
    # ====================================
    def run_validation_menu(self):
        """
        대화형 검증 메뉴
        
        학습 포인트:
        - 사용자 인터페이스 설계
        - 메뉴 기반 네비게이션
        - 입력 검증
        """
        while True:
            print("=" * 50)
            print("좌표 변환 검증 도구")
            print("=" * 50)
            print("1. 시각적 검사 - 좌표계 축 표시")
            print("2. 평면 선택 검증 - 마우스로 평면 선택")
            print("3. RANSAC 평면 검출 - 자동 평면 찾기")
            print("4. 통계 분석 - 변환 통계 확인")
            print("5. 종료")
            print("-" * 50)
            
            choice = input("선택 (1-5): ").strip()
            
            if choice == '1':
                self.visual_inspection()
            elif choice == '2':
                self.interactive_plane_selection()
            elif choice == '3':
                self.ransac_plane_detection()
            elif choice == '4':
                self.show_statistics()
            elif choice == '5':
                print("프로그램을 종료합니다.")
                break
            else:
                print("잘못된 선택입니다. 다시 선택해주세요.")
            
            print()
    
    # ====================================
    # 기능 1: 시각적 검사
    # ====================================
    def visual_inspection(self):
        """좌표계 축을 포함한 시각적 검사"""
        print("\n[시각적 검사]")
        
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )
        
        o3d.visualization.draw_geometries(
            [self.pcd, coord_frame],
            window_name="검증: 시각적 검사",
            width=1024, height=768
        )
        
        print("✓ 시각적 검사 완료")
    
    # ====================================
    # ★ NEW: 기능 2 - 마우스 평면 선택
    # ====================================
    def interactive_plane_selection(self):
        """
        마우스로 평면 포인트 선택
        
        학습 포인트:
        - VisualizerWithEditing 사용
        - 사용자 상호작용
        - 선택된 포인트 처리
        """
        print("\n[대화형 평면 선택]")
        print("사용법:")
        print("  - Shift + 왼쪽 클릭: 포인트 선택")
        print("  - Shift + 오른쪽 클릭: 선택 취소")
        print("  - Q: 선택 완료 및 분석")
        
        # 포인트 피커 생성
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="평면 포인트 선택", 
                         width=1024, height=768)
        vis.add_geometry(self.pcd)
        
        # 사용자가 포인트 선택
        vis.run()
        vis.destroy_window()
        
        # 선택된 포인트 가져오기
        picked_indices = vis.get_picked_points()
        
        if len(picked_indices) < 3:
            print("⚠ 평면 검출을 위해 최소 3개 이상의 포인트가 필요합니다.")
            return
        
        print(f"\n선택된 포인트 수: {len(picked_indices)}")
        
        # 선택된 포인트로 평면 피팅
        selected_points = self.points_ros[picked_indices]
        self._fit_and_validate_plane(selected_points, "사용자 선택")
    
    # ====================================
    # 기능 3: RANSAC 평면 검출
    # ====================================
    def ransac_plane_detection(self):
        """RANSAC을 사용한 자동 평면 검출"""
        print("\n[RANSAC 평면 검출]")
        
        z_values = self.points_ros[:, 2]
        
        # 높이별 평면 검출
        height_ranges = [
            (0.35, 0.41, "테이블 상단"),
            (0.0, 0.05, "바닥"),
        ]
        
        for z_min, z_max, name in height_ranges:
            mask = (z_values >= z_min) & (z_values <= z_max)
            points_in_range = self.points_ros[mask]
            
            if len(points_in_range) < 100:
                continue
            
            print(f"\n{name} 검출 (Z: {z_min:.2f}~{z_max:.2f}m)")
            print(f"포인트 수: {len(points_in_range):,}")
            
            self._fit_and_validate_plane(points_in_range, name)
    
    # ====================================
    # 기능 4: 통계 분석
    # ====================================
    def show_statistics(self):
        """변환 통계 표시"""
        print("\n[변환 통계]")
        print("-" * 40)
        
        # 기본 통계
        print(f"총 포인트 수: {self.stats['num_points']:,}")
        
        # 좌표계별 중심점
        centroid_cam = self.stats['centroid_camera']
        centroid_ros = self.stats['centroid_ros']
        
        print(f"\n카메라 좌표계 중심점:")
        print(f"  X: {centroid_cam[0]:.4f}m")
        print(f"  Y: {centroid_cam[1]:.4f}m")
        print(f"  Z: {centroid_cam[2]:.4f}m (깊이)")
        
        print(f"\nROS 좌표계 중심점:")
        print(f"  X: {centroid_ros[0]:.4f}m (전방)")
        print(f"  Y: {centroid_ros[1]:.4f}m (좌측)")
        print(f"  Z: {centroid_ros[2]:.4f}m (상방)")
        
        # Z 분포
        z_values = self.points_ros[:, 2]
        print(f"\nZ 좌표 분포:")
        print(f"  최소: {np.min(z_values):.3f}m")
        print(f"  최대: {np.max(z_values):.3f}m")
        print(f"  평균: {np.mean(z_values):.3f}m")
        
        # 히스토그램 옵션
        if input("\nZ 분포 히스토그램을 보시겠습니까? (y/n): ").lower() == 'y':
            self._plot_z_histogram(z_values)
    
    # ====================================
    # 헬퍼 메서드들
    # ====================================
    def _fit_and_validate_plane(self, points, plane_name):
        """평면 피팅 및 검증"""
        if len(points) < 3:
            print("포인트가 부족합니다.")
            return
        
        # RANSAC 평면 피팅
        X = points[:, :2]
        y = points[:, 2]
        
        ransac = RANSACRegressor(
            random_state=42,
            min_samples=3,
            residual_threshold=0.01
        )
        
        try:
            ransac.fit(X, y)
            
            # 평면 파라미터
            a, b = ransac.estimator_.coef_
            c = -1.0
            d = ransac.estimator_.intercept_
            
            # 정규화
            norm = np.sqrt(a**2 + b**2 + c**2)
            normal = np.array([a, b, c]) / norm
            
            # 결과 계산
            inliers = ransac.inlier_mask_
            num_inliers = np.sum(inliers)
            inlier_ratio = num_inliers / len(points)
            mean_z = np.mean(points[inliers, 2])
            std_z = np.std(points[inliers, 2])
            
            # 수평도
            z_axis = np.array([0, 0, 1])
            angle = np.arccos(np.abs(np.dot(normal, z_axis))) * 180 / np.pi
            
            # 결과 출력
            print(f"\n[{plane_name} 평면 분석]")
            print(f"  평면 방정식: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
            print(f"  법선 벡터: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
            print(f"  평균 높이: {mean_z:.3f}m (표준편차: {std_z:.4f}m)")
            print(f"  수평도: {angle:.2f}° (0°가 완벽한 수평)")
            print(f"  인라이어: {num_inliers:,}/{len(points):,} ({inlier_ratio*100:.1f}%)")
            
            # 검증 결과
            if angle < 5.0 and std_z < 0.01:
                print(f"  ✓ 평면이 수평이며 일정한 높이를 유지합니다.")
            elif angle < 5.0:
                print(f"  ⚠ 평면은 수평이지만 높이 편차가 있습니다.")
            else:
                print(f"  ⚠ 평면이 기울어져 있습니다.")
            
            # 시각화 옵션
            if input("\n평면을 시각화하시겠습니까? (y/n): ").lower() == 'y':
                self._visualize_plane(points, inliers, normal, mean_z)
                
        except Exception as e:
            print(f"평면 피팅 실패: {e}")
    
    def _visualize_plane(self, points, inliers, normal, height):
        """평면 시각화"""
        # 포인트 클라우드
        pcd_plane = o3d.geometry.PointCloud()
        pcd_plane.points = o3d.utility.Vector3dVector(points)
        
        # 색상 설정
        colors = np.zeros((len(points), 3))
        colors[inliers, 1] = 1.0   # 인라이어: 녹색
        colors[~inliers, 0] = 1.0  # 아웃라이어: 빨간색
        pcd_plane.colors = o3d.utility.Vector3dVector(colors)
        
        # 평면 메시 생성 (시각화용)
        x_range = np.ptp(points[:, 0])
        y_range = np.ptp(points[:, 1])
        center = np.mean(points[inliers], axis=0)
        
        # 평면 상의 점들 생성
        xx, yy = np.meshgrid(
            np.linspace(center[0] - x_range/2, center[0] + x_range/2, 10),
            np.linspace(center[1] - y_range/2, center[1] + y_range/2, 10)
        )
        zz = np.ones_like(xx) * height
        
        # 메시 생성
        plane_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        plane_pcd = o3d.geometry.PointCloud()
        plane_pcd.points = o3d.utility.Vector3dVector(plane_points)
        plane_pcd.paint_uniform_color([0, 0, 1])  # 파란색
        
        # 법선 화살표
        arrow = self._create_arrow(center, normal)
        
        # 시각화
        o3d.visualization.draw_geometries(
            [pcd_plane, plane_pcd, arrow],
            window_name="평면 검증 결과",
            width=1024, height=768
        )
    
    def _create_arrow(self, center, normal):
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
            
            rot = Rotation.from_rotvec(rotation_angle * rotation_axis)
            arrow.rotate(rot.as_matrix(), center=[0, 0, 0])
        
        arrow.translate(center)
        arrow.paint_uniform_color([1, 1, 0])  # Yellow
        
        return arrow
    
    def _plot_z_histogram(self, z_values):
        """Z 분포 히스토그램"""
        plt.figure(figsize=(10, 6))
        plt.hist(z_values, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(0.38, color='r', linestyle='--', label='Table Height (0.38m)')
        plt.xlabel('Z Coordinate (m)')
        plt.ylabel('Number of Points')
        plt.title('ROS Coordinate System Z Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# ====================================
# 메인 함수
# ====================================
def main() -> None:
    """Stage 4 메인 실행 함수"""
    
    print("=" * 60)
    print("Stage 4: 대화형 검증 시스템 (전체 통합)")
    print("=" * 60)
    print("\n학습 목표:")
    print("  1. 대화형 인터페이스로 사용성 향상")
    print("  2. 마우스로 관심 영역 선택")
    print("  3. 통합된 검증 시스템 구축")
    
    try:
        # 검증기 생성 및 실행
        validator = PointCloudValidator()
        validator.run_validation_menu()
        
    except FileNotFoundError as e:
        print(f"\n❌ 오류: {e}")
        print("\n먼저 coordinate_transform_pipeline.py를 실행하세요.")
        return
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        return
    
    # 완료
    print("\n" + "=" * 60)
    print("✓ Stage 4 완료!")
    print("=" * 60)
    print("\n전체 검증 시스템 기능:")
    print("  • 시각적 검사: 좌표축과 포인트 클라우드")
    print("  • 대화형 선택: 마우스로 평면 선택")
    print("  • 자동 검출: RANSAC 평면 찾기")
    print("  • 통계 분석: 변환 결과 검증")
    print("\n🎉 좌표 변환 검증 도구 완성!")

if __name__ == "__main__":
    main()