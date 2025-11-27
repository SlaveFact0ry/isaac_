#!/usr/bin/env python3
"""
Stage 1: 데이터 로드와 2D 시각화
- RGB-D 데이터 로드
- 2D 시각화 (RGB + Depth)
- 데이터 통계 출력

학습 포인트:
- RGB-D 데이터 구조 이해
- 깊이값의 의미 (distance_to_image_plane)
- 데이터 범위와 유효성 확인
"""

import os
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
# Step 1: 데이터 로드
# ====================================
def load_data(frame_id: int = 53) -> Tuple[np.ndarray, np.ndarray]:
    """
    RGB-D 데이터 로드
    
    학습 포인트:
    - RGB: 512x512x3 컬러 이미지
    - Depth: 512x512 깊이 맵 (미터 단위)
    - distance_to_image_plane: Z-depth (수직 거리)
    """
    
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
# Step 2: 2D 시각화
# ====================================
def visualize_data_2d(rgb_image: np.ndarray, depth_map: np.ndarray) -> None:
    """
    2D 데이터 시각화
    
    학습 포인트:
    - RGB 이미지: 일반적인 카메라 영상
    - Depth 맵: 거리 정보를 색상으로 표현
    - 가까운 곳: 파란색, 먼 곳: 노란색 (VIRIDIS 컬러맵)
    """
    
    print("\n[시각화]")
    print("  RGB 이미지와 Depth 맵을 나란히 표시합니다.")
    print("  Depth 맵 색상: 파란색(가까움) → 노란색(멀음)")
    
    # Depth 맵 정규화 및 컬러맵 적용
    depth_norm = ((depth_map - depth_map.min()) / 
                  (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_VIRIDIS)
    
    # RGB를 BGR로 변환 (OpenCV 표시용)
    vis_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
    # 두 이미지 합치기
    combined = np.hstack([vis_bgr, depth_colored])
    
    # 제목과 정보 추가
    title_height = 40
    info_height = 60
    titled = np.zeros((combined.shape[0] + title_height + info_height, 
                       combined.shape[1], 3), dtype=np.uint8)
    titled[title_height:title_height + combined.shape[0]] = combined
    titled[:title_height] = (50, 50, 50)
    titled[-info_height:] = (50, 50, 50)
    
    # 제목 추가
    cv2.putText(titled, 'RGB Image', (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(titled, 'Depth Map (distance_to_image_plane)', 
               (rgb_image.shape[1] + 10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 통계 정보 추가
    stats_y = combined.shape[0] + title_height + 20
    cv2.putText(titled, f"Depth range: {depth_map.min():.3f} - {depth_map.max():.3f}m", 
               (10, stats_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(titled, f"Valid pixels: {np.sum(depth_map > 0):,} / {depth_map.size:,}", 
               (10, stats_y + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # 윈도우 표시
    try:
        cv2.imshow('Stage 1: RGB-D Data Visualization', titled)
        print("\n  → 아무 키나 누르면 계속됩니다...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except (RuntimeError, ImportError) as e:
        # 헤드리스 환경인 경우 파일로 저장
        save_path = Path("output") / "stage1_visualization.png"
        save_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(save_path), titled)
        print(f"\n  → 디스플레이를 사용할 수 없어 이미지를 저장했습니다: {save_path}")

# ====================================
# Step 3: 데이터 분석
# ====================================
def analyze_depth_data(depth_map: np.ndarray) -> None:
    """
    깊이 데이터 상세 분석
    
    학습 포인트:
    - 유효한 깊이값 vs 무효한 깊이값 (0)
    - 깊이 분포 통계
    - 이미지 영역별 깊이 특성
    """
    
    print("\n[데이터 분석]")
    
    # 전체 통계
    valid_mask = depth_map > 0
    valid_depths = depth_map[valid_mask]
    
    print(f"  전체 픽셀: {depth_map.size:,}")
    print(f"  유효 픽셀: {np.sum(valid_mask):,} ({np.sum(valid_mask)/depth_map.size*100:.1f}%)")
    print(f"  무효 픽셀: {np.sum(~valid_mask):,} ({np.sum(~valid_mask)/depth_map.size*100:.1f}%)")
    
    if len(valid_depths) > 0:
        print(f"\n  깊이 통계 (유효한 픽셀만):")
        print(f"    최소값: {valid_depths.min():.3f}m")
        print(f"    최대값: {valid_depths.max():.3f}m")
        print(f"    평균값: {valid_depths.mean():.3f}m")
        print(f"    중앙값: {np.median(valid_depths):.3f}m")
        print(f"    표준편차: {valid_depths.std():.3f}m")
    
    # 영역별 분석 (4분할)
    h, w = depth_map.shape
    regions = {
        "좌상": depth_map[:h//2, :w//2],
        "우상": depth_map[:h//2, w//2:],
        "좌하": depth_map[h//2:, :w//2],
        "우하": depth_map[h//2:, w//2:]
    }
    
    print(f"\n  영역별 평균 깊이:")
    for name, region in regions.items():
        valid_region = region[region > 0]
        if len(valid_region) > 0:
            print(f"    {name}: {valid_region.mean():.3f}m")

# ====================================
# 메인 함수
# ====================================
def main() -> None:
    """Stage 1 메인 실행 함수"""
    
    print("=" * 60)
    print("Stage 1: 데이터 로드와 2D 시각화")
    print("=" * 60)
    print("\n학습 목표:")
    print("  1. RGB-D 데이터 구조 이해")
    print("  2. 깊이값(distance_to_image_plane)의 의미 파악")
    print("  3. 데이터 시각화와 분석")
    
    # Step 1: 데이터 로드
    t = print_step(1, "RGB-D 데이터 로드 중")
    frame_id = 53
    rgb_image, depth_map = load_data(frame_id)
    print_done(t)
    
    # Step 2: 2D 시각화
    t = print_step(2, "2D 데이터 시각화")
    visualize_data_2d(rgb_image, depth_map)
    print_done(t)
    
    # Step 3: 데이터 분석
    t = print_step(3, "깊이 데이터 분석")
    analyze_depth_data(depth_map)
    print_done(t)
    
    # 완료
    print(f"\n{'=' * 60}")
    print("✓ Stage 1 완료!")
    print("=" * 60)
    print("\n핵심 포인트:")
    print("  • RGB: 512×512×3 컬러 이미지")
    print("  • Depth: 512×512 깊이 맵 (미터 단위)")
    print("  • distance_to_image_plane: 카메라에서 이미지 평면까지의 수직 거리")
    print("  • 깊이값 0: 측정 불가능한 영역 (무한 거리 등)")
    print("\n다음 단계: Stage 2에서 이 2D 데이터를 3D로 변환합니다.")

if __name__ == "__main__":
    main()