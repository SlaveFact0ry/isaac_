#!/usr/bin/env python3
"""
Stage 1: 데이터 로드 및 기본 시각화
- RGB 이미지 로드
- Depth map 로드
- 카메라 파라미터 설정
- 2D 시각화

학습 포인트:
- Isaac Sim Replicator 데이터 구조
- Distance to image plane depth 이해
- 카메라 내부 파라미터
"""

import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib

# 환경에 따라 적절한 백엔드 선택
if 'DISPLAY' in os.environ:
    try:
        matplotlib.use('TkAgg')  # GUI 백엔드 시도
    except (ImportError, RuntimeError):
        matplotlib.use('Agg')  # GUI 실패 시 Agg 사용
else:
    matplotlib.use('Agg')  # 디스플레이 없으면 Agg 사용

import matplotlib.pyplot as plt

# 경로 설정
BASE_DATA_PATH = Path(__file__).parent.parent / "lecture-standalone" / "replicator_output" / "advanced_dataset" / "Replicator_04"

# 카메라 파라미터 상수 (Isaac Sim 고정값)
CAMERA_FX = 395.26  # 초점 거리 X
CAMERA_FY = 395.26  # 초점 거리 Y
CAMERA_CX = 256.0   # 주점 X
CAMERA_CY = 256.0   # 주점 Y
CAMERA_WIDTH = 512  # 이미지 너비
CAMERA_HEIGHT = 512 # 이미지 높이


def print_step(step_num: int, description: str) -> None:
    """단계 출력 헬퍼"""
    print(f"\n[Step {step_num}] {description}")


def load_rgb_image(frame_idx: int) -> np.ndarray:
    """
    RGB 이미지 로드
    
    Args:
        frame_idx: 프레임 인덱스 (0-419)
    
    Returns:
        RGB 이미지 (BGR 형식)
    """
    rgb_path = f"{BASE_DATA_PATH}/rgb/rgb_{frame_idx:04d}.png"
    
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"RGB 이미지를 찾을 수 없습니다: {rgb_path}")
    
    image = cv2.imread(rgb_path)
    print(f"  ✓ RGB 이미지 로드: {rgb_path}")
    print(f"    - 크기: {image.shape[1]}x{image.shape[0]}")
    print(f"    - 채널: {image.shape[2]}")
    
    return image


def load_depth_map(frame_idx: int) -> np.ndarray:
    """
    Depth map 로드 (Distance to image plane)
    
    Args:
        frame_idx: 프레임 인덱스
    
    Returns:
        Depth map (미터 단위)
    """
    depth_path = f"{BASE_DATA_PATH}/distance_to_image_plane/distance_to_image_plane_{frame_idx:04d}.npy"
    
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth 데이터를 찾을 수 없습니다: {depth_path}")
    
    depth_map = np.load(depth_path)
    
    print(f"  ✓ Depth map 로드: {depth_path}")
    print(f"    - 크기: {depth_map.shape}")
    print(f"    - 범위: {np.min(depth_map):.2f}m ~ {np.max(depth_map):.2f}m")
    print(f"    - 평균: {np.mean(depth_map[depth_map > 0]):.2f}m")
    
    return depth_map


def get_camera_parameters() -> dict:
    """
    카메라 내부 파라미터 설정
    
    Returns:
        카메라 파라미터 딕셔너리
    """
    # Isaac Sim 카메라 설정
    camera_params = {
        'fx': CAMERA_FX,
        'fy': CAMERA_FY,
        'cx': CAMERA_CX,
        'cy': CAMERA_CY,
        'width': CAMERA_WIDTH,
        'height': CAMERA_HEIGHT
    }
    
    print(f"  ✓ 카메라 파라미터:")
    print(f"    - 초점 거리: fx={camera_params['fx']:.1f}, fy={camera_params['fy']:.1f}")
    print(f"    - 주점: cx={camera_params['cx']:.1f}, cy={camera_params['cy']:.1f}")
    print(f"    - 이미지 크기: {camera_params['width']}x{camera_params['height']}")
    
    return camera_params


def visualize_2d_data(rgb_image: np.ndarray, depth_map: np.ndarray):
    """
    RGB와 Depth 2D 시각화
    
    Args:
        rgb_image: RGB 이미지
        depth_map: Depth map
    """
    plt.figure(figsize=(12, 5))
    
    # RGB 이미지 표시
    plt.subplot(1, 2, 1)
    rgb_display = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_display)
    plt.title('RGB Image')
    plt.axis('off')
    
    # Depth map 표시
    plt.subplot(1, 2, 2)
    # 유효한 depth 값만 사용하여 시각화
    valid_depth = depth_map.copy()
    valid_depth[depth_map <= 0] = np.nan
    
    plt.imshow(valid_depth, cmap='viridis')
    plt.colorbar(label='Distance (m)')
    plt.title('Depth Map (Distance to Image Plane)')
    plt.axis('off')
    
    plt.tight_layout()
    
    # 파일로 저장
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "stage1_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 시각화 완료: {output_path} 저장됨")
    
    # 화면에 표시 시도 (GUI 백엔드가 있는 경우만)
    if matplotlib.get_backend() != 'Agg':
        try:
            plt.show()
        except (RuntimeError, ImportError) as e:
            # 예외 발생 시 무시
            pass
    else:
        print("  ℹ GUI 백엔드가 없어 화면 표시를 건너뜁니다. 파일로 저장되었습니다.")


def main() -> None:
    """메인 실행 함수"""
    print("=" * 60)
    print("Stage 1: 데이터 로드 및 기본 시각화")
    print("=" * 60)
    print("\n학습 목표:")
    print("  1. Isaac Sim Replicator 데이터 구조 이해")
    print("  2. Distance to image plane depth 개념 학습")
    print("  3. 카메라 내부 파라미터 이해")
    
    # 프레임 인덱스 설정
    frame_idx = 83  # 예제 프레임
    print(f"\n처리할 프레임: {frame_idx:04d}")
    
    # Step 1: RGB 이미지 로드
    print_step(1, "RGB 이미지 로드")
    rgb_image = load_rgb_image(frame_idx)
    
    # Step 2: Depth map 로드
    print_step(2, "Depth map 로드")
    depth_map = load_depth_map(frame_idx)
    
    # Step 3: 카메라 파라미터 설정
    print_step(3, "카메라 파라미터 설정")
    camera_params = get_camera_parameters()
    
    # Step 4: 2D 시각화
    print_step(4, "2D 데이터 시각화")
    visualize_2d_data(rgb_image, depth_map)
    
    # 완료
    print("\n" + "=" * 60)
    print("✓ Stage 1 완료!")
    print("=" * 60)
    print("\n핵심 포인트:")
    print("  • RGB와 Depth는 픽셀 단위로 정렬되어 있음")
    print("  • Distance to image plane은 카메라 평면까지의 직선 거리")
    print("  • 카메라 파라미터는 3D 변환에 필수")
    print("\n다음 단계: Stage 2에서 YOLO 세그멘테이션 추가")


if __name__ == "__main__":
    main()