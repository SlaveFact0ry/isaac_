#!/usr/bin/env python3
"""
Stage 3: 3D 변환 추가
- Stage 1, 2 기능 포함
- Depth와 마스크 통합
- 3D 좌표 변환
- 좌표계 변환 (Camera → Isaac → ROS)

새로운 학습 포인트:
- Pinhole 카메라 모델
- 좌표계 변환 행렬
- Point cloud 생성
"""

import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib
from typing import Dict, List, Tuple, Optional, Any

# 환경에 따라 적절한 백엔드 선택
if 'DISPLAY' in os.environ:
    try:
        matplotlib.use('TkAgg')  # GUI 백엔드 시도
    except (ImportError, RuntimeError):
        matplotlib.use('Agg')  # GUI 실패 시 Agg 사용
else:
    matplotlib.use('Agg')  # 디스플레이 없으면 Agg 사용

import matplotlib.pyplot as plt
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# 경로 설정
BASE_DATA_PATH = Path(__file__).parent.parent / "lecture-standalone" / "replicator_output" / "advanced_dataset" / "Replicator_04"
MODEL_PATH = "models/best.pt"

# 카메라 파라미터 상수 (Isaac Sim 고정값)
CAMERA_FX = 395.26  # 초점 거리 X
CAMERA_FY = 395.26  # 초점 거리 Y
CAMERA_CX = 256.0   # 주점 X
CAMERA_CY = 256.0   # 주점 Y
CAMERA_WIDTH = 512  # 이미지 너비
CAMERA_HEIGHT = 512 # 이미지 높이

# Isaac Sim 객체 클래스
CLASS_NAMES = ['banana', 'meat_can', 'soup_can', 'marker']
CLASS_COLORS = [
    (0, 255, 255),   # banana - Yellow
    (0, 128, 255),   # meat_can - Orange
    (0, 255, 0),     # soup_can - Green
    (255, 0, 255)    # marker - Magenta
]

def print_step(step_num: int, description: str) -> None:
    """단계 출력 헬퍼"""
    print(f"\n[Step {step_num}] {description}")


# ====== Stage 1 함수들 ======
def load_rgb_image(frame_idx: int) -> np.ndarray:
    """RGB 이미지 로드"""
    rgb_path = f"{BASE_DATA_PATH}/rgb/rgb_{frame_idx:04d}.png"
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"RGB 이미지를 찾을 수 없습니다: {rgb_path}")
    
    image = cv2.imread(rgb_path)
    print(f"  ✓ RGB 이미지 로드: {image.shape[1]}x{image.shape[0]}")
    return image


def load_depth_map(frame_idx: int) -> np.ndarray:
    """Depth map 로드"""
    depth_path = f"{BASE_DATA_PATH}/distance_to_image_plane/distance_to_image_plane_{frame_idx:04d}.npy"
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth 데이터를 찾을 수 없습니다: {depth_path}")
    
    depth_map = np.load(depth_path)
    print(f"  ✓ Depth map 로드: 범위 {np.min(depth_map):.2f}m ~ {np.max(depth_map):.2f}m")
    return depth_map


def get_camera_parameters() -> dict:
    """카메라 내부 파라미터 설정"""
    camera_params = {
        'fx': CAMERA_FX,
        'fy': CAMERA_FY,
        'cx': CAMERA_CX,
        'cy': CAMERA_CY,
        'width': CAMERA_WIDTH,
        'height': CAMERA_HEIGHT
    }
    
    print(f"  ✓ 카메라 파라미터 설정")
    return camera_params


# ====== Stage 2 함수들 ======
def load_yolo_model() -> YOLO:
    """학습된 YOLO 세그멘테이션 모델 로드"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
    
    model = YOLO(MODEL_PATH)
    print(f"  ✓ YOLO 모델 로드: {MODEL_PATH}")
    return model


def perform_segmentation(image: np.ndarray, model: YOLO, conf_threshold: float = 0.5) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """YOLO 세그멘테이션 수행"""
    results = model(image, conf=conf_threshold, verbose=False)
    
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        
        print(f"  ✓ 검출된 객체: {len(masks)}개")
        for i, (cls, conf) in enumerate(zip(classes, confidences)):
            class_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"class_{cls}"
            print(f"    - Object {i+1}: {class_name} (신뢰도: {conf:.2%})")
        
        return masks, boxes, confidences, classes
    else:
        print("  ✗ 검출된 객체 없음")
        return None, None, None, None


def refine_mask(mask: np.ndarray) -> np.ndarray:
    """Morphological operations으로 마스크 정제"""
    mask_binary = (mask > 0.5).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_closed = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
    mask_refined = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
    return mask_refined


# ====== Stage 3 함수들 ======
def get_camera_transform() -> Dict[str, Any]:
    """
    카메라 변환 정보 생성 (coordinate-transform과 동일한 방식)
    Camera → Isaac Sim World → ROS (3단계 변환)
    
    Returns:
        카메라 포즈 정보 (position, rotation matrix)
    """
    # 이전 coordinate-transform과 동일한 카메라 설정

    # 카메라 위치
    camera_position = np.array([0.5, 1.5, 0.0])  # World(Isaac) 좌표계 기준

    # 카메라 회전 정의
    cam_rotation_x = np.array([
        [1, 0, 0],
        [0, 0, -1],  # Y와 Z 교환 (카메라가 아래를 봄)
        [0, 1, 0]
    ])
    
    cam_rotation_z = np.array([
        [0, -1, 0],  # X와 Y 회전
        [1, 0, 0],
        [0, 0, 1]
    ])
    
    # 전체 회전 = Z 회전 후 X 회전
    cam_rotation = cam_rotation_x @ cam_rotation_z
    
    print(f"  ✓ 카메라 변환 정보 생성")
    print(f"    - 카메라 위치 (World): {camera_position}")
    
    return {
        'position': camera_position,
        'rotation': cam_rotation
    }


def integrate_depth_and_mask(mask: np.ndarray, depth_map: np.ndarray, 
                            rgb_image: np.ndarray, camera_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Depth와 마스크를 통합하여 3D 포인트 생성
    
    Args:
        mask: 세그멘테이션 마스크
        depth_map: Depth map
        rgb_image: RGB 이미지
        camera_params: 카메라 파라미터
    
    Returns:
        3D 포인트와 색상 정보
    """
    h, w = depth_map.shape
    
    # 마스크 리사이즈 및 정제
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_refined = refine_mask(mask_resized)
    
    # 유효한 픽셀 찾기
    valid_mask = (mask_refined > 0) & (depth_map > 0)
    
    if np.sum(valid_mask) == 0:
        return None
    
    # 픽셀 좌표 생성
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
    
    # 유효한 픽셀의 depth 값
    valid_depths = depth_map[valid_mask]
    valid_u = u_coords[valid_mask]
    valid_v = v_coords[valid_mask]
    
    # Pinhole 카메라 모델로 3D 변환
    # x = (u - cx) × d / fx
    # y = (v - cy) × d / fy
    # z = d
    x_cam = (valid_u - camera_params['cx']) * valid_depths / camera_params['fx']
    y_cam = (valid_v - camera_params['cy']) * valid_depths / camera_params['fy']
    z_cam = valid_depths
    
    # 카메라 좌표계 포인트
    points_camera = np.stack([x_cam, y_cam, z_cam], axis=-1)
    
    # RGB 색상 추출
    rgb_colors = rgb_image[v_coords[valid_mask], u_coords[valid_mask]]
    
    return {
        'points_camera': points_camera,
        'colors': rgb_colors,
        'num_points': len(points_camera)
    }


def transform_camera_to_world(points_camera: np.ndarray, camera_pose: Dict[str, Any]) -> np.ndarray:
    """
    카메라 좌표계 → World(Isaac) 좌표계 변환
    
    Args:
        points_camera: 카메라 좌표계 포인트
        camera_pose: 카메라 포즈 정보 (position, rotation)
    
    Returns:
        World 좌표계 포인트
    """
    # 4x4 변환 행렬 생성
    T = np.eye(4)
    T[:3, :3] = camera_pose['rotation']
    T[:3, 3] = camera_pose['position']
    
    # Homogeneous 좌표로 변환
    points_homo = np.ones((points_camera.shape[0], 4))
    points_homo[:, :3] = points_camera
    
    # 변환 적용: P_world = T @ P_camera
    points_world = (T @ points_homo.T).T[:, :3]
    
    return points_world


def transform_world_to_ros(points_world: np.ndarray) -> np.ndarray:
    """
    World(Isaac Y-up) → ROS(Z-up) 좌표계 변환
    coordinate-transform과 동일한 변환 규칙 적용
    
    Args:
        points_world: World 좌표계 포인트
    
    Returns:
        ROS 좌표계 포인트
    """
    points_ros = np.zeros_like(points_world)
    
    # Isaac(Y-up) to ROS(Z-up) 변환
    points_ros[:, 0] = points_world[:, 0]   # X → X
    points_ros[:, 1] = -points_world[:, 2]  # Z → -Y
    points_ros[:, 2] = points_world[:, 1]   # Y → Z
    
    return points_ros


def visualize_3d_results(objects_3d: List[Dict[str, Any]], classes: np.ndarray) -> None:
    """
    3D 변환 결과 시각화
    
    Args:
        objects_3d: 3D 객체 정보 리스트
        classes: 클래스 인덱스
    """
    fig = plt.figure(figsize=(15, 5))
    
    # 3D 산점도
    ax1 = fig.add_subplot(131, projection='3d')
    
    for obj, cls in zip(objects_3d, classes):
        if obj is not None:
            points = obj['points_ros']
            colors = obj['colors']
            
            # 포인트 샘플링 (시각화용)
            sample_idx = np.random.choice(len(points), 
                                        min(500, len(points)), 
                                        replace=False)
            
            # 원본 RGB와 클래스 색상 블렌딩
            colors_sampled = colors[sample_idx].astype(np.float64) / 255.0
            colors_rgb = colors_sampled[:, [2, 1, 0]]  # BGR → RGB
            class_color = np.array(CLASS_COLORS[cls][::-1]) / 255.0
            blended_colors = colors_rgb * 0.7 + class_color * 0.3
            
            ax1.scatter(points[sample_idx, 0], 
                       points[sample_idx, 1], 
                       points[sample_idx, 2],
                       c=blended_colors, s=1, alpha=0.8,
                       label=CLASS_NAMES[cls])
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Point Clouds (ROS Coordinates)')
    ax1.legend()
    
    # XY 평면 투영
    ax2 = fig.add_subplot(132)
    for obj, cls in zip(objects_3d, classes):
        if obj is not None:
            points = obj['points_ros']
            color = np.array(CLASS_COLORS[cls]) / 255.0
            ax2.scatter(points[:, 0], points[:, 1], 
                       c=[color], s=1, alpha=0.5,
                       label=CLASS_NAMES[cls])
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Plane (Top View)')
    ax2.axis('equal')
    ax2.grid(True)
    ax2.legend()
    
    # Z 분포
    ax3 = fig.add_subplot(133)
    for obj, cls in zip(objects_3d, classes):
        if obj is not None:
            points = obj['points_ros']
            ax3.hist(points[:, 2], bins=30, alpha=0.5, 
                    label=CLASS_NAMES[cls],
                    color=np.array(CLASS_COLORS[cls])/255.0)
    
    ax3.set_xlabel('Z (m)')
    ax3.set_ylabel('Count')
    ax3.set_title('Z Distribution')
    ax3.legend()
    
    plt.tight_layout()
    
    # 파일로 저장
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "stage3_3d_transform.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 3D 시각화 완료: {output_path} 저장됨")
    
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
    print("Stage 3: 3D 변환 추가")
    print("=" * 60)
    print("\n학습 목표:")
    print("  1. Pinhole 카메라 모델 이해")
    print("  2. 좌표계 변환 (Camera → Isaac → ROS)")
    print("  3. Point cloud 생성")
    
    # 프레임 인덱스 설정
    frame_idx = 83
    print(f"\n처리할 프레임: {frame_idx:04d}")
    
    # Step 1: 데이터 로드 (Stage 1)
    print_step(1, "데이터 로드")
    rgb_image = load_rgb_image(frame_idx)
    depth_map = load_depth_map(frame_idx)
    camera_params = get_camera_parameters()
    
    # Step 2: YOLO 세그멘테이션 (Stage 2)
    print_step(2, "YOLO 세그멘테이션")
    model = load_yolo_model()
    masks, boxes, confidences, classes = perform_segmentation(rgb_image, model)
    
    # Step 3: 카메라 변환 설정
    print_step(3, "카메라 변환 설정")
    camera_pose = get_camera_transform()
    
    # Step 4: 3D 변환 (3단계 프로세스)
    print_step(4, "3D 변환 수행 (Camera → World → ROS)")
    objects_3d = []
    
    if masks is not None:
        for i, (mask, cls) in enumerate(zip(masks, classes)):
            # Depth와 마스크 통합
            result = integrate_depth_and_mask(mask, depth_map, 
                                             rgb_image, camera_params)
            
            if result is not None:
                # Step 1: Camera → World 변환
                points_world = transform_camera_to_world(result['points_camera'], 
                                                        camera_pose)
                
                # Step 2: World → ROS 변환
                points_ros = transform_world_to_ros(points_world)
                
                obj_3d = {
                    'points_ros': points_ros,
                    'colors': result['colors'],
                    'num_points': result['num_points'],
                    'class': CLASS_NAMES[cls]
                }
                objects_3d.append(obj_3d)
                
                print(f"    - {CLASS_NAMES[cls]}: {result['num_points']} points")
                print(f"      Position: ({np.mean(points_ros[:, 0]):.2f}, "
                      f"{np.mean(points_ros[:, 1]):.2f}, "
                      f"{np.mean(points_ros[:, 2]):.2f}) m")
            else:
                objects_3d.append(None)
    
    # NEW Step 5: 3D 시각화
    print_step(5, "3D 결과 시각화")
    if masks is not None:
        visualize_3d_results(objects_3d, classes)
    
    # 완료
    print("\n" + "=" * 60)
    print("✓ Stage 3 완료!")
    print("=" * 60)
    print("\n핵심 포인트:")
    print("  • Pinhole 모델: (u,v,d) → (x,y,z)")
    print("  • 좌표계 변환으로 ROS 표준 준수")
    print("  • 각 객체는 독립적인 3D point cloud")
    print("\n다음 단계: Stage 4에서 Open3D 시각화 및 저장")


if __name__ == "__main__":
    main()