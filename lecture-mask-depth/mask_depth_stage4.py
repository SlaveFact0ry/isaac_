#!/usr/bin/env python3
"""
Stage 4: 통합 및 저장
- Stage 1, 2, 3 기능 포함
- Open3D 시각화
- PLY 파일 저장
- JSON 메타데이터 저장

새로운 학습 포인트:
- Open3D point cloud 생성
- PLY 파일 포맷
- 메타데이터 관리
"""

import os
from pathlib import Path
import json
import numpy as np
import cv2
import matplotlib
from typing import Dict, List, Tuple, Optional, Any
matplotlib.use('Agg')  # 디스플레이 없이 실행
import open3d as o3d
from datetime import datetime
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# 경로 설정
BASE_DATA_PATH = Path(__file__).parent.parent / "lecture-standalone" / "replicator_output" / "advanced_dataset" / "Replicator_04"
MODEL_PATH = "models/best.pt"
OUTPUT_DIR = Path(__file__).parent / "output"

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
    return camera_params


# ====== Stage 2 함수들 ======
def load_yolo_model() -> YOLO:
    """학습된 YOLO 세그멘테이션 모델 로드"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
    
    model = YOLO(MODEL_PATH)
    print(f"  ✓ YOLO 모델 로드")
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
            print(f"    - {class_name}: 신뢰도 {conf:.2%}")
        
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
def get_camera_pose() -> dict:
    """카메라 포즈 정보 반환 (3단계 변환용)"""
    # coordinate-transform과 동일한 설정
    camera_position = np.array([0.5, 1.5, 0.0])  # coordinate-transform과 동일
    
    # 카메라 회전 행렬 (X축 90도, Z축 90도 회전)
    R_x = np.array([
        [1, 0, 0],
        [0, 0, -1],  # cos(90) = 0, -sin(90) = -1
        [0, 1, 0]    # sin(90) = 1, cos(90) = 0
    ])
    
    R_z = np.array([
        [0, -1, 0],  # cos(90) = 0, -sin(90) = -1
        [1, 0, 0],   # sin(90) = 1, cos(90) = 0
        [0, 0, 1]
    ])
    
    R_camera = R_x @ R_z
    
    return {
        'rotation': R_camera,
        'position': camera_position
    }


def integrate_depth_and_mask(mask: np.ndarray, depth_map: np.ndarray, 
                            rgb_image: np.ndarray, camera_params: dict) -> dict:
    """Depth와 마스크를 통합하여 3D 포인트 생성"""
    h, w = depth_map.shape
    
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_refined = refine_mask(mask_resized)
    
    valid_mask = (mask_refined > 0) & (depth_map > 0)
    
    if np.sum(valid_mask) == 0:
        return None
    
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
    
    valid_depths = depth_map[valid_mask]
    valid_u = u_coords[valid_mask]
    valid_v = v_coords[valid_mask]
    
    x_cam = (valid_u - camera_params['cx']) * valid_depths / camera_params['fx']
    y_cam = (valid_v - camera_params['cy']) * valid_depths / camera_params['fy']
    z_cam = valid_depths
    
    points_camera = np.stack([x_cam, y_cam, z_cam], axis=-1)
    rgb_colors = rgb_image[v_coords[valid_mask], u_coords[valid_mask]]
    
    return {
        'points_camera': points_camera,
        'colors': rgb_colors,
        'num_points': len(points_camera)
    }


def transform_camera_to_world(points_camera: np.ndarray, camera_pose: dict) -> np.ndarray:
    """카메라 좌표계 → 월드(Isaac) 좌표계 변환"""
    # 4x4 변환 행렬 생성
    T = np.eye(4)
    T[:3, :3] = camera_pose['rotation']
    T[:3, 3] = camera_pose['position']
    
    # 동차 좌표 변환
    points_homo = np.ones((points_camera.shape[0], 4))
    points_homo[:, :3] = points_camera
    
    # 변환 적용: P_world = T @ P_camera
    points_world = (T @ points_homo.T).T[:, :3]
    
    return points_world


def transform_world_to_ros(points_world: np.ndarray) -> np.ndarray:
    """월드(Isaac Y-up) → ROS(Z-up) 좌표계 변환"""
    points_ros = np.zeros_like(points_world)
    
    # Isaac(Y-up) → ROS(Z-up) 변환
    points_ros[:, 0] = points_world[:, 0]   # X → X (전방 유지)
    points_ros[:, 1] = -points_world[:, 2]  # Z → -Y (오른쪽 → 왼쪽)
    points_ros[:, 2] = points_world[:, 1]   # Y → Z (위 축 변경)
    
    return points_ros


# ====== Stage 4 함수들 ======
def create_output_directory() -> Path:
    """출력 디렉토리 생성"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"  ✓ 출력 디렉토리 생성: {OUTPUT_DIR}")
    else:
        print(f"  ✓ 출력 디렉토리 확인: {OUTPUT_DIR}")


def save_point_cloud_ply(points: np.ndarray, colors: np.ndarray, 
                        filename: str) -> bool:
    """
    Point cloud를 PLY 파일로 저장
    
    Args:
        points: 3D 포인트
        colors: RGB 색상
        filename: 저장할 파일명
    
    Returns:
        저장 성공 여부
    """
    try:
        # Open3D point cloud 생성
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 색상 정규화 (0-255 → 0-1)
        colors_norm = colors.astype(np.float64) / 255.0
        # BGR → RGB 변환
        colors_rgb = colors_norm[:, [2, 1, 0]]
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
        
        # PLY 파일 저장
        filepath = os.path.join(OUTPUT_DIR, filename)
        o3d.io.write_point_cloud(filepath, pcd)
        
        print(f"    ✓ {filename}: {len(points)} points")
        return True
        
    except Exception as e:
        print(f"    ✗ {filename} 저장 실패: {e}")
        return False


def save_metadata_json(frame_idx: int, objects_info: List[Dict[str, Any]]) -> None:
    """
    메타데이터를 JSON으로 저장
    
    Args:
        frame_idx: 프레임 인덱스
        objects_info: 객체 정보 리스트
    """
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'frame_index': frame_idx,
        'data_path': str(BASE_DATA_PATH),
        'model_path': str(MODEL_PATH),
        'objects': objects_info
    }
    
    json_path = OUTPUT_DIR / f'frame_{frame_idx:04d}_metadata.json'
    
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ 메타데이터 저장: {json_path}")


def visualize_with_open3d(objects_3d: List[Dict[str, Any]], classes: np.ndarray) -> None:
    """
    Open3D로 통합 시각화
    
    Args:
        objects_3d: 3D 객체 정보 리스트
        classes: 클래스 인덱스
    """
    # 시각화 요소 리스트
    geometries = []
    
    # 각 객체의 point cloud 추가
    for obj, cls in zip(objects_3d, classes):
        if obj is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obj['points_ros'])
            
            # 원본 RGB 색상과 클래스 색상 블렌딩
            colors_norm = obj['colors'].astype(np.float64) / 255.0
            colors_rgb = colors_norm[:, [2, 1, 0]]  # BGR → RGB
            
            # 클래스 색상 (BGR → RGB 변환 후 정규화)
            class_color = np.array(CLASS_COLORS[cls][::-1]) / 255.0
            
            # 블렌딩 (70% 원본 색상 + 30% 클래스 색상)
            blended_colors = colors_rgb * 0.7 + class_color * 0.3
            
            pcd.colors = o3d.utility.Vector3dVector(blended_colors)
            
            geometries.append(pcd)
    
    # 좌표축 추가
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.3, origin=[0, 0, 0]
    )
    geometries.append(coord_frame)
    
    # 테이블 평면 추가 (시각화용)
    table_box = o3d.geometry.TriangleMesh.create_box(
        width=2.0,   # X
        height=2.0,  # Y
        depth=0.01   # Z
    )
    table_box.translate([-1.0, -1.0, 0.375])
    table_box.paint_uniform_color([0.7, 0.7, 0.7])
    geometries.append(table_box)
    
    # 시각화
    print("\n  Open3D 시각화 창이 열립니다:")
    print("    - 마우스 드래그: 회전")
    print("    - Ctrl + 드래그: 이동")
    print("    - 스크롤: 줌")
    print("    - Q: 종료")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Stage 4: Integrated Point Clouds",
        width=1024,
        height=768
    )


def process_frame_complete(frame_idx: int) -> None:
    """
    프레임 처리 완전 통합 파이프라인
    
    Args:
        frame_idx: 처리할 프레임 인덱스
    """
    # Stage 1: 데이터 로드
    print_step(1, "데이터 로드")
    rgb_image = load_rgb_image(frame_idx)
    depth_map = load_depth_map(frame_idx)
    camera_params = get_camera_parameters()
    
    # Stage 2: YOLO 세그멘테이션
    print_step(2, "YOLO 세그멘테이션")
    model = load_yolo_model()
    masks, boxes, confidences, classes = perform_segmentation(rgb_image, model)
    
    if masks is None:
        print("\n객체가 검출되지 않았습니다.")
        return
    
    # Stage 3: 3D 변환 (3단계 프로세스)
    print_step(3, "3D 변환 (Camera → World → ROS)")
    camera_pose = get_camera_pose()
    objects_3d = []
    objects_info = []
    
    for i, (mask, cls, conf) in enumerate(zip(masks, classes, confidences)):
        # Depth와 마스크 통합
        result = integrate_depth_and_mask(mask, depth_map, 
                                         rgb_image, camera_params)
        
        if result is not None:
            # 3단계 변환: Camera → World → ROS
            points_world = transform_camera_to_world(result['points_camera'], 
                                                     camera_pose)
            points_ros = transform_world_to_ros(points_world)
            
            # 객체 정보 저장
            obj_3d = {
                'points_ros': points_ros,
                'colors': result['colors'],
                'num_points': result['num_points'],
                'class': CLASS_NAMES[cls]
            }
            objects_3d.append(obj_3d)
            
            # 메타데이터용 정보
            objects_info.append({
                'class': CLASS_NAMES[cls],
                'confidence': float(conf),
                'num_points': result['num_points'],
                'centroid': points_ros.mean(axis=0).tolist(),
                'min_bound': points_ros.min(axis=0).tolist(),
                'max_bound': points_ros.max(axis=0).tolist()
            })
        else:
            objects_3d.append(None)
    
    # Stage 4: 저장 및 시각화
    print_step(4, "데이터 저장")
    create_output_directory()
    
    # PLY 파일 저장
    for i, (obj, cls) in enumerate(zip(objects_3d, classes)):
        if obj is not None:
            filename = f"frame_{frame_idx:04d}_object_{i}_{CLASS_NAMES[cls]}.ply"
            save_point_cloud_ply(obj['points_ros'], obj['colors'], filename)
    
    # 메타데이터 저장
    save_metadata_json(frame_idx, objects_info)
    
    # Open3D 시각화
    print_step(5, "Open3D 시각화")
    visualize_with_open3d(objects_3d, classes)


def main() -> None:
    """메인 실행 함수"""
    print("=" * 60)
    print("Stage 4: 통합 및 저장")
    print("=" * 60)
    print("\n학습 목표:")
    print("  1. Open3D point cloud 관리")
    print("  2. PLY 파일 포맷으로 저장")
    print("  3. 메타데이터 JSON 생성")
    print("  4. 통합 시각화")
    
    # 프레임 처리
    frame_idx = 83
    print(f"\n처리할 프레임: {frame_idx:04d}")
    
    # 전체 파이프라인 실행
    process_frame_complete(frame_idx)
    
    # 완료
    print("\n" + "=" * 60)
    print("✓ Stage 4 완료!")
    print("=" * 60)
    print("\n최종 성과:")
    print("  • 완전한 segmentation + depth 파이프라인")
    print("  • PLY 파일로 3D 데이터 영구 저장")
    print("  • JSON 메타데이터로 추적 가능")
    print("  • Open3D로 인터랙티브 시각화")
    print(f"\n결과 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()