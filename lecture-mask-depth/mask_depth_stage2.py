#!/usr/bin/env python3
"""
Stage 2: YOLO 세그멘테이션 추가
- Stage 1 기능 포함
- YOLO 모델 로드
- 세그멘테이션 추론
- 마스크 정제

새로운 학습 포인트:
- YOLO 인스턴스 세그멘테이션
- Morphological operations
- 마스크 시각화
"""

import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib
from typing import Tuple, Optional

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
MODEL_PATH = "models/best.pt"  # 상위 디렉토리의 모델

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


# ====== Stage 1 함수들 (재사용) ======
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


# ====== NEW: Stage 2 함수들 ======
def load_yolo_model() -> YOLO:
    """
    학습된 YOLO 세그멘테이션 모델 로드
    
    Returns:
        YOLO 모델
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
    
    model = YOLO(MODEL_PATH)
    print(f"  ✓ YOLO 모델 로드: {MODEL_PATH}")
    print(f"    - 클래스: {', '.join(CLASS_NAMES)}")
    
    return model


def perform_segmentation(image: np.ndarray, model: YOLO, conf_threshold: float = 0.5) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    YOLO 세그멘테이션 수행
    
    Args:
        image: 입력 이미지
        model: YOLO 모델
        conf_threshold: 신뢰도 임계값
    
    Returns:
        masks, boxes, confidences, classes
    """
    # YOLO 추론
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
    """
    Morphological operations으로 마스크 정제
    
    Args:
        mask: 원본 마스크
    
    Returns:
        정제된 마스크
    """
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 이진화
    mask_binary = (mask > 0.5).astype(np.uint8)

    # Close operation (작은 구멍 채우기)
    mask_closed = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
    
    # Open operation (노이즈 제거)
    mask_refined = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
    
    return mask_refined


def visualize_segmentation(rgb_image: np.ndarray, masks: Optional[np.ndarray], classes: Optional[np.ndarray]) -> None:
    """
    세그멘테이션 결과 시각화
    
    Args:
        rgb_image: RGB 이미지
        masks: 검출된 마스크들
        classes: 클래스 인덱스들
    """
    # 오버레이 이미지 생성
    overlay = rgb_image.copy()
    
    if masks is not None:
        h, w = rgb_image.shape[:2]
        
        for i, (mask, cls) in enumerate(zip(masks, classes)):
            # 마스크 리사이즈
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # 마스크 정제
            mask_refined = refine_mask(mask_resized)
            
            # 색상 적용
            color = CLASS_COLORS[cls] if cls < len(CLASS_COLORS) else (255, 255, 255)
            
            # 마스크 영역에 색상 오버레이
            mask_bool = mask_refined.astype(bool)
            overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array(color) * 0.5
    
    # 시각화
    plt.figure(figsize=(15, 5))
    
    # 원본 이미지
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # 세그멘테이션 결과
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title('Segmentation Result')
    plt.axis('off')
    
    # 개별 마스크
    plt.subplot(1, 3, 3)
    if masks is not None:
        combined_mask = np.zeros((h, w))
        for i, (mask, cls) in enumerate(zip(masks, classes)):
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_refined = refine_mask(mask_resized)
            combined_mask[mask_refined > 0] = cls + 1

        vmax = len(CLASS_NAMES)
        plt.imshow(combined_mask, cmap='tab10')
        plt.title('Individual Masks')
        
        # 범례 추가
        for i, cls in enumerate(np.unique(classes)):
            class_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"class_{cls}"
            value_in_mask = cls + 1
            color = plt.cm.tab10(value_in_mask / vmax)
            plt.scatter([], [], c=[color], label=class_name, s=100)
        plt.legend(loc='upper right')
    else:
        plt.text(0.5, 0.5, 'No masks detected', ha='center', va='center')
    plt.axis('off')
    
    plt.tight_layout()
    
    # 파일로 저장
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "stage2_segmentation.png"
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
    print("Stage 2: YOLO 세그멘테이션 추가")
    print("=" * 60)
    print("\n학습 목표:")
    print("  1. YOLO 인스턴스 세그멘테이션 이해")
    print("  2. Morphological operations으로 마스크 정제")
    print("  3. 세그멘테이션 결과 시각화")
    
    # 프레임 인덱스 설정
    frame_idx = 83
    print(f"\n처리할 프레임: {frame_idx:04d}")
    
    # Step 1: 데이터 로드 (Stage 1)
    print_step(1, "데이터 로드")
    rgb_image = load_rgb_image(frame_idx)
    depth_map = load_depth_map(frame_idx)
    
    # NEW Step 2: YOLO 모델 로드
    print_step(2, "YOLO 모델 로드")
    model = load_yolo_model()
    
    # NEW Step 3: 세그멘테이션 수행
    print_step(3, "세그멘테이션 수행")
    masks, boxes, confidences, classes = perform_segmentation(rgb_image, model)
    
    # NEW Step 4: 세그멘테이션 시각화
    print_step(4, "세그멘테이션 결과 시각화")
    visualize_segmentation(rgb_image, masks, classes)
    
    # 완료
    print("\n" + "=" * 60)
    print("✓ Stage 2 완료!")
    print("=" * 60)
    print("\n핵심 포인트:")
    print("  • YOLO는 객체별로 정확한 마스크 생성")
    print("  • Morphological operations으로 마스크 품질 향상")
    print("  • 각 객체는 고유한 인스턴스로 구분됨")
    print("\n다음 단계: Stage 3에서 3D 변환 추가")


if __name__ == "__main__":
    main()