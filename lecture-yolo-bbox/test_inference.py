#!/usr/bin/env python3
"""
YOLO 모델 추론 테스트
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import sys

# 모델 경로 (인자로 받거나 기본값 사용)
if len(sys.argv) > 1:
    model_path = sys.argv[1]
else:
    model_path = "runs/train/yolo11_isaac_sim/weights/best.pt"

if not Path(model_path).exists():
    print(f"Error: 모델을 찾을 수 없습니다: {model_path}")
    exit(1)

# 모델 로드
print(f"모델 로드: {model_path}")
model = YOLO(model_path)

# 테스트 이미지 선택
test_dir = Path("yolo_dataset/images/test")
test_images = list(test_dir.glob("*.png"))

if not test_images:
    print("Error: 테스트 이미지가 없습니다.")
    exit(1)

# 첫 번째 테스트 이미지로 추론
test_image = test_images[0]
print(f"테스트 이미지: {test_image.name}")

# 추론 실행
results = model.predict(source=str(test_image), conf=0.25, verbose=False)

# 결과 출력
for r in results:
    if r.boxes is not None and len(r.boxes) > 0:
        print(f"\n검출된 객체: {len(r.boxes)}개")
        for box in r.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            print(f"  - {model.names[cls]}: {conf:.1%}")
    else:
        print("\n검출된 객체 없음")

# 결과 이미지 저장
annotated = results[0].plot()
output_path = "inference_result.jpg"
cv2.imwrite(output_path, annotated)
print(f"\n✓ 결과 저장: {output_path}")