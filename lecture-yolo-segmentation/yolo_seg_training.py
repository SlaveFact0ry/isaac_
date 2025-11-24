#!/usr/bin/env python3
"""
실습 : YOLOv11 세그멘테이션 전이학습
Isaac Sim Replicator로 생성한 Instance Segmentation 데이터로 YOLOv11 모델 학습
"""

import sys
import json
import shutil
import argparse
import ast
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
import cv2
import time
from PIL import Image
from datetime import datetime

# YOLO 관련 imports
try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("Error: ultralytics가 설치되지 않았습니다.")
    print("설치: pip install ultralytics torch torchvision")
    sys.exit(1)

print("=" * 50)
print("YOLOv11 세그멘테이션 전이학습 실습")
print("픽셀 단위 객체 분할 모델 구현")
print("=" * 50)
print()


class IsaacToYOLOSegmentationConverter:
    """Isaac Sim instance segmentation 데이터를 YOLO 형식으로 변환"""
    
    def __init__(self, isaac_dir, output_dir):
        self.isaac_dir = Path(isaac_dir)
        self.output_dir = Path(output_dir)
        self.class_mapping = {}
        self.class_names = {}
        
    def load_class_mappings(self):
        """Isaac Sim instance segmentation 매핑 파일에서 클래스 정보 로드"""
        # instance_segmentation 디렉토리 확인
        seg_dir = self.isaac_dir / "instance_segmentation"
        
        if not seg_dir.exists():
            raise FileNotFoundError(f"Instance segmentation 디렉토리를 찾을 수 없습니다: {seg_dir}")
        
        # semantics mapping 파일 찾기
        mapping_files = list(seg_dir.glob("*semantics_mapping*.json"))
        
        if not mapping_files:
            raise FileNotFoundError("Instance segmentation 매핑 파일을 찾을 수 없습니다.")
        
        # 첫 번째 매핑 파일 로드
        with open(mapping_files[0], 'r') as f:
            mapping_data = json.load(f)
        
        # RGBA 색상 -> 클래스 매핑 생성
        yolo_id = 0
        self.rgba_to_class = {}
        
        for rgba_str, class_info in mapping_data.items():
            class_name = class_info['class']
            
            # 배경과 레이블 없는 것 제외
            if class_name.upper() in ['BACKGROUND', 'UNLABELLED']:
                continue
                
            # RGBA 문자열을 튜플로 안전하게 변환
            # eval() 대신 ast.literal_eval() 사용
            rgba = ast.literal_eval(rgba_str)  # "(r, g, b, a)" -> (r, g, b, a)
            
            self.rgba_to_class[rgba] = class_name
            
            # 클래스가 아직 매핑되지 않았으면 추가
            if class_name not in self.class_names.values():
                self.class_names[yolo_id] = class_name
                yolo_id += 1
        
        print(f"✓ {len(self.class_names)}개 클래스 로드 완료")
        for yid, name in self.class_names.items():
            print(f"  [{yid}] {name}")
            
    def setup_directories(self):
        """YOLO 디렉토리 구조 생성"""
        for split in ['train', 'val', 'test']:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
            
    def extract_polygons_from_mask(self, instance_mask):
        """바이너리 마스크에서 polygon 좌표 추출"""
        # Contour 찾기
        contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        h, w = instance_mask.shape
        
        for contour in contours:
            # Contour 단순화 (Douglas-Peucker)
            epsilon = 0.001 * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            
            # 최소 3개 점 필요
            if len(simplified) >= 3:
                polygon = simplified.reshape(-1, 2).astype(np.float32)
                # 정규화 (0~1)
                polygon[:, 0] /= w
                polygon[:, 1] /= h
                polygons.append(polygon.flatten())
                
        return polygons
    
    def convert_frame(self, frame_num, seg_path):
        """단일 프레임을 YOLO 세그멘테이션 형식으로 변환"""
        # 세그멘테이션 이미지 로드
        seg_img = np.array(Image.open(seg_path))
        
        if seg_img is None:
            return []
        
        # RGBA 형식 확인
        if len(seg_img.shape) == 3 and seg_img.shape[2] == 3:
            # RGB를 RGBA로 변환
            rgba_img = np.zeros((seg_img.shape[0], seg_img.shape[1], 4), dtype=np.uint8)
            rgba_img[:, :, :3] = seg_img
            rgba_img[:, :, 3] = 255
            seg_img = rgba_img
        
        yolo_annotations = []
        
        # 이미지에서 유니크한 색상 추출
        unique_colors = np.unique(seg_img.reshape(-1, 4), axis=0)
        
        for color in unique_colors:
            color_tuple = tuple(color)
            
            # 배경 스킵
            if color_tuple in [(0, 0, 0, 0), (0, 0, 0, 255)]:
                continue
            
            # 클래스 찾기
            class_name = self.rgba_to_class.get(color_tuple)
            if not class_name:
                continue
            
            # YOLO 클래스 ID 찾기
            yolo_class = None
            for yid, name in self.class_names.items():
                if name == class_name:
                    yolo_class = yid
                    break
            
            if yolo_class is None:
                continue
            
            # 해당 색상의 마스크 생성
            instance_mask = np.all(seg_img == color, axis=-1).astype(np.uint8)
            
            # Polygon 추출
            polygons = self.extract_polygons_from_mask(instance_mask)
            
            for polygon in polygons:
                # YOLO 형식: class_id x1 y1 x2 y2 ... xn yn
                annotation = f"{yolo_class} " + " ".join([f"{coord:.6f}" for coord in polygon])
                yolo_annotations.append(annotation)
        
        return yolo_annotations
    
    def convert_dataset(self, train_ratio=0.7, val_ratio=0.2):
        """전체 데이터셋 변환"""
        print("\n[데이터 변환 시작]")
        print("[Step 1] 클래스 매핑 로드 중...")
        
        # 클래스 매핑 로드
        self.load_class_mappings()
        print("✓ 클래스 매핑 완료")
        
        print("\n[Step 2] 디렉토리 구조 생성 중...")
        # 디렉토리 구조 생성
        self.setup_directories()
        print("✓ 디렉토리 생성 완료")
        
        print("\n[Step 3] 이미지 파일 검색 중...")
        # RGB 이미지와 세그멘테이션 매칭
        rgb_dir = self.isaac_dir / "rgb"
        seg_dir = self.isaac_dir / "instance_segmentation"
        
        rgb_files = sorted(list(rgb_dir.glob("rgb_*.png")))
        print(f"✓ {len(rgb_files)}개 이미지 발견")
        
        # 데이터 분할
        num_train = int(len(rgb_files) * train_ratio)
        num_val = int(len(rgb_files) * val_ratio)
        
        train_files = rgb_files[:num_train]
        val_files = rgb_files[num_train:num_train + num_val]
        test_files = rgb_files[num_train + num_val:]
        
        print(f"✓ 데이터 분할: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        
        # 각 분할별 처리
        stats = {'train': 0, 'val': 0, 'test': 0}
        total_polygons = 0
        
        print("\n[Step 4] 데이터 변환 중...")
        for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            if not files:
                continue
                
            print(f"\n{split.upper()} 세트 처리 중...")
            
            for rgb_path in tqdm(files, desc=f"{split} 변환"):
                # 프레임 번호 추출
                frame_num = int(rgb_path.stem.split('_')[-1])
                
                # 대응하는 세그멘테이션 파일 찾기
                seg_path = seg_dir / f"instance_segmentation_{frame_num:04d}.png"
                
                if not seg_path.exists():
                    continue
                
                # YOLO 어노테이션 생성
                yolo_annotations = self.convert_frame(frame_num, seg_path)
                
                if not yolo_annotations:
                    continue
                
                # 이미지 복사
                dest_img_path = self.output_dir / 'images' / split / rgb_path.name
                shutil.copy2(rgb_path, dest_img_path)
                
                # 레이블 저장
                label_path = self.output_dir / 'labels' / split / f"{rgb_path.stem}.txt"
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
                stats[split] += 1
                total_polygons += len(yolo_annotations)
        
        print("\n[Step 5] YAML 설정 파일 생성 중...")
        # YAML 설정 파일 생성
        self.create_yaml_config()
        print("✓ 설정 파일 생성 완료")
        
        # 변환 통계 출력
        print(f"\n[변환 완료]")
        print(f"✓ 총 {sum(stats.values())}개 이미지 변환")
        print(f"✓ 총 {total_polygons}개 polygon 어노테이션")
        for split, count in stats.items():
            print(f"  - {split}: {count}개")
            
        return stats
    
    def create_yaml_config(self):
        """YOLO 학습용 dataset.yaml 생성"""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': self.class_names,
            'nc': len(self.class_names)
        }
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
        print(f"✓ 설정 파일 생성: {yaml_path}")


class YOLOSegmentationTrainer:
    """YOLOv11 세그멘테이션 모델 학습 관리"""
    
    def __init__(self, data_yaml, model_name='yolo11n-seg.pt'):
        self.data_yaml = Path(data_yaml)
        self.model_name = model_name
        self.model = None
        self.device = self.get_device()
        
    def get_device(self):
        """사용 가능한 디바이스 확인"""
        if torch.cuda.is_available():
            device = 0
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ GPU 사용: {gpu_name} ({gpu_memory:.1f}GB)")
            return device
        else:
            print("⚠ GPU를 사용할 수 없습니다. CPU로 학습합니다.")
            return 'cpu'
    
    def train(self, epochs=50, batch_size=8, imgsz=640):
        """모델 학습"""
        print(f"\n[YOLOv11 세그멘테이션 학습 시작]")
        print(f"모델: {self.model_name}")
        print(f"에폭: {epochs}")
        print(f"배치 크기: {batch_size}")
        print(f"이미지 크기: {imgsz}")
        
        # 모델 초기화
        self.model = YOLO(self.model_name)
        
        # 학습 설정
        training_args = {
            'data': str(self.data_yaml),
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': imgsz,
            'device': self.device,
            'project': 'runs/segment',
            'name': f'yolo11_seg_isaac_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': True,
            'patience': 20,
            'save': True,
            'plots': True,
            'verbose': True,
            
            # 최적화 설정
            'optimizer': 'auto',
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            
            # 증강 설정
            'hsv_h': 0.015,  # 색조
            'hsv_s': 0.7,    # 채도
            'hsv_v': 0.4,    # 명도
            'degrees': 10.0,
            'translate': 0.1,
            'scale': 0.5,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.3,  # 세그멘테이션 특화 증강
        }
        
        # 학습 실행
        print("\n학습 진행 중...")
        start_time = time.time()
        
        results = self.model.train(**training_args)
        
        end_time = time.time()
        print(f"\n✓ 학습 완료! (소요 시간: {(end_time - start_time)/60:.2f}분)")
        
        # 결과 저장 경로
        if hasattr(results, 'save_dir'):
            print(f"  모델 저장 위치: {results.save_dir}")
        
        return results
    
    def evaluate(self, test_split='test'):
        """테스트 세트 평가"""
        if self.model is None:
            print("⚠ 모델이 학습되지 않았습니다.")
            return None
            
        print(f"\n[{test_split.upper()} 세트 평가]")
        
        metrics = self.model.val(
            data=str(self.data_yaml),
            split=test_split,
            verbose=True
        )
        
        print(f"\n평가 결과:")
        if hasattr(metrics, 'box'):
            print(f"  Box mAP50: {metrics.box.map50:.4f}")
            print(f"  Box mAP50-95: {metrics.box.map:.4f}")
        if hasattr(metrics, 'seg'):
            print(f"  Mask mAP50: {metrics.seg.map50:.4f}")
            print(f"  Mask mAP50-95: {metrics.seg.map:.4f}")
            
        return metrics
    
    def inference_test(self, test_image_path=None):
        """추론 테스트 및 시각화"""
        if self.model is None:
            print("⚠ 모델이 학습되지 않았습니다.")
            return
            
        print("\n[추론 테스트]")
        
        # 테스트 이미지 선택
        if test_image_path is None:
            test_dir = Path(self.data_yaml).parent / 'images' / 'test'
            test_images = list(test_dir.glob("*.png"))
            if test_images:
                test_image_path = test_images[0]
            else:
                print("⚠ 테스트 이미지를 찾을 수 없습니다.")
                return
                
        print(f"테스트 이미지: {test_image_path}")
        
        # 추론 실행
        results = self.model(test_image_path)
        
        # 결과 출력
        for r in results:
            if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
                print(f"\n검출된 객체: {len(r.boxes)}개")
                for i, box in enumerate(r.boxes):
                    cls = int(box.cls)
                    conf = float(box.conf)
                    class_name = self.model.names[cls]
                    print(f"  - {class_name}: {conf:.2%}")
                    
                    # 마스크 정보
                    if hasattr(r, 'masks') and r.masks is not None and i < len(r.masks.data):
                        mask = r.masks.data[i]
                        mask_area = mask.sum().item()
                        total_pixels = mask.numel()
                        coverage = mask_area / total_pixels * 100
                        print(f"    마스크 커버리지: {coverage:.1f}%")
            else:
                print("검출된 객체가 없습니다.")
        
        # 결과 이미지 저장 (마스크 오버레이 포함)
        output_path = Path("segmentation_result.jpg")
        annotated = results[0].plot(masks=True, boxes=True)
        cv2.imwrite(str(output_path), annotated)
        print(f"\n✓ 결과 이미지 저장: {output_path}")


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='YOLOv11 세그멘테이션 전이학습')
    
    # 데이터 경로
    parser.add_argument('--isaac-dir', type=str,
                       default='../lecture-standalone/replicator_output/advanced_dataset/Replicator_04',
                       help='Isaac Sim 데이터 디렉토리')
    parser.add_argument('--output-dir', type=str,
                       default='./yolo_seg_dataset',
                       help='YOLO 데이터셋 출력 디렉토리')
    
    # 학습 설정
    parser.add_argument('--model', type=str, default='yolo11n-seg.pt',
                       help='사용할 YOLO 세그멘테이션 모델')
    parser.add_argument('--epochs', type=int, default=50,
                       help='학습 에폭 수')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='배치 크기')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='입력 이미지 크기')
    
    # 실행 옵션
    parser.add_argument('--convert-only', action='store_true',
                       help='데이터 변환만 수행')
    parser.add_argument('--train-only', action='store_true',
                       help='학습만 수행 (데이터 변환 건너뜀)')
    parser.add_argument('--inference-only', action='store_true',
                       help='추론만 수행')
    parser.add_argument('--weights', type=str,
                       help='추론에 사용할 가중치 파일')
    
    args = parser.parse_args()
    
    print("="*60)
    print(" 실습 6: YOLOv11 세그멘테이션 전이학습")
    print(" 픽셀 단위 객체 분할")
    print("="*60)
    print()
    print(f"설정:")
    print(f"  - Isaac 데이터: {args.isaac_dir}")
    print(f"  - 출력 디렉토리: {args.output_dir}")
    print(f"  - 모델: {args.model}")
    print()
    
    # 1. 데이터 변환
    if not args.train_only and not args.inference_only:
        converter = IsaacToYOLOSegmentationConverter(args.isaac_dir, args.output_dir)
        stats = converter.convert_dataset()
        
        if args.convert_only:
            print("\n✓ 데이터 변환 완료. 프로그램을 종료합니다.")
            return
    
    # 2. 모델 학습
    if not args.convert_only and not args.inference_only:
        data_yaml = Path(args.output_dir) / 'dataset.yaml'
        
        if not data_yaml.exists():
            print(f"⚠ 데이터셋 설정 파일을 찾을 수 없습니다: {data_yaml}")
            print("먼저 데이터 변환을 수행하세요.")
            return
            
        trainer = YOLOSegmentationTrainer(data_yaml, args.model)
        results = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz
        )
        
        # 평가
        trainer.evaluate('test')
        
        # 추론 테스트
        trainer.inference_test()
    
    # 3. 추론만 수행
    if args.inference_only:
        if not args.weights:
            print("⚠ --weights 옵션으로 가중치 파일을 지정하세요.")
            return
            
        data_yaml = Path(args.output_dir) / 'dataset.yaml'
        trainer = YOLOSegmentationTrainer(data_yaml)
        trainer.model = YOLO(args.weights)
        trainer.inference_test()
    
    print(f"\n{'='*60}")
    print(" 실습 완료!")
    print("="*60)


if __name__ == "__main__":
    main()