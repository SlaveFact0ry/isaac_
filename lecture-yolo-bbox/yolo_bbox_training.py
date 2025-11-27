#!/usr/bin/env python3
"""
실습 : YOLOv11 전이학습 - Isaac Sim 합성 데이터 활용
Isaac Sim Replicator로 생성한 데이터로 YOLOv11 모델 학습
"""

import sys
import json
import shutil
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
import cv2

# YOLO 관련 imports
try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("Error: ultralytics가 설치되지 않았습니다.")
    print("설치: pip install ultralytics torch torchvision")
    sys.exit(1)


class IsaacToYOLOConverter:
    """Isaac Sim 데이터를 YOLO 형식으로 변환"""
    
    def __init__(self, isaac_dir, output_dir):
        self.isaac_dir = Path(isaac_dir)
        self.output_dir = Path(output_dir)
        self.class_mapping = {}
        self.class_names = {}
        
    def load_class_mappings(self):
        """클래스 매핑 정보 로드"""
        labels_file = self.isaac_dir / "bounding_box_2d_tight" / "bounding_box_2d_tight_labels_0000.json"
        
        if not labels_file.exists():
            raise FileNotFoundError(f"레이블 파일을 찾을 수 없습니다: {labels_file}")
            
        with open(labels_file, 'r') as f:
            labels_data = json.load(f)
            
        # semantic_id -> yolo_class_id 매핑
        yolo_id = 0
        for semantic_id, class_info in labels_data.items():
            # Isaac Sim Replicator는 0부터 시작하는 semantic_id 사용
            # 모든 클래스 포함 (배경은 별도로 처리됨)
            class_name = class_info['class']
            self.class_mapping[int(semantic_id)] = yolo_id
            self.class_names[yolo_id] = class_name
            yolo_id += 1
            
        print(f"✓ {len(self.class_mapping)}개 클래스 로드 완료")
        for yolo_id, name in self.class_names.items():
            print(f"  [{yolo_id}] {name}")
            
    def setup_directories(self):
        """YOLO 디렉토리 구조 생성"""
        for split in ['train', 'val', 'test']:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
            
    def parse_bbox_data(self, bbox_data, img_width, img_height):
        """Isaac Sim bbox 데이터를 YOLO 형식으로 변환"""
        yolo_annotations = []
        
        if bbox_data is None or len(bbox_data) == 0:
            return yolo_annotations
            
        for bbox in bbox_data:
            # semantic ID 확인
            semantic_id = int(bbox['semanticId'])
            if semantic_id not in self.class_mapping:
                continue
                
            class_id = self.class_mapping[semantic_id]
            
            # 바운딩 박스 좌표
            x_min = float(bbox['x_min'])
            y_min = float(bbox['y_min'])
            x_max = float(bbox['x_max'])
            y_max = float(bbox['y_max'])
            
            # 유효성 검사
            if x_max <= x_min or y_max <= y_min:
                continue
                
            # YOLO 형식으로 변환 (정규화된 중심 좌표와 크기)
            x_center = (x_min + x_max) / 2.0 / img_width
            y_center = (y_min + y_max) / 2.0 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
            
            # 값 범위 제한
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            # 너무 작은 박스 제외
            if width < 0.01 or height < 0.01:
                continue
                
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
        return yolo_annotations
        
    def convert_dataset(self, train_ratio=0.7, val_ratio=0.2):
        """전체 데이터셋 변환"""
        print("\n[데이터 변환 시작]")
        
        # 클래스 매핑 로드
        self.load_class_mappings()
        
        # 디렉토리 구조 생성
        self.setup_directories()
        
        # RGB 이미지와 bbox 파일 매칭
        rgb_dir = self.isaac_dir / "rgb"
        bbox_dir = self.isaac_dir / "bounding_box_2d_tight"
        
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
        total_objects = 0
        
        for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            if not files:
                continue
                
            print(f"\n{split.upper()} 세트 처리 중...")
            
            for rgb_path in tqdm(files, desc=f"{split} 변환"):
                # 대응하는 bbox 파일 찾기
                frame_num = rgb_path.stem.split('_')[1]
                bbox_path = bbox_dir / f"bounding_box_2d_tight_{frame_num}.npy"
                
                if not bbox_path.exists():
                    continue
                    
                # 이미지 로드
                img = cv2.imread(str(rgb_path))
                if img is None:
                    continue
                    
                img_height, img_width = img.shape[:2]
                
                # bbox 데이터 로드 및 변환
                bbox_data = np.load(bbox_path, allow_pickle=True)
                yolo_annotations = self.parse_bbox_data(bbox_data, img_width, img_height)
                
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
                total_objects += len(yolo_annotations)
                
        # YAML 설정 파일 생성
        self.create_yaml_config()
        
        # 변환 통계 출력
        print(f"\n[변환 완료]")
        print(f"✓ 총 {sum(stats.values())}개 이미지 변환")
        print(f"✓ 총 {total_objects}개 객체 어노테이션")
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


class YOLOTrainer:
    """YOLOv11 모델 학습 관리"""
    
    def __init__(self, data_yaml, model_name='yolo11n.pt'):
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
            
    def train(self, epochs=50, batch_size=16, imgsz=640):
        """모델 학습"""
        print(f"\n[YOLOv11 학습 시작]")
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
            'project': 'runs/train',
            'name': 'yolo11_isaac_sim',
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
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 10.0,
            'translate': 0.1,
            'scale': 0.5,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
        }
        
        # 학습 실행
        print("\n학습 진행 중...")
        results = self.model.train(**training_args)
        
        print("\n✓ 학습 완료!")
        print(f"  최종 mAP50: {results.maps[0]:.4f}")
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
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        
        # 클래스별 성능
        if hasattr(metrics.box, 'ap_class_index'):
            print(f"\n클래스별 mAP50:")
            for i, class_idx in enumerate(metrics.box.ap_class_index):
                if int(class_idx) in self.model.names:
                    class_name = self.model.names[int(class_idx)]
                    if hasattr(metrics.box, 'ap50'):
                        map50 = metrics.box.ap50[i]
                        print(f"  {class_name}: {map50:.4f}")
                        
        return metrics
        
    def inference_test(self, test_image_path=None):
        """추론 테스트"""
        if self.model is None:
            print("⚠ 모델이 학습되지 않았습니다.")
            return
            
        print("\n[추론 테스트]")
        
        # 테스트 이미지 선택
        if test_image_path is None:
            # 테스트 세트에서 임의 선택
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
            if len(r.boxes) > 0:
                print(f"\n검출된 객체: {len(r.boxes)}개")
                for box in r.boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    class_name = self.model.names[cls]
                    print(f"  - {class_name}: {conf:.2%}")
            else:
                print("검출된 객체가 없습니다.")
                
        # 결과 이미지 저장
        output_path = Path("inference_result.jpg")
        annotated = results[0].plot()
        cv2.imwrite(str(output_path), annotated)
        print(f"\n✓ 결과 이미지 저장: {output_path}")


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='YOLOv11 전이학습 - Isaac Sim 합성 데이터')
    
    # 데이터 경로
    parser.add_argument('--isaac-dir', type=str,
                       default='../lecture-standalone/replicator_output/advanced_dataset/Replicator_04',
                       help='Isaac Sim 데이터 디렉토리')
    parser.add_argument('--output-dir', type=str,
                       default='./yolo_dataset',
                       help='YOLO 데이터셋 출력 디렉토리')
    
    # 학습 설정
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                       help='사용할 YOLO 모델')
    parser.add_argument('--epochs', type=int, default=50,
                       help='학습 에폭 수')
    parser.add_argument('--batch-size', type=int, default=16,
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
    print(" YOLOv11 전이학습 실습")
    print(" Isaac Sim 합성 데이터 활용")
    print("="*60)
    
    # 1. 데이터 변환
    if not args.train_only and not args.inference_only:
        converter = IsaacToYOLOConverter(args.isaac_dir, args.output_dir)
        converter.convert_dataset()
        
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
            
        trainer = YOLOTrainer(data_yaml, args.model)
        trainer.train(
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
        trainer = YOLOTrainer(data_yaml)
        trainer.model = YOLO(args.weights)
        trainer.inference_test()
        
    print(f"\n{'='*60}")
    print(" 실습 완료!")
    print("="*60)


if __name__ == "__main__":
    main()