"""
카메라 관련 유틸리티
- 카메라 내부 파라미터 관리
- Pinhole 카메라 모델 구현
- 2D ↔ 3D 변환
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any


class CameraIntrinsics:
    """
    카메라 내부 파라미터 클래스
    Isaac Sim 카메라 설정 (512×512)
    """
    
    def __init__(self, fx=None, fy=None, cx=None, cy=None, width=512, height=512):
        """
        카메라 내부 파라미터 초기화
        
        Args:
            fx, fy: 초점거리 (픽셀 단위)
            cx, cy: 주점 (principal point)
            width, height: 이미지 크기
        """
        # Isaac Sim 기본값
        self.fx = fx if fx is not None else 395.26
        self.fy = fy if fy is not None else 395.26
        self.cx = cx if cx is not None else 256.0
        self.cy = cy if cy is not None else 256.0
        self.width = width
        self.height = height
    
    def get_matrix(self):
        """
        카메라 내부 행렬 K 반환
        
        Returns:
            3×3 내부 파라미터 행렬
        """
        return np.array([
            [self.fx, 0,       self.cx],
            [0,       self.fy, self.cy],
            [0,       0,       1]
        ])
    
    def __str__(self):
        """문자열"""
        return (f"CameraIntrinsics(fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.2f}, cy={self.cy:.2f}, "
                f"size={self.width}×{self.height})")
    
    def __repr__(self):
        return self.__str__()


