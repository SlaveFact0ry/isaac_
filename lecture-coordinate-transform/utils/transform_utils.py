"""
좌표 변환 유틸리티
- 회전 행렬과 quaternion 변환
- 좌표계 변환 (카메라 → 월드 → ROS)
- 변환 행렬 연산
"""

import numpy as np
from scipy.spatial.transform import Rotation


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """
    Quaternion을 회전 행렬로 변환
    
    Args:
        quaternion: [x, y, z, w] 형식의 quaternion
    
    Returns:
        3×3 회전 행렬
    """
    # scipy Rotation 사용
    r = Rotation.from_quat(quaternion)
    return r.as_matrix()


def create_transformation_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    4×4 동차 변환 행렬 생성
    
    Args:
        rotation: 3×3 회전 행렬 또는 [x,y,z,w] quaternion
        translation: 3D 이동 벡터 [x, y, z]
    
    Returns:
        4×4 변환 행렬
    """
    T = np.eye(4)
    
    # Quaternion인 경우 회전 행렬로 변환
    if rotation.shape == (4,):
        R = quaternion_to_rotation_matrix(rotation)
    else:
        R = rotation
    
    T[:3, :3] = R
    T[:3, 3] = translation
    
    return T
