# -*- coding: utf-8 -*-
"""
YOLO目标检测模块

功能：
- 基于YOLOv8的目标检测推理
- 支持模型热加载和自动降级
- 与TemplateMatcher接口兼容
- 支持多分辨率自适应检测
"""

import os
import sys
import cv2
import numpy as np
from typing import Optional, List, Dict, Tuple
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLO目标检测器"""

    def __init__(self, model_path: str = None, confidence: float = 0.5, device: str = 'cpu'):
        """
        初始化YOLO检测器

        Args:
            model_path: YOLO模型路径（.pt文件），None则使用默认路径
            confidence: 默认置信度阈值
            device: 推理设备 ('cpu', 'cuda', '0', '0,1' 等)
        """
        self.confidence = confidence
        self.device = device
        self.model = None
        self.model_path = model_path
        self.class_names = {}
        self._loaded = False

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

    def _load_model(self, model_path: str) -> bool:
        """加载YOLO模型"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model_path = model_path

            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
            self._loaded = True
            logger.info(f"YOLO模型加载成功: {model_path}, 类别数: {len(self.class_names)}")
            return True
        except ImportError:
            logger.warning("ultralytics未安装，请运行: pip install ultralytics")
            return False
        except Exception as e:
            logger.error(f"YOLO模型加载失败: {e}")
            return False

    def load_model(self, model_path: str = None) -> bool:
        """公开接口：加载模型"""
        if model_path:
            return self._load_model(model_path)
        if self.model_path:
            return self._load_model(self.model_path)
        return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None

    def detect(
        self,
        screenshot: np.ndarray,
        confidence: float = None,
        target_classes: List[str] = None
    ) -> List[Dict]:
        """
        执行目标检测

        Args:
            screenshot: 截图图像数组（BGR格式）
            confidence: 置信度阈值，None则使用默认值
            target_classes: 只返回指定类别的结果，None返回所有

        Returns:
            检测结果列表，每项格式：
            {
                'class_name': str,      # 类别名称
                'class_id': int,        # 类别ID
                'confidence': float,    # 置信度
                'center_x': int,        # 中心X坐标
                'center_y': int,        # 中心Y坐标
                'bbox': (x, y, w, h),   # 边界框
            }
        """
        if not self.is_loaded:
            logger.warning("YOLO模型未加载")
            return []

        if screenshot is None or screenshot.size == 0:
            return []

        conf = confidence if confidence is not None else self.confidence

        try:
            results = self.model.predict(
                screenshot,
                conf=conf,
                device=self.device,
                verbose=False,
                imgsz=640
            )
        except Exception as e:
            logger.error(f"YOLO推理失败: {e}")
            return []

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.class_names.get(cls_id, f'class_{cls_id}')
                conf_val = float(box.conf[0])

                if target_classes and cls_name not in target_classes:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                cx, cy = x + w // 2, y + h // 2

                detections.append({
                    'class_name': cls_name,
                    'class_id': cls_id,
                    'confidence': conf_val,
                    'center_x': cx,
                    'center_y': cy,
                    'bbox': (x, y, w, h)
                })

        detections.sort(key=lambda d: d['confidence'], reverse=True)
        return detections

    def detect_single(
        self,
        screenshot: np.ndarray,
        class_name: str,
        confidence: float = None
    ) -> Optional[Dict]:
        """
        检测单个指定类别（返回最高置信度结果）

        Args:
            screenshot: 截图图像数组（BGR格式）
            class_name: 目标类别名称
            confidence: 置信度阈值

        Returns:
            匹配成功返回字典，失败返回None
        """
        results = self.detect(screenshot, confidence, target_classes=[class_name])
        if results:
            return results[0]
        return None

    def detect_all(
        self,
        screenshot: np.ndarray,
        class_name: str,
        confidence: float = None
    ) -> List[Dict]:
        """
        检测指定类别的所有实例

        Args:
            screenshot: 截图图像数组（BGR格式）
            class_name: 目标类别名称
            confidence: 置信度阈值

        Returns:
            检测结果列表
        """
        return self.detect(screenshot, confidence, target_classes=[class_name])

    def detect_region(
        self,
        screenshot: np.ndarray,
        region: Tuple[int, int, int, int],
        confidence: float = None,
        target_classes: List[str] = None
    ) -> List[Dict]:
        """
        在指定区域内执行目标检测

        Args:
            screenshot: 截图图像数组（BGR格式）
            region: 检测区域 (x1, y1, x2, y2)
            confidence: 置信度阈值
            target_classes: 只返回指定类别的结果

        Returns:
            检测结果列表（坐标已转换为全图坐标）
        """
        if screenshot is None or screenshot.size == 0:
            return []

        x1, y1, x2, y2 = region
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(screenshot.shape[1], int(x2))
        y2 = min(screenshot.shape[0], int(y2))

        roi = screenshot[y1:y2, x1:x2]
        if roi.size == 0:
            return []

        detections = self.detect(roi, confidence, target_classes)

        for det in detections:
            det['center_x'] += x1
            det['center_y'] += y1
            bx, by, bw, bh = det['bbox']
            det['bbox'] = (bx + x1, by + y1, bw, bh)

        return detections

    def get_class_names(self) -> Dict[int, str]:
        """获取所有类别名称"""
        return self.class_names.copy()

    def get_class_id(self, class_name: str) -> Optional[int]:
        """根据类别名称获取类别ID"""
        for cid, cname in self.class_names.items():
            if cname == class_name:
                return cid
        return None


def create_detector_from_config(config_path: str = None) -> YOLODetector:
    """
    根据配置文件创建检测器

    Args:
        config_path: 配置文件路径，None使用默认配置

    Returns:
        YOLODetector实例
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_model = os.path.join(base_dir, 'yolo_models', 'eve_detector.pt')

    if config_path and os.path.exists(config_path):
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return YOLODetector(
            model_path=config.get('model_path', default_model),
            confidence=config.get('confidence', 0.5),
            device=config.get('device', 'cpu')
        )

    return YOLODetector(model_path=default_model, confidence=0.5, device='cpu')
