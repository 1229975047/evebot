# -*- coding: utf-8 -*-
"""
模板匹配模块

功能：
- 单模板匹配（返回最佳匹配位置）
- 多模板匹配（返回所有匹配位置，非极大值抑制去重）
- 支持中文路径

算法：OpenCV的TM_CCOEFF_NORMED（归一化相关系数匹配）
"""

import cv2
import numpy as np
from typing import Optional, List, Dict
import logging

# 配置日志
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def imread_unicode(filepath: str) -> Optional[np.ndarray]:
    """
    支持中文路径的图像读取

    Args:
        filepath: 图片路径（支持中文）

    Returns:
        OpenCV图像数组（BGR格式），失败返回None
    """
    import os
    import sys
    try:
        # 临时抑制libpng的iCCP警告（不影响功能）
        stderr_fd = sys.stderr.fileno()
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(stderr_fd)
        os.dup2(devnull, stderr_fd)
        try:
            with open(filepath, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        finally:
            os.dup2(old_stderr, stderr_fd)
            os.close(devnull)
        return image
    except Exception as e:
        logger.error(f"读取图像失败 {filepath}: {e}")
        return None


class TemplateMatcher:
    """
    模板匹配器

    基于OpenCV的模板匹配功能，用于在截图中查找目标图像的位置
    """

    def __init__(self):
        pass

    def _load_template(self, template_path: str) -> Optional[np.ndarray]:
        """加载模板图像（支持中文路径）"""
        return imread_unicode(template_path)

    def _convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """将图像转换为灰度图（如果需要）"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def _create_match_result(self, x: int, y: int, w: int, h: int, confidence: float) -> Dict:
        """
        创建标准化的匹配结果字典

        Args:
            x, y: 匹配区域左上角坐标
            w, h: 匹配区域宽高
            confidence: 置信度

        Returns:
            包含中心坐标、置信度、包围框的字典
        """
        return {
            'center_x': int(x + w // 2),  # 中心X坐标
            'center_y': int(y + h // 2),  # 中心Y坐标
            'confidence': float(confidence),  # 匹配置信度
            'bbox': (x, y, w, h)  # 包围框 (左上x, 左上y, 宽, 高)
        }

    def find_template(
        self,
        screenshot: np.ndarray,
        template_path: str,
        threshold: float = 0.8
    ) -> Optional[Dict]:
        """
        查找单个最佳匹配

        Args:
            screenshot: 截图图像数组（BGR格式）
            template_path: 模板图片路径（支持中文）
            threshold: 匹配阈值（0.0-1.0），默认0.8

        Returns:
            匹配成功返回字典 {center_x, center_y, confidence, bbox}
            匹配失败返回 None
        """
        template = self._load_template(template_path)
        if template is None:
            logger.error(f"加载模板失败: {template_path}")
            return None

        # 转换为灰度图进行匹配
        screenshot_gray = self._convert_to_grayscale(screenshot)
        template_gray = self._convert_to_grayscale(template)

        # 执行模板匹配（归一化相关系数）
        result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)

        # 获取最佳匹配位置
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            h, w = template_gray.shape[:2]
            logger.info(f"模板匹配成功: ({max_loc[0]}, {max_loc[1]}), 置信度: {max_val:.3f}")
            return self._create_match_result(max_loc[0], max_loc[1], w, h, max_val)

        logger.debug(f"模板匹配失败 (最佳: {max_val:.3f}, 阈值: {threshold})")
        return None

    def find_all_matches(
        self,
        screenshot: np.ndarray,
        template_path: str,
        threshold: float = 0.8
    ) -> List[Dict]:
        """
        查找所有匹配（带非极大值抑制）

        Args:
            screenshot: 截图图像数组（BGR格式）
            template_path: 模板图片路径（支持中文）
            threshold: 匹配阈值（0.0-1.0），默认0.8

        Returns:
            匹配结果列表，按置信度降序排列
            每项包含 {center_x, center_y, confidence, bbox}
        """
        template = self._load_template(template_path)
        if template is None:
            logger.error(f"加载模板失败: {template_path}")
            return []

        # 转换为灰度图
        screenshot_gray = self._convert_to_grayscale(screenshot)
        template_gray = self._convert_to_grayscale(template)

        # 执行模板匹配
        result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)

        h, w = template_gray.shape[:2]

        # 找出所有超过阈值的匹配位置
        locations = np.where(result >= threshold)

        if len(locations[0]) == 0:
            logger.debug(f"未找到匹配 (阈值: {threshold})")
            return []

        # 收集所有匹配及其置信度
        matches = []
        for y, x in zip(locations[0], locations[1]):
            confidence = result[y, x]
            matches.append({
                'x': int(x),
                'y': int(y),
                'w': w,
                'h': h,
                'confidence': float(confidence)
            })

        # 按置信度降序排列
        matches.sort(key=lambda m: m['confidence'], reverse=True)

        # 应用非极大值抑制去除重叠匹配
        filtered = self._non_maximum_suppression(matches)

        # 转换为标准格式
        results = [
            self._create_match_result(m['x'], m['y'], m['w'], m['h'], m['confidence'])
            for m in filtered
        ]

        logger.info(f"找到 {len(results)} 个匹配 (原始: {len(matches)} 个超过阈值)")
        return results

    def _non_maximum_suppression(self, matches: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        非极大值抑制 - 去除重叠的匹配框

        Args:
            matches: 匹配列表
            iou_threshold: IOU阈值，超过此值认为重叠

        Returns:
            过滤后的匹配列表
        """
        if not matches:
            return []

        # 转换为numpy数组
        boxes = np.array([[m['x'], m['y'], m['x'] + m['w'], m['y'] + m['h']] for m in matches])
        scores = np.array([m['confidence'] for m in matches])

        # 计算每个框的面积
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # 按置信度降序排列
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # 计算与剩余框的IOU
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)

            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # 保留IOU低于阈值的框
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return [matches[i] for i in keep]
