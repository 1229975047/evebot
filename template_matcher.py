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

    def find_template_multiscale(
        self,
        screenshot: np.ndarray,
        template_path: str,
        threshold: float = 0.65,
        scale_range: tuple = (0.5, 1.5),
        scale_steps: int = 20
    ) -> Optional[Dict]:
        """
        多尺度模板匹配 - 在多个尺度上搜索模板以适应不同分辨率

        Args:
            screenshot: 截图图像数组（BGR格式）
            template_path: 模板图片路径（支持中文）
            threshold: 匹配阈值（0.0-1.0），默认0.65
            scale_range: 缩放范围 (min_scale, max_scale)，默认(0.5, 1.5)
            scale_steps: 缩放步数，越多越精确但越慢，默认20

        Returns:
            匹配成功返回字典 {center_x, center_y, confidence, bbox, scale}
            匹配失败返回 None
        """
        template = self._load_template(template_path)
        if template is None:
            logger.error(f"加载模板失败: {template_path}")
            return None

        screenshot_gray = self._convert_to_grayscale(screenshot)
        template_gray = self._convert_to_grayscale(template)

        h, w = template_gray.shape[:2]
        
        # 生成缩放比例列表
        min_scale, max_scale = scale_range
        scales = np.linspace(min_scale, max_scale, scale_steps)
        
        best_match = None
        best_confidence = threshold - 0.01  # 稍微低于阈值，确保能记录最佳结果
        
        for scale in scales:
            # 缩放模板
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # 跳过太小的模板
            if new_w < 10 or new_h < 10:
                continue
            # 跳过大于截图的模板
            if new_w > screenshot_gray.shape[1] or new_h > screenshot_gray.shape[0]:
                continue
                
            scaled_template = cv2.resize(template_gray, (new_w, new_h))
            
            # 执行模板匹配
            try:
                result = cv2.matchTemplate(screenshot_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
            except cv2.error:
                continue
                
            # 获取最佳匹配
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_confidence:
                best_confidence = max_val
                best_match = {
                    'x': int(max_loc[0]),
                    'y': int(max_loc[1]),
                    'w': new_w,
                    'h': new_h,
                    'confidence': float(max_val),
                    'scale': float(scale)
                }
        
        if best_match and best_match['confidence'] >= threshold:
            # 转换坐标为模板原始尺寸（去除缩放因子，返回参考尺寸的坐标）
            original_x = int(best_match['x'] / best_match['scale'])
            original_y = int(best_match['y'] / best_match['scale'])
            logger.info(f"多尺度匹配成功: 尺度={best_match['scale']:.2f}, 置信度={best_match['confidence']:.3f}")
            return self._create_match_result(original_x, original_y, w, h, best_match['confidence'])
        
        logger.debug(f"多尺度匹配失败 (最佳: {best_confidence:.3f}, 阈值: {threshold})")
        return None

    def find_all_matches_multiscale(
        self,
        screenshot: np.ndarray,
        template_path: str,
        threshold: float = 0.65,
        scale_range: tuple = (0.5, 1.5),
        scale_steps: int = 10
    ) -> List[Dict]:
        """
        多尺度模板匹配 - 查找所有匹配

        Args:
            screenshot: 截图图像数组（BGR格式）
            template_path: 模板图片路径（支持中文）
            threshold: 匹配阈值（0.0-1.0），默认0.65
            scale_range: 缩放范围 (min_scale, max_scale)
            scale_steps: 缩放步数，默认10

        Returns:
            匹配结果列表，按置信度降序排列
        """
        template = self._load_template(template_path)
        if template is None:
            logger.error(f"加载模板失败: {template_path}")
            return []

        screenshot_gray = self._convert_to_grayscale(screenshot)
        template_gray = self._convert_to_grayscale(template)

        h, w = template_gray.shape[:2]
        
        min_scale, max_scale = scale_range
        scales = np.linspace(min_scale, max_scale, scale_steps)
        
        all_matches = []
        
        for scale in scales:
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            if new_w < 10 or new_h < 10:
                continue
            if new_w > screenshot_gray.shape[1] or new_h > screenshot_gray.shape[0]:
                continue
                
            scaled_template = cv2.resize(template_gray, (new_w, new_h))
            
            try:
                result = cv2.matchTemplate(screenshot_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
            except cv2.error:
                continue
                
            # 找出所有超过阈值的匹配位置
            locations = np.where(result >= threshold)
            
            for y, x in zip(locations[0], locations[1]):
                confidence = result[y, x]
                # 转换回原始尺寸坐标
                orig_x = int(x / scale)
                orig_y = int(y / scale)
                all_matches.append({
                    'x': orig_x,
                    'y': orig_y,
                    'w': w,
                    'h': h,
                    'confidence': float(confidence),
                    'scale': float(scale)
                })
        
        if not all_matches:
            logger.debug(f"多尺度匹配未找到任何结果")
            return []
        
        # 按置信度降序排列
        all_matches.sort(key=lambda m: m['confidence'], reverse=True)
        
        # 应用非极大值抑制
        filtered = self._non_maximum_suppression(all_matches)
        
        results = [
            self._create_match_result(m['x'], m['y'], m['w'], m['h'], m['confidence'])
            for m in filtered
        ]

        logger.info(f"多尺度找到 {len(results)} 个匹配")
        return results

    def find_template_adaptive(
        self,
        screenshot: np.ndarray,
        template_path: str,
        threshold: float = 0.6,
        screenshot_aspect_ratio: float = None,
        template_aspect_ratio: float = None
    ) -> Optional[Dict]:
        """
        自适应模板匹配 - 综合多种策略实现跨分辨率识别

        策略顺序：
        1. 根据纵横比调整的多尺度匹配
        2. ORB特征点匹配后备方案
        3. 极限缩放范围搜索

        Args:
            screenshot: 截图图像数组（BGR格式）
            template_path: 模板图片路径
            threshold: 匹配阈值，默认0.6
            screenshot_aspect_ratio: 截图纵横比（宽/高），默认自动检测
            template_aspect_ratio: 模板纵横比（宽/高），默认自动检测

        Returns:
            匹配成功返回字典 {center_x, center_y, confidence, bbox}
            匹配失败返回 None
        """
        template = self._load_template(template_path)
        if template is None:
            logger.error(f"加载模板失败: {template_path}")
            return None

        screenshot_gray = self._convert_to_grayscale(screenshot)
        template_gray = self._convert_to_grayscale(template)

        h_s, w_s = screenshot_gray.shape[:2]
        h_t, w_t = template_gray.shape[:2]

        # 自动检测纵横比
        if screenshot_aspect_ratio is None:
            screenshot_aspect_ratio = w_s / h_s
        if template_aspect_ratio is None:
            template_aspect_ratio = w_t / h_t

        # 计算分辨率缩放比例
        scale_from_template_to_screenshot = min(w_s / w_t, h_s / h_t)

        # 根据纵横比差异计算纵横比校正因子
        aspect_ratio_diff = screenshot_aspect_ratio / template_aspect_ratio
        aspect_correction = aspect_ratio_diff if aspect_ratio_diff > 1.0 else 1.0 / aspect_ratio_diff

        # 纵横比自适应：更智能的缩放范围
        # 考虑主分辨率差异 + 纵横比差异
        min_scale = max(0.3, scale_from_template_to_screenshot / aspect_correction * 0.8)
        max_scale = min(2.5, scale_from_template_to_screenshot * aspect_correction * 1.2)

        # 如果纵横比相近，使用更精确的范围
        if 0.9 < aspect_ratio_diff < 1.1:
            min_scale = max(0.5, scale_from_template_to_screenshot * 0.85)
            max_scale = min(1.8, scale_from_template_to_screenshot * 1.15)

        # 纵横比自适应多尺度匹配
        result = self._find_template_with_aspect_adaptation(
            screenshot_gray, template_gray,
            threshold, min_scale, max_scale, 40
        )
        if result:
            logger.info(f"纵横比自适应匹配成功: scale={result.get('scale', 1.0):.2f}, conf={result['confidence']:.3f}")
            return result

        # ORB特征匹配后备方案
        result = self._find_template_with_orb(
            screenshot_gray, template_gray, threshold
        )
        if result:
            logger.info(f"ORB特征匹配成功: conf={result['confidence']:.3f}")
            return result

        # 极限缩放范围搜索（针对极端分辨率）
        result = self._find_template_extreme_scale(
            screenshot_gray, template_gray, threshold,
            base_scale=scale_from_template_to_screenshot
        )
        if result:
            logger.info(f"极限缩放匹配成功: scale={result.get('scale', 1.0):.2f}, conf={result['confidence']:.3f}")
            return result

        logger.debug(f"自适应匹配全部失败 (模板: {template_path})")
        return None

    def _find_template_with_aspect_adaptation(
        self,
        screenshot_gray: np.ndarray,
        template_gray: np.ndarray,
        threshold: float,
        min_scale: float,
        max_scale: float,
        scale_steps: int
    ) -> Optional[Dict]:
        """纵横比自适应的多尺度匹配"""
        h_t, w_t = template_gray.shape[:2]
        scales = np.linspace(min_scale, max_scale, scale_steps)

        best_match = None
        best_confidence = threshold - 0.01

        for scale in scales:
            new_w = int(w_t * scale)
            new_h = int(h_t * scale)

            if new_w < 10 or new_h < 10:
                continue
            if new_w > screenshot_gray.shape[1] or new_h > screenshot_gray.shape[0]:
                continue

            scaled_template = cv2.resize(template_gray, (new_w, new_h))

            try:
                result = cv2.matchTemplate(screenshot_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
            except cv2.error:
                continue

            if max_val > best_confidence:
                best_confidence = max_val
                best_match = {
                    'x': int(max_loc[0]),
                    'y': int(max_loc[1]),
                    'w': new_w,
                    'h': new_h,
                    'confidence': float(max_val),
                    'scale': float(scale)
                }

        if best_match and best_match['confidence'] >= threshold:
            original_x = int(best_match['x'] / best_match['scale'])
            original_y = int(best_match['y'] / best_match['scale'])
            return self._create_match_result(original_x, original_y, w_t, h_t, best_match['confidence'])

        return None

    def _find_template_with_orb(
        self,
        screenshot_gray: np.ndarray,
        template_gray: np.ndarray,
        threshold: float = 0.6
    ) -> Optional[Dict]:
        """
        ORB特征点匹配 - 当模板匹配失败时的后备方案

        使用ORB算法检测特征点并匹配，适用于模板有独特纹理特征的情况
        """
        try:
            orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)

            # 检测特征点和计算描述符
            kp1, des1 = orb.detectAndCompute(template_gray, None)
            kp2, des2 = orb.detectAndCompute(screenshot_gray, None)

            if des1 is None or des2 is None or len(des1) < 3 or len(des2) < 3:
                return None

            # 使用BFMatcher进行匹配
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)

            # Lowe's ratio test 过滤优质匹配
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) < 4:
                logger.debug(f"ORB匹配点不足: {len(good_matches)}/4")
                return None

            # 获取匹配点的坐标
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # 使用RANSAC算法估计单应性矩阵
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is None:
                return None

            # 计算内点数量作为置信度
            inliers = mask.ravel().sum()
            confidence = min(inliers / len(good_matches), 1.0)

            if confidence < threshold:
                logger.debug(f"ORB匹配置信度不足: {confidence:.3f} < {threshold}")
                return None

            # 使用单应性矩阵变换模板的四个角点
            h_t, w_t = template_gray.shape[:2]
            template_corners = np.float32([
                [0, 0], [w_t, 0], [w_t, h_t], [0, h_t]
            ]).reshape(-1, 1, 2)

            transformed_corners = cv2.perspectiveTransform(template_corners, H)

            # 计算变换后的边界框
            x_coords = transformed_corners[:, 0, 0]
            y_coords = transformed_corners[:, 0, 1]
            x = int(np.min(x_coords))
            y = int(np.min(y_coords))
            w = int(np.max(x_coords) - np.min(x_coords))
            h = int(np.max(y_coords) - np.min(y_coords))

            return self._create_match_result(x, y, w, h, confidence)

        except Exception as e:
            logger.debug(f"ORB匹配异常: {e}")
            return None

    def _find_template_extreme_scale(
        self,
        screenshot_gray: np.ndarray,
        template_gray: np.ndarray,
        threshold: float,
        base_scale: float
    ) -> Optional[Dict]:
        """
        极限缩放范围搜索 - 针对极端分辨率差异

        在更宽广的范围内搜索，包括非常小和非常大的缩放
        """
        h_t, w_t = template_gray.shape[:2]

        # 生成两个阶段的缩放范围
        # 第一阶段：极小缩放
        small_scales = np.linspace(0.1, min(0.5, base_scale * 0.5), 20)
        # 第二阶段：极大缩放
        large_scales = np.linspace(max(1.5, base_scale * 1.5), 3.0, 20)

        all_scales = np.concatenate([small_scales, large_scales])

        best_match = None
        best_confidence = threshold - 0.01

        for scale in all_scales:
            new_w = int(w_t * scale)
            new_h = int(h_t * scale)

            if new_w < 5 or new_h < 5:
                continue
            if new_w > screenshot_gray.shape[1] * 1.5 or new_h > screenshot_gray.shape[0] * 1.5:
                continue
            if new_w > screenshot_gray.shape[1] or new_h > screenshot_gray.shape[0]:
                continue

            scaled_template = cv2.resize(template_gray, (new_w, new_h))

            try:
                result = cv2.matchTemplate(screenshot_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
            except cv2.error:
                continue

            if max_val > best_confidence:
                best_confidence = max_val
                best_match = {
                    'x': int(max_loc[0]),
                    'y': int(max_loc[1]),
                    'w': new_w,
                    'h': new_h,
                    'confidence': float(max_val),
                    'scale': float(scale)
                }

        if best_match and best_match['confidence'] >= threshold:
            original_x = int(best_match['x'] / best_match['scale'])
            original_y = int(best_match['y'] / best_match['scale'])
            return self._create_match_result(original_x, original_y, w_t, h_t, best_match['confidence'])

        return None

    def calibrate_scale_with_anchors(
        self,
        screenshot: np.ndarray,
        anchor_template_pairs: list
    ) -> float:
        """
        使用锚点模板对校准缩放比例

        Args:
            screenshot: 截图图像数组
            anchor_template_pairs: 锚点对列表 [(template_path, expected_x, expected_y), ...]

        Returns:
            最佳缩放比例，失败返回1.0
        """
        if not anchor_template_pairs:
            return 1.0

        screenshot_gray = self._convert_to_grayscale(screenshot)
        scales = []

        for template_path, exp_x, exp_y in anchor_template_pairs:
            template = self._load_template(template_path)
            if template is None:
                continue

            template_gray = self._convert_to_grayscale(template)
            h_t, w_t = template_gray.shape[:2]

            # 在期望位置附近搜索
            search_radius = 100
            x1 = max(0, exp_x - search_radius)
            y1 = max(0, exp_y - search_radius)
            x2 = min(screenshot_gray.shape[1], exp_x + search_radius)
            y2 = min(screenshot_gray.shape[0], exp_y + search_radius)

            search_region = screenshot_gray[y1:y2, x1:x2]

            if search_region.size == 0:
                continue

            # 多尺度搜索
            for scale in np.linspace(0.5, 1.5, 20):
                new_w = int(w_t * scale)
                new_h = int(h_t * scale)

                if new_w < 5 or new_h < 5:
                    continue
                if new_w > search_region.shape[1] or new_h > search_region.shape[0]:
                    continue

                scaled_template = cv2.resize(template_gray, (new_w, new_h))

                try:
                    result = cv2.matchTemplate(search_region, scaled_template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                except cv2.error:
                    continue

                if max_val > 0.7:
                    # 计算检测到的位置相对于搜索区域的位置
                    det_x = x1 + max_loc[0]
                    det_y = y1 + max_loc[1]

                    # 计算该位置对应的缩放比例
                    detected_scale_x = det_x / exp_x if exp_x > 0 else 1.0
                    detected_scale_y = det_y / exp_y if exp_y > 0 else 1.0

                    if 0.5 < detected_scale_x < 2.0 and 0.5 < detected_scale_y < 2.0:
                        scales.append((detected_scale_x + detected_scale_y) / 2)

        if scales:
            return np.median(scales)

        return 1.0
