# -*- coding: utf-8 -*-
"""
屏幕截图处理模块

功能：
- 加载和保存图像
- 图像区域裁剪
- 像素颜色获取
- 颜色查找匹配

依赖：OpenCV, PIL, NumPy
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def imread_unicode(filepath: str) -> Optional[np.ndarray]:
    """
    支持中文路径的图像读取

    Args:
        filepath: 图片路径（支持中文）

    Returns:
        OpenCV图像数组（BGR格式），失败返回None
    """
    try:
        with open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return image
    except Exception as e:
        logger.error(f"读取图像失败 {filepath}: {e}")
        return None


class ScreenCapture:
    """
    屏幕截图处理类

    用于加载图像、区域裁剪、颜色查找等操作
    """

    def __init__(self):
        self.current_screenshot = None  # 当前截图
        self.screenshot_array = None    # NumPy数组格式

    def load_from_file(self, file_path: str) -> bool:
        """
        从文件加载截图

        Args:
            file_path: 图片路径（支持中文）

        Returns:
            加载是否成功
        """
        try:
            self.current_screenshot = imread_unicode(file_path)
            if self.current_screenshot is not None:
                self.screenshot_array = np.array(self.current_screenshot)
                logger.info(f"已加载截图: {file_path}")
                return True
            logger.error("截图加载失败")
            return False
        except Exception as e:
            logger.error(f"加载截图出错: {e}")
            return False

    def load_from_array(self, image_array: np.ndarray) -> bool:
        """
        从NumPy数组加载图像

        Args:
            image_array: OpenCV图像数组

        Returns:
            加载是否成功
        """
        try:
            self.current_screenshot = image_array
            self.screenshot_array = image_array
            return True
        except Exception as e:
            logger.error(f"从数组加载图像出错: {e}")
            return False

    def get_screenshot(self) -> Optional[np.ndarray]:
        """获取当前截图"""
        return self.current_screenshot

    def get_region(self, x: int, y: int, width: int, height: int) -> Optional[np.ndarray]:
        """
        裁剪图像区域

        Args:
            x, y: 裁剪区域左上角坐标
            width, height: 裁剪区域宽高

        Returns:
            裁剪后的图像数组，失败返回None
        """
        if self.current_screenshot is None:
            logger.error("无截图可裁剪")
            return None

        try:
            # 注意：OpenCV中 y 是行（高度），x 是列（宽度）
            region = self.current_screenshot[y:y+height, x:x+width]
            return region
        except Exception as e:
            logger.error(f"裁剪区域出错: {e}")
            return None

    def get_pixel_color(self, x: int, y: int) -> Optional[Tuple[int, int, int]]:
        """
        获取指定像素的RGB颜色

        Args:
            x, y: 像素坐标

        Returns:
            RGB元组 (R, G, B)，失败返回None
        """
        if self.current_screenshot is None:
            logger.error("无截图可查询")
            return None

        try:
            # OpenCV使用BGR格式，需要转换
            bgr = self.current_screenshot[y, x]
            rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
            return rgb
        except Exception as e:
            logger.error(f"获取像素颜色出错: {e}")
            return None

    def find_color(self,
                   target_color: Tuple[int, int, int],
                   tolerance: int = 10,
                   region: Optional[Tuple[int, int, int, int]] = None) -> List[Tuple[int, int]]:
        """
        在图像中查找指定颜色的所有位置

        Args:
            target_color: 目标RGB颜色 (R, G, B)
            tolerance: 颜色容差（0-255）
            region: 可选，搜索区域 (x, y, width, height)

        Returns:
            匹配的坐标列表 [(x1,y1), (x2,y2), ...]
        """
        if self.current_screenshot is None:
            logger.error("无截图可搜索")
            return []

        try:
            if region:
                x, y, w, h = region
                search_area = self.current_screenshot[y:y+h, x:x+w]
                offset_x, offset_y = x, y
            else:
                search_area = self.current_screenshot
                offset_x, offset_y = 0, 0

            # RGB转BGR（OpenCV格式）
            target_bgr = (target_color[2], target_color[1], target_color[0])

            # 计算颜色范围
            lower_bound = np.array([max(0, target_bgr[i] - tolerance) for i in range(3)])
            upper_bound = np.array([min(255, target_bgr[i] + tolerance) for i in range(3)])

            # 创建掩码
            mask = cv2.inRange(search_area, lower_bound, upper_bound)

            # 找到所有匹配点
            points = np.where(mask > 0)

            results = []
            for i in range(len(points[0])):
                # 转换回原图坐标
                px, py = points[1][i] + offset_x, points[0][i] + offset_y
                results.append((px, py))

            return results
        except Exception as e:
            logger.error(f"查找颜色出错: {e}")
            return []

    def save_screenshot(self, output_path: str) -> bool:
        """
        保存截图到文件

        Args:
            output_path: 保存路径

        Returns:
            保存是否成功
        """
        if self.current_screenshot is None:
            logger.error("无截图可保存")
            return False

        try:
            cv2.imwrite(output_path, self.current_screenshot)
            logger.info(f"截图已保存: {output_path}")
            return True
        except Exception as e:
            logger.error(f"保存截图出错: {e}")
            return False

    def resize_screenshot(self, scale: float) -> bool:
        """
        缩放截图

        Args:
            scale: 缩放比例（如0.5表示缩小一半，2表示放大两倍）

        Returns:
            缩放是否成功
        """
        if self.current_screenshot is None:
            logger.error("无截图可缩放")
            return False

        try:
            new_width = int(self.current_screenshot.shape[1] * scale)
            new_height = int(self.current_screenshot.shape[0] * scale)
            self.current_screenshot = cv2.resize(
                self.current_screenshot,
                (new_width, new_height),
                interpolation=cv2.INTER_LINEAR
            )
            self.screenshot_array = self.current_screenshot
            logger.info(f"截图已缩放: {new_width}x{new_height}")
            return True
        except Exception as e:
            logger.error(f"缩放截图出错: {e}")
            return False

    def convert_to_grayscale(self) -> bool:
        """
        转换为灰度图

        Returns:
            转换是否成功
        """
        if self.current_screenshot is None:
            logger.error("无截图可转换")
            return False

        try:
            self.current_screenshot = cv2.cvtColor(
                self.current_screenshot,
                cv2.COLOR_BGR2GRAY
            )
            self.screenshot_array = self.current_screenshot
            return True
        except Exception as e:
            logger.error(f"转换灰度图出错: {e}")
            return False
