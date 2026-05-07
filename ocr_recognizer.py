# -*- coding: utf-8 -*-
"""
OCR文本识别模块

功能：
- 文字识别（中英文支持）
- 查找指定文字
- 区域识别
- 数字识别

依赖：PaddleOCR
"""

import cv2
import numpy as np
import logging

# 配置日志
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# 尝试导入PaddleOCR
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
    logger.info("PaddleOCR导入成功")
except ImportError as e:
    PADDLEOCR_AVAILABLE = False
    logger.error(f"PaddleOCR导入失败: {e}")


def preprocess_image(image: np.ndarray,
                   scale: float = 2.0,
                   enhance_contrast: bool = True,
                   denoise: bool = True) -> np.ndarray:
    """
    图像预处理 - 提高OCR识别率

    Args:
        image: 输入图像
        scale: 缩放比例（2.0表示放大2倍）
        enhance_contrast: 是否增强对比度
        denoise: 是否去噪

    Returns:
        处理后的灰度图
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w > 0 and new_h > 0:
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 对比度增强（CLAHE）
    if enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # 去噪
    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # 锐化
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    gray = cv2.filter2D(gray, -1, kernel)

    return gray


class OCRRecognizer:
    """
    PaddleOCR文字识别器

    支持中英文识别，可进行全文识别或指定文字查找
    """

    def __init__(self, languages: list = ['ch_sim', 'en']):
        """
        初始化OCR识别器

        Args:
            languages: 识别语言列表，默认['ch_sim', 'en']（简体中文+英文）
        """
        self.reader = None
        self.languages = languages
        self.available = False

        if not PADDLEOCR_AVAILABLE:
            logger.error("PaddleOCR不可用！")
            return

        try:
            logger.info("初始化PaddleOCR（支持中文）...")
            self.reader = PaddleOCR(lang='ch')
            self.available = True
            logger.info("PaddleOCR初始化成功")
        except Exception as e:
            logger.error(f"PaddleOCR初始化失败: {e}")
            self.reader = None
            self.available = False

    def recognize_text(self,
                     image: np.ndarray,
                     detail: int = 1,
                     use_preprocessing: bool = False) -> list:
        """
        识别图像中的文字

        Args:
            image: 输入图像数组
            detail: 详细信息级别
            use_preprocessing: 是否使用图像预处理

        Returns:
            识别结果列表，每项包含:
            - text: 识别的文字
            - confidence: 置信度
            - bbox: 文字包围框
        """
        if not self.available or self.reader is None:
            logger.error("OCR识别器未初始化")
            return []

        try:
            processed_image = image

            # 执行OCR识别
            result = self.reader.ocr(processed_image)

            text_results = []

            if result and isinstance(result, list) and len(result) > 0:
                first_result = result[0]

                # 处理不同格式的返回结果
                if isinstance(first_result, dict):
                    # 字典格式
                    rec_texts = first_result.get('rec_texts', [])
                    rec_scores = first_result.get('rec_scores', [])
                    rec_polys = first_result.get('rec_polys', [])
                    dt_polys = first_result.get('dt_polys', [])

                    for i, text in enumerate(rec_texts):
                        confidence = rec_scores[i] if i < len(rec_scores) else 1.0

                        bbox = None
                        if i < len(rec_polys):
                            bbox = rec_polys[i].tolist() if hasattr(rec_polys[i], 'tolist') else rec_polys[i]
                        elif i < len(dt_polys):
                            bbox = dt_polys[i].tolist() if hasattr(dt_polys[i], 'tolist') else dt_polys[i]

                        if bbox:
                            text_results.append({
                                'text': text,
                                'confidence': float(confidence),
                                'bbox': bbox
                            })
                elif isinstance(first_result, list):
                    # 列表格式
                    for line in first_result:
                        if line and len(line) >= 2:
                            bbox = line[0]
                            text_info = line[1]
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                text = text_info[0]
                                confidence = text_info[1]

                                text_results.append({
                                    'text': text,
                                    'confidence': float(confidence),
                                    'bbox': bbox if isinstance(bbox, list) else bbox.tolist()
                                })

            logger.info(f"识别到 {len(text_results)} 个文字项")
            return text_results

        except Exception as e:
            logger.error(f"OCR识别出错: {e}")
            return []

    def recognize_text_with_confidence_check(self,
                                           image: np.ndarray,
                                           min_confidence: float = 0.7) -> list:
        """
        识别文字并过滤低置信度结果

        Args:
            image: 输入图像
            min_confidence: 最小置信度阈值

        Returns:
            过滤后的识别结果
        """
        results = self.recognize_text(image, use_preprocessing=True)

        filtered_results = [r for r in results if r['confidence'] >= min_confidence]

        logger.info(f"识别: {len(results)} 个结果, {len(filtered_results)} 个通过置信度过滤")

        return filtered_results

    def find_text(self,
                  image: np.ndarray,
                  target_text: str,
                  case_sensitive: bool = False) -> list:
        """
        查找包含指定文字的结果

        Args:
            image: 输入图像
            target_text: 要查找的文字
            case_sensitive: 是否区分大小写

        Returns:
            匹配的结果列表
        """
        results = self.recognize_text(image)

        if not case_sensitive:
            target_text = target_text.lower()

        matches = []
        for result in results:
            text = result['text']
            if not case_sensitive:
                text = text.lower()

            if target_text in text:
                matches.append(result)

        logger.info(f"查找 '{target_text}': 找到 {len(matches)} 个匹配")
        return matches

    def find_text_position(self,
                          image: np.ndarray,
                          target_text: str,
                          case_sensitive: bool = False) -> list:
        """
        查找文字的位置（包围框）

        Args:
            image: 输入图像
            target_text: 要查找的文字
            case_sensitive: 是否区分大小写

        Returns:
            位置列表 [(x1, y1, x2, y2), ...]
        """
        matches = self.find_text(image, target_text, case_sensitive)

        positions = []
        for match in matches:
            if match['bbox']:
                bbox = match['bbox']
                if len(bbox) == 4:
                    xs = [point[0] for point in bbox]
                    ys = [point[1] for point in bbox]
                    x1, y1 = int(min(xs)), int(min(ys))
                    x2, y2 = int(max(xs)), int(max(ys))
                    positions.append((x1, y1, x2, y2))

        return positions

    def get_text_center(self, bbox: list) -> tuple:
        """
        计算文字包围框的中心点

        Args:
            bbox: 包围框坐标

        Returns:
            中心点坐标 (center_x, center_y)
        """
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]
        center_x = int(sum(xs) / len(xs))
        center_y = int(sum(ys) / len(ys))
        return (center_x, center_y)

    def extract_region(self,
                     image: np.ndarray,
                     x: int,
                     y: int,
                     width: int,
                     height: int) -> np.ndarray:
        """
        裁剪图像区域

        Args:
            image: 输入图像
            x, y: 左上角坐标
            width, height: 宽高

        Returns:
            裁剪后的图像
        """
        return image[y:y+height, x:x+width]

    def recognize_in_region(self,
                          image: np.ndarray,
                          x: int,
                          y: int,
                          width: int,
                          height: int) -> list:
        """
        在指定区域识别文字

        Args:
            image: 输入图像
            x, y: 区域左上角坐标
            width, height: 区域宽高

        Returns:
            识别结果列表
        """
        region = self.extract_region(image, x, y, width, height)
        return self.recognize_text(region)

    def draw_text_boxes(self,
                       image: np.ndarray,
                       text_results: list,
                       color: tuple = (0, 255, 0),
                       thickness: int = 2) -> np.ndarray:
        """
        在图像上绘制文字包围框

        Args:
            image: 输入图像
            text_results: 文字识别结果
            color: 框颜色 (B, G, R)
            thickness: 线条粗细

        Returns:
            绘制了框的图像
        """
        result = image.copy()

        for item in text_results:
            if item['bbox']:
                bbox = item['bbox']
                pts = np.array(bbox, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(result, [pts], True, color, thickness)

                text = item['text']
                x, y = int(bbox[0][0]), int(bbox[0][1]) - 10
                cv2.putText(
                    result,
                    text,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )

        return result

    def get_all_text(self, image: np.ndarray) -> str:
        """
        获取图像中所有文字（拼接成字符串）

        Args:
            image: 输入图像

        Returns:
            所有文字，用换行符分隔
        """
        results = self.recognize_text(image)
        texts = [item['text'] for item in results]
        return '\n'.join(texts)

    def find_numbers(self, image: np.ndarray) -> list:
        """
        查找图像中的数字

        Args:
            image: 输入图像

        Returns:
            数字结果列表，每项包含text, value, confidence, bbox
        """
        results = self.recognize_text(image)
        number_results = []

        for result in results:
            text = result['text'].strip()
            try:
                # 尝试转换为数字
                num = float(text.replace(',', ''))
                number_results.append({
                    'text': text,
                    'value': num,
                    'confidence': result['confidence'],
                    'bbox': result['bbox']
                })
            except ValueError:
                continue

        logger.info(f"找到 {len(number_results)} 个数字")
        return number_results
