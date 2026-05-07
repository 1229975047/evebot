# -*- coding: utf-8 -*-
"""
ADB控制器 - 安卓设备通信模块

功能：
- 通过ADB连接安卓设备（USB/无线）
- 屏幕截图、点击、滑动、输入文本
- 获取设备屏幕尺寸等信息

依赖：adb命令行工具（Android SDK自带）
"""

import subprocess
import time
import os
from typing import Optional, Tuple, List
import logging
import cv2
import numpy as np

# 配置日志
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class ADBController:
    """ADB设备控制器"""

    def __init__(self, device_id: Optional[str] = None, custom_adb_path: Optional[str] = None):
        """
        初始化ADB控制器

        Args:
            device_id: 设备ID（可选，用于多设备时指定设备）
            custom_adb_path: 自定义adb路径（可选）
        """
        self.device_id = device_id

        if custom_adb_path:
            self.adb_path = custom_adb_path
        else:
            self.adb_path = self._find_adb()

        if self.adb_path:
            logger.info(f"ADB已找到: {self.adb_path}")
        else:
            logger.warning("ADB未找到常见位置")

    def _find_adb(self) -> Optional[str]:
        """查找系统中的adb工具路径"""
        # 常见ADB路径（Windows系统）
        common_paths = [
            os.path.expanduser("~\\AppData\\Local\\Android\\Sdk\\platform-tools\\adb.exe"),
            os.path.expanduser("~\\Android\\Sdk\\platform-tools\\adb.exe"),
            "C:\\Android\\platform-tools\\adb.exe",
            "D:\\Android\\platform-tools\\adb.exe",
            "E:\\Android\\platform-tools\\adb.exe",
        ]

        # 遍历常见路径
        for path in common_paths:
            if os.path.exists(path):
                logger.info(f"ADB已找到: {path}")
                return path

        # 尝试从PATH中查找
        try:
            result = subprocess.run(
                ["where", "adb"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                adb_path = result.stdout.strip().split('\n')[0]
                logger.info(f"ADB已在PATH中: {adb_path}")
                return adb_path
        except:
            pass

        return None

    def _execute_command(self, command: List[str]) -> str:
        """
        执行ADB命令

        Args:
            command: ADB命令参数列表

        Returns:
            命令输出字符串
        """
        try:
            if self.device_id:
                cmd = [self.adb_path, "-s", self.device_id] + command
            else:
                cmd = [self.adb_path] + command

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            logger.error("ADB命令执行超时")
            raise
        except Exception as e:
            logger.error(f"ADB命令执行失败: {e}")
            raise

    def connect(self, ip: str, port: int = 5555) -> bool:
        """
        无线连接到设备

        Args:
            ip: 设备IP地址
            port: ADB端口（默认5555）

        Returns:
            连接是否成功
        """
        try:
            result = self._execute_command(["connect", f"{ip}:{port}"])
            return "connected" in result.lower() or "already connected" in result.lower()
        except Exception as e:
            logger.error(f"连接设备失败: {e}")
            return False

    def disconnect(self, ip: str, port: int = 5555) -> bool:
        """断开无线连接"""
        try:
            self._execute_command(["disconnect", f"{ip}:{port}"])
            return True
        except Exception as e:
            logger.error(f"断开连接失败: {e}")
            return False

    def get_devices(self) -> List[Tuple[str, str]]:
        """
        获取已连接的设备列表

        Returns:
            设备列表，每项为(device_id, status)元组
        """
        try:
            output = self._execute_command(["devices", "-l"])
            devices = []
            for line in output.split('\n')[1:]:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        device_id = parts[0]
                        status = parts[1]
                        devices.append((device_id, status))
            return devices
        except Exception as e:
            logger.error(f"获取设备列表失败: {e}")
            return []

    def screenshot(self, save_path: str = "screenshot.png") -> Optional[str]:
        """
        截图并保存到文件（传统方式，先pull再保存）

        Args:
            save_path: 保存路径

        Returns:
            保存路径，失败返回None
        """
        try:
            self._execute_command(["shell", "screencap", "-p", "/sdcard/screenshot.png"])
            self._execute_command(["pull", "/sdcard/screenshot.png", save_path])
            return save_path
        except Exception as e:
            logger.error(f"截图失败: {e}")
            return None

    def tap(self, x: int, y: int) -> bool:
        """
        点击屏幕坐标

        Args:
            x: X坐标
            y: Y坐标

        Returns:
            是否成功
        """
        try:
            self._execute_command(["shell", "input", "tap", str(x), str(y)])
            logger.info(f"点击坐标: ({x}, {y})")
            return True
        except Exception as e:
            logger.error(f"点击失败: {e}")
            return False

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: int = 300) -> bool:
        """
        滑动屏幕

        Args:
            x1, y1: 起点坐标
            x2, y2: 终点坐标
            duration: 持续时间（毫秒）

        Returns:
            是否成功
        """
        try:
            self._execute_command([
                "shell", "input", "swipe",
                str(x1), str(y1), str(x2), str(y2), str(duration)
            ])
            logger.info(f"滑动: ({x1}, {y1}) -> ({x2}, {y2})")
            return True
        except Exception as e:
            logger.error(f"滑动失败: {e}")
            return False

    def input_text(self, text: str) -> bool:
        """输入文本"""
        try:
            self._execute_command(["shell", "input", "text", text])
            logger.info(f"输入文本: {text}")
            return True
        except Exception as e:
            logger.error(f"输入文本失败: {e}")
            return False

    def press_key(self, keycode: int) -> bool:
        """按下按键（keycode为Android按键码）"""
        try:
            self._execute_command(["shell", "input", "keyevent", str(keycode)])
            logger.info(f"按键: {keycode}")
            return True
        except Exception as e:
            logger.error(f"按键失败: {e}")
            return False

    def get_screen_size(self) -> Optional[Tuple[int, int]]:
        """获取物理屏幕尺寸"""
        try:
            output = self._execute_command(["shell", "wm", "size"])
            if "Physical size:" in output:
                size_str = output.split("Physical size:")[1].strip()
                width, height = map(int, size_str.split('x'))
                return (width, height)
            return None
        except Exception as e:
            logger.error(f"获取屏幕尺寸失败: {e}")
            return None

    def get_display_size(self) -> Optional[Tuple[int, int]]:
        """获取实际显示尺寸（可能因旋转与物理尺寸不同）"""
        try:
            output = self._execute_command(["shell", "dumpsys", "window", "displays"])
            # 从输出中解析 "cur=3200x2136"
            if "cur=" in output:
                import re
                match = re.search(r'cur=(\d+)x(\d+)', output)
                if match:
                    width, height = int(match.group(1)), int(match.group(2))
                    return (width, height)
            # 备用方案：使用wm size
            return self.get_screen_size()
        except Exception as e:
            logger.error(f"获取显示尺寸失败: {e}")
            return None

    def screenshot_fast(self) -> Optional[np.ndarray]:
        """
        快速截图（使用exec-out直接获取图像数据）

        Returns:
            numpy数组（BGR格式），失败返回None
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                import subprocess
                cmd = [self.adb_path]
                if self.device_id:
                    cmd += ["-s", self.device_id]
                cmd += ["exec-out", "screencap", "-p"]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=15
                )
                if result.returncode == 0 and result.stdout:
                    # 将原始数据解码为图像
                    data = np.frombuffer(result.stdout, dtype=np.uint8)
                    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    return image
                if attempt < max_retries - 1:
                    time.sleep(0.5)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
                logger.error(f"快速截图失败: {e}")

        # fallback: 使用传统方式截图再读取
        try:
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
            self._execute_command(["shell", "screencap", "-p", "/sdcard/screenshot.png"])
            self._execute_command(["pull", "/sdcard/screenshot.png", tmp_path])
            img = cv2.imread(tmp_path)
            try:
                os.unlink(tmp_path)
            except:
                pass
            if img is not None:
                logger.info(f"截图成功（备用方式）")
            return img
        except Exception as e2:
            logger.error(f"备用截图也失败: {e2}")
            return None

    def screenshot_to_file(self, save_path: str = "screenshot.png") -> Optional[str]:
        """
        截图并保存到文件（使用快速截图方式）

        Args:
            save_path: 保存路径

        Returns:
            保存路径，失败返回None
        """
        image = self.screenshot_fast()
        if image is not None:
            cv2.imwrite(save_path, image)
            return save_path
        return None
