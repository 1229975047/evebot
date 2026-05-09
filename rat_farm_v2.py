# -*- coding: utf-8 -*-
"""EVE Bot 刷怪脚本 v4

功能：
1. 检测离站按钮判断在站内还是站外
2. 站内：点击离站，等待25s，检查站按钮还存在？存在重复点击，没有则到太空
3. 点击侧边界面张开/隐藏
4. 匹配异常空间模板（优先：大型 > 中型 > 小型），点击中心，等待3s
5. 匹配跃迁模板，点击，等待30s
6. 匹配惯性稳定器，点击一次，等待30s
7. 循环检查一键锁定：
   - 有就点击，间隔最短10s
   - 点击过一次一键锁定后要点击：母舰防御模块、综合火控系统、旗舰掠能器
   - 这些按钮每次点击跃迁后只重复点击一次，直到第二次点击跃迁后重置状态
8. 循环匹配敌对目标和一键锁定，两者有一个匹配到就一直循环点击一键锁定
9. 直到两个模板30s内都匹配不到，才又开始匹配异常
"""

import time
import os
import sys
import random
import warnings
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, List, Dict

# 过滤libpng警告（cv2.imread读取PNG时的iCCP警告，不影响功能）
warnings.filterwarnings('ignore', message='.*iCCP.*')
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'

sys.path.insert(0, os.path.dirname(__file__))
from adb_controller import ADBController
from screen_capture import ScreenCapture
from template_matcher import TemplateMatcher, imread_unicode
# from ocr_recognizer import OCRRecognizer  # 暂时屏蔽OCR


class RatFarmV2:
    """刷怪自动化脚本 v4 - 基于模板匹配"""

    # 参考分辨率（模板匹配使用的分辨率）
    REFERENCE_WIDTH = 3200
    REFERENCE_HEIGHT = 2136

    # 模板阈值
    TEMPLATE_THRESHOLD = 0.65
    ENEMY_TEMPLATE_THRESHOLD = 0.75  # 敌对目标检测置信度阈值

    def __init__(self, device_id=None):
        self.adb = ADBController(device_id)
        self.screen = ScreenCapture()
        self.matcher = TemplateMatcher()
        self._ocr = None  # 懒加载OCR

        # 截图保存目录
        self.screenshot_dir = os.path.join(os.path.dirname(__file__), "screenshots")
        os.makedirs(self.screenshot_dir, exist_ok=True)

        # 检测实际分辨率
        self.actual_width, self.actual_height = self.detect_resolution()
        self.scale_x = self.actual_width / self.REFERENCE_WIDTH
        self.scale_y = self.actual_height / self.REFERENCE_HEIGHT
        print(f"\n[分辨率] 参考: {self.REFERENCE_WIDTH}x{self.REFERENCE_HEIGHT}, 实际: {self.actual_width}x{self.actual_height}, 缩放: ({self.scale_x:.3f}, {self.scale_y:.3f})")

        # 模板路径映射
        self.templates = {
            # 离站检测
            'undock_btn': 'mods/undock_btn.png',

            # 侧边栏
            'sidebar_toggle': 'mods/sidebar_toggle.png',

            # 异常空间（优先级从高到低）
            'anomaly_angel_large': 'mods/anomaly_angel_large.png',
            'anomaly_angel_medium': 'mods/anomaly_angel_medium.png',
            'anomaly_angel_inspector': 'mods/anomaly_angel_inspector.png',
            'anomaly_angel_small': 'mods/anomaly_angel_small.png',

            # 检察官异常特殊按钮
            'accelerate_track_side_icon': 'mods/accelerate_track_side_icon.png',  # 加速轨道侧边图标
            'accelerate_track_icon': 'mods/accelerate_track_icon.png',
            'activate_btn': 'mods/activate_btn.png',
            'anomaly_filter_btn': 'mods/anomaly_filter_btn.png',
            'concentrated_fire': 'mods/concentrated_fire.png',  # 集中火力

            # 跃迁
            'warp_btn': 'mods/warp_btn.png',

            # 装备
            'inertia': 'mods/equip_inertia.png',
            'one_key_lock': 'mods/one_key_lock.png',
            'defense_module': 'mods/equip_defense_module.png',
            'fire_control': 'mods/equip_fire_control.png',
            'cap_energy': 'mods/equip_cap_Energy.png',
            'salvager': 'mods/equip_salvager.png',  # 打捞器

            # 敌对目标（新标注）
            'enemy_1': 'mods/new_enemies/enemy_1.png',  # 挑战隐匿
            'enemy_2': 'mods/new_enemies/enemy_2.png',  # 暴风级守卫
            'enemy_3': 'mods/new_enemies/enemy_3.png',  # 狂暴级战列舰
            'enemy_4': 'mods/new_enemies/enemy_4.png',  # 猎犬级护卫
            'enemy_5': 'mods/new_enemies/enemy_5.png',  # 探索级护卫
            'enemy_cruiser': 'mods/new_enemies/enemy_cruiser.png',    # 龙卷风级巡洋
            'enemy_cruiser2': 'mods/new_enemies/enemy_cruiser2.png',  # 龙卷风级站巡
            'enemy_battleship': 'mods/new_enemies/enemy_battleship.png',  # 台风级战列舰
            'enemy_assassin': 'mods/new_enemies/enemy_assassin.png',  # 刺客级巡洋
            'enemy_shark': 'mods/new_enemies/enemy_shark.png',        # 长尾鲛级护卫
            'enemy_vortex': 'mods/new_enemies/enemy_vortex.png',      # 死亡漩涡级战列舰
            'enemy_sinabo': 'mods/new_enemies/enemy_sinabo.png',      # 赛那波级巡洋
            'enemy_delamir': 'mods/new_enemies/enemy_delamir.png',    # 德拉米尔级护卫
            'enemy_player_tag': 'mods/enemy_player_tag.png',
            'local_player_detect': 'mods/local_player_detect.png',

            # 工业船（战斗检测）
            'industrial_nerus': 'mods/new_enemies/industrial_nerus.png',      # 工业级涅鲁斯
            'industrial_zuotouyu': 'mods/new_enemies/industrial_zuotouyu.png', # 工业级座头鲸
            'industrial_imika': 'mods/new_enemies/industrial_imika.png',       # 工业级伊米卡

            # 存点相关
            'save_point_00': 'mods/save_point_00.png',
            'set_destination': 'mods/set_destination.png',
            'navigate_to_destination': 'mods/navigate_to_destination.png',

            # 装备
            'armor_repair': 'mods/equip_armor_repair.png',

            # 装备状态
            'equip_status_bar': 'mods/equip_status_bar.png',

            # 状态模板
            'status_loss': 'mods/status_loss.png',
            'status_health': 'mods/status_health.png',
        }

        # 敌对目标检测区域 (x1, y1, x2, y2)
        self.enemy_detection_region = (649, 20, 2395, 247)

        # 状态检测区域 (x1, y1, x2, y2) - 检测损失/健康状态
        self.status_detection_region = (1572, 1731, 1616, 1762)  # 装甲维修状态检测区域
        self.status_check_interval = 30  # 状态检测间隔（秒）

        # 检察官异常特殊区域 (x1, y1, x2, y2) - 加速轨道和异常选项按钮
        self.inspector_special_region = (3050, 126, 3197, 1406)
        self.inspector_activate_wait = 45  # 激活后等待秒数
        self.inspector_defense_interval = 180  # 防御模块点击间隔（秒）

        # 危险颜色检测区域 (x1, y1, x2, y2)
        self.danger_color_region = (59, 1875, 397, 1986)  # 敌对玩家检测区域
        self.danger_color = (153, 32, 20)  # RGB红色
        self.danger_color_tolerance = 20

        self.enemy_tag_threshold = 0.9  # 敌对标签置信度阈值

        # 装备验证参数
        self.equip_verify_wait = 3  # 点击后等待验证秒数(综合火控3秒)
        self.equip_verify_radius = 200  # 验证范围半径(px)
        self.equip_success_confidence = 0.7  # 验证成功置信度阈值

        # 验证模板存在
        for _, path in list(self.templates.items()):
            if not os.path.exists(path):
                print(f"  警告: 模板不存在 {path}")

        # 配置参数
        self.undock_wait = 25  # 离站等待
        self.anomaly_click_wait = 3  # 点击异常后等待
        self.warp_wait = 44  # 跃迁后等待
        self.one_key_lock_interval = 10  # 一键锁定最小间隔
        self.no_target_timeout = 15  # 没有目标超时（15秒内没有目标则退出）

        # 装备点击间隔（压缩时间）
        self.equip_click_interval = 3  # 装备点击间隔秒数

        # 装备独立计时配置
        self.armor_repair_first_click = 5  # 旗舰装甲维修器首次点击
        self.armor_repair_second_click = 5  # 旗舰装甲维修器二次点击

        # 综合火控系统 - 点击间隔
        self.fire_control_first_click = 5
        self.fire_control_second_click = 5

        # 旗舰掠能器 - 点击间隔
        self.cap_energy_first_click = 5
        self.cap_energy_second_click = 5

        # 惯性稳定器 - 跃迁落地后首次点击时间
        self.inertia_first_click = 5
        self.inertia_second_click = 0  # 不需要二次点击

        # 母舰防御模块 - 点击间隔
        self.defense_first_click = 5
        self.defense_second_click = 60

        # 状态
        self.last_one_key_lock_time = 0  # 上次点击一键锁定的时间
        self.warp_cycle_counter = 0  # 跃迁周期计数器
        self.armor_repair_start_time = 0  # 旗舰装甲维修器开启时间
        self.defense_module_last_click_time = 0  # 防御模块上次点击时间
        self.defense_module_activated = False  # 防御模块是否已激活（True=激活中，False=关闭）
        self.defense_module_deactivate_cooldown = 0  # 关闭后60秒禁止点击的时间戳

        # 异常空间追踪（避免重复选择同一位置）
        self.last_anomaly_type = None  # 上次选择的异常类型
        self.last_anomaly_positions = []  # 本次跃迁选择的异常位置（用于失败重试）
        self.all_anomaly_positions = []  # 所有历史异常位置（跨跃迁持久化）
        self.anomaly_position_threshold = 100  # 位置重复判定阈值(px)
        self.inspector_mode = False  # 是否为检察官异常特殊模式
        self._cached_screenshot = None  # 内存缓存的截图（用于加速多次匹配）

    @property
    def ocr(self):
        """懒加载OCR识别器（暂时屏蔽）"""
        if self._ocr is None:
            # 创建一个模拟OCR对象，不实际加载PaddleOCR
            class MockOCR:
                available = False
                def recognize_text(self, *args, **kwargs):
                    return []
            self._ocr = MockOCR()
        return self._ocr

    def ts(self) -> str:
        """返回当前时间戳字符串"""
        return datetime.now().strftime("%H:%M:%S")

    def detect_resolution(self) -> Tuple[int, int]:
        """检测设备实际分辨率"""
        # 截取一张截图获取分辨率信息
        screenshot_path = self.adb.screenshot_to_file(os.path.join(self.screenshot_dir, "_res_check.png"))
        if screenshot_path:
            import cv2
            # 临时抑制libpng警告
            stderr_fd = sys.stderr.fileno()
            devnull = os.open(os.devnull, os.O_WRONLY)
            old_stderr = os.dup(stderr_fd)
            os.dup2(devnull, stderr_fd)
            try:
                img = cv2.imread(screenshot_path)
            finally:
                os.dup2(old_stderr, stderr_fd)
                os.close(devnull)
            if img is not None:
                h, w = img.shape[:2]
                os.remove(screenshot_path)
                return w, h
        # 默认返回参考分辨率
        return self.REFERENCE_WIDTH, self.REFERENCE_HEIGHT

    def scale_coords(self, x: int, y: int) -> Tuple[int, int]:
        """将参考坐标转换为实际屏幕坐标"""
        scaled_x = int(x * self.scale_x)
        scaled_y = int(y * self.scale_y)
        return scaled_x, scaled_y

    def random_wait(self, base_seconds: float, variance: float = 0.3) -> float:
        """
        随机等待时间，模拟人类操作的不规律性

        Args:
            base_seconds: 基础等待秒数
            variance: 随机波动范围（正负），默认0.3秒

        Returns:
            随机等待秒数，范围 [base - variance, base + variance]
        """
        import random
        return base_seconds + random.uniform(-variance, variance)

    def detect_ring_progress(self, screenshot: np.ndarray, center_x: int, center_y: int,
                             outer_r_frac: float = 0.45, inner_r_frac: float = 0.28) -> float:
        """
        检测装备外圈环形进度条进度（角度法：12点位置顺时针到24点为一周）。

        Args:
            screenshot: 完整截图
            center_x, center_y: 装备图标中心点（已缩放坐标）
            outer_r_frac: 外环半径（相对图标区域比例，默认0.45）
            inner_r_frac: 内环半径（默认0.28）

        Returns:
            进度 0.0（空）~ 1.0（满），检测失败返回 -1
        """
        import numpy as np
        h, w = screenshot.shape[:2]

        size = int(min(h, w) * outer_r_frac * 2.2)
        size = max(size, 20)
        x1 = max(0, int(center_x) - size // 2)
        y1 = max(0, int(center_y) - size // 2)
        x2 = min(w, int(center_x) + size // 2)
        y2 = min(h, int(center_y) + size // 2)
        region = screenshot[y1:y2, x1:x2]

        if region.size == 0:
            return -1.0

        rh, rw = region.shape[:2]
        cx, cy = rw // 2, rh // 2
        outer_r = int(min(rh, rw) * outer_r_frac)
        inner_r = int(min(rh, rw) * inner_r_frac)

        if outer_r <= inner_r or outer_r < 2:
            return -1.0

        y_idx, x_idx = np.ogrid[:rh, :rw]
        dist = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
        ring_mask = (dist <= outer_r) & (dist >= inner_r)
        ring_area = np.sum(ring_mask)

        if ring_area == 0:
            return -1.0

        # 计算每个像素相对于12点钟方向的顺时针角度（0~2π）
        # arctan2(dy, dx): 12点(dy>0,dx=0)=π/2, 3点(dy=0,dx>0)=0, 6点(dy<0,dx=0)=-π/2, 9点(dy=0,dx<0)=π
        # 转换为从12点顺时针：angle = (angle_raw - π/2) % 2π
        #   12点: (π/2 - π/2) % 2π = 0
        #   3点:  (0 - π/2) % 2π = 3π/2 (但顺时针3点=π/2，需+π修正)
        #   6点:  (-π/2 - π/2) % 2π = π
        # 实际验证用+π修正：+π后3点=π/2, 6点=π, 9点=3π/2
        dx = x_idx - cx
        dy = cy - y_idx  # 反转y轴，使上方为正（12点钟方向）
        angle_raw = np.arctan2(dy, dx)  # -pi to pi
        angle = (angle_raw + np.pi) % (2 * np.pi)  # 0 at 12 o'clock, clockwise

        # RGB (242, 255, 255) 相近颜色检测（装备激活进度）
        # OpenCV BGR: (242, 255, 255)，允许偏差±13
        bgr = region.astype(np.float32)
        target = np.array([242.0, 255.0, 255.0], dtype=np.float32)  # BGR顺序
        diff = np.abs(bgr - target)
        # B±13, G±13, R±13
        color_mask = (diff[:, :, 0] <= 13) & (diff[:, :, 1] <= 13) & (diff[:, :, 2] <= 13)
        filled_mask = ring_mask & color_mask

        # 统计填充像素的角度分布
        filled_pixels_angle = angle[filled_mask]
        if filled_pixels_angle.size == 0:
            return 0.0

        # 找出最远的填充角度（沿顺时针方向最远的那个点）
        # 由于是角度在环形上，最远角度应该是最大角度值（填充从0开始顺时针）
        progress_angle = np.max(filled_pixels_angle)
        progress = progress_angle / (2 * np.pi)

        return float(np.clip(progress, 0.0, 1.0))

    def scale_coords_from_list(self, coords: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """将参考坐标区域 (x1, y1, x2, y2) 转换为实际屏幕坐标"""
        x1, y1, x2, y2 = coords
        sx1, sy1 = self.scale_coords(x1, y1)
        sx2, sy2 = self.scale_coords(x2, y2)
        return (sx1, sy1, sx2, sy2)

    def get_scaled_region(self, region: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """获取缩放后的区域坐标"""
        return self.scale_coords_from_list(region)

    def get_screenshot(self) -> Optional[str]:
        """获取当前截图（保存到文件，仅用于调试）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(self.screenshot_dir, f"rat_v2_{timestamp}.png")

        if self.adb.screenshot_to_file(screenshot_path):
            return screenshot_path
        return None

    def get_screenshot_array(self) -> Optional[np.ndarray]:
        """
        获取当前截图（内存模式，直接返回numpy数组）

        性能优化：避免磁盘IO，使用ADB exec-out直接获取图像数据
        """
        return self.adb.screenshot_fast()

    def capture_and_cache(self) -> Optional[np.ndarray]:
        """
        截取新截图并缓存到内存

        用于在需要"新鲜"截图时调用，之后的多次模板匹配会复用缓存的截图
        """
        self._cached_screenshot = self.get_screenshot_array()
        return self._cached_screenshot

    def load_screenshot(self, path: str = None) -> Optional[np.ndarray]:
        """
        加载截图

        优先使用缓存的截图以提高性能，如果指定了path则从文件读取
        """
        if path is None:
            # 返回缓存的截图（由capture_and_cache或之前的get_screenshot_array填充）
            if hasattr(self, '_cached_screenshot') and self._cached_screenshot is not None:
                return self._cached_screenshot
            # 如果没有缓存，截取新图
            return self.capture_and_cache()
        return imread_unicode(path)

    def tap(self, x: int, y: int):
        """点击"""
        self.adb.tap(x, y)
        # 点击后清除缓存，确保下次获取新截图
        self._cached_screenshot = None

    def tap_with_offset(self, x: int, y: int, offset: int = 10):
        """点击（带随机偏移）"""
        import random
        dx = random.randint(-offset, offset)
        dy = random.randint(-offset, offset)
        tx, ty = x + dx, y + dy
        self.adb.tap(tx, ty)
        # 点击后清除缓存，确保下次获取新截图
        self._cached_screenshot = None

    def swipe_with_human(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500):
        """模拟人工滑动（带随机偏移和时长）"""
        import random
        # 起点终点各加随机偏移
        sx = x1 + random.randint(-10, 10)
        sy = y1 + random.randint(-10, 10)
        ex = x2 + random.randint(-10, 10)
        ey = y2 + random.randint(-10, 10)
        # 时长随机±20%
        dur = int(duration_ms * random.uniform(0.8, 1.2))
        self.adb.swipe(sx, sy, ex, ey, dur)
        self._cached_screenshot = None

    def click_defense_module(self, cx: int, cy: int) -> bool:
        """点击防御模块（统一状态管理）

        规则：
        - 激活状态点击 → 关闭 + 60秒冷却
        - 关闭状态点击 → 激活 + 172秒自动关闭
        - 冷却中不能点击

        Returns:
            True if clicked, False if skipped (cooldown or other reason)
        """
        current_time = time.time()

        # 检查冷却中不能点击
        if current_time < self.defense_module_deactivate_cooldown:
            remaining = self.defense_module_deactivate_cooldown - current_time
            print(f"[防御模块] 冷却中，{remaining:.0f}s后可再次点击")
            return False

        if self.defense_module_activated:
            # 当前是激活状态 → 点击关闭
            self.tap_with_offset(cx, cy, offset=10)
            self.defense_module_activated = False
            self.defense_module_deactivate_cooldown = current_time + 60  # 60秒冷却
            self.defense_module_last_click_time = 0  # 关闭时清零计时
            print(f"[防御模块] 已关闭，60秒冷却")
            return True
        else:
            # 当前是关闭状态 → 点击激活
            self.tap_with_offset(cx, cy, offset=10)
            self.defense_module_activated = True
            self.defense_module_last_click_time = current_time
            print(f"[防御模块] 已激活，172秒后自动关闭")
            return True

    def find_template(self, name: str, screenshot=None) -> Optional[Dict]:
        """查找单个模板"""
        if name not in self.templates:
            return None

        if screenshot is None:
            screenshot = self.load_screenshot()
        if screenshot is None:
            return None

        path = self.templates[name]
        if not os.path.exists(path):
            return None

        result = self.matcher.find_template(screenshot, path, threshold=self.TEMPLATE_THRESHOLD)
        return result

    def find_all_template(self, name: str, screenshot=None) -> List[Dict]:
        """查找所有匹配的模板"""
        if name not in self.templates:
            return []

        if screenshot is None:
            screenshot = self.load_screenshot()
        if screenshot is None:
            return []

        path = self.templates[name]
        if not os.path.exists(path):
            return []

        return self.matcher.find_all_matches(screenshot, path, threshold=self.TEMPLATE_THRESHOLD)

    def is_in_station(self, screenshot=None) -> bool:
        """检测是否在空间站内"""
        result = self.find_template('undock_btn', screenshot)
        if result:
            print(f"    [站内检测] 离站按钮找到 at ({result['center_x']}, {result['center_y']}), 置信度: {result['confidence']:.2f}")
            return True
        print(f"    [站内检测] 离站按钮未找到")
        return False

    def step_undock(self) -> bool:
        """离开空间站"""
        print("\n" + "=" * 50)
        print("步骤: 离开空间站")
        print("=" * 50)

        screenshot = self.load_screenshot()

        # 第一次点击离站
        result = self.find_template('undock_btn', screenshot)
        if result:
            self.tap_with_offset(result['center_x'], result['center_y'], offset=10)

            # 离站后29秒点击指定坐标
            wait_before_click = 29
            remaining_wait = self.undock_wait - wait_before_click
            print(f"    等待 {wait_before_click} 秒后点击指定坐标...")
            time.sleep(wait_before_click)

            # 点击指定坐标 (1495, 1927) + 随机偏移
            import random
            offset_x = random.randint(-15, 15)
            offset_y = random.randint(-15, 15)
            click_x = 1495 + offset_x
            click_y = 1927 + offset_y
            self.tap(click_x, click_y)
            print(f"    点击指定坐标 ({click_x}, {click_y}) [+随机偏移({offset_x},{offset_y})]")

            # 继续等待剩余时间
            print(f"    继续等待 {remaining_wait} 秒...")
            time.sleep(remaining_wait)

            # 再次检查离站按钮是否还存在
            screenshot2 = self.load_screenshot()
            result2 = self.find_template('undock_btn', screenshot2)
            if result2:
                print(f"    离站按钮仍存在，再次点击")
                self.tap_with_offset(result2['center_x'], result2['center_y'], offset=10)
                print(f"    再等待 {self.undock_wait} 秒...")
                time.sleep(self.undock_wait)

                # 第三次检查
                screenshot3 = self.load_screenshot()
                if self.find_template('undock_btn', screenshot3):
                    print(f"    警告: 离站按钮第三次存在，可能已在太空或卡住")
                    return True
            return True
        else:
            print(f"    不在站内，无需离站")
            return True

    def step_toggle_sidebar(self):
        """点击侧边界面张开/隐藏，并在指定区域点击异常图标筛选"""
        print("\n" + "=" * 50)
        print("步骤: 点击侧边界面开关")
        print("=" * 50)

        result = self.find_template('sidebar_toggle')
        if result:
            cx, cy = result['center_x'], result['center_y']

            # 参考坐标位置
            skip_ref_x, skip_ref_y = 2300, 1342
            click_ref_x, click_ref_y = 3117, 1350
            skip_ref_x, skip_ref_y = self.scale_coords(skip_ref_x, skip_ref_y)
            click_ref_x, click_ref_y = self.scale_coords(click_ref_x, click_ref_y)

            # 位置判断：不点击区域
            if abs(cx - skip_ref_x) <= 250 and abs(cy - skip_ref_y) <= 250:
                print(f"    [跳过] 侧边按钮在 ({skip_ref_x}, {skip_ref_y}) ± 250 范围内，不点击")
                return

            # 位置判断：点击区域
            if abs(cx - click_ref_x) <= 200 and abs(cy - click_ref_y) <= 200:
                print(f"    [点击] 侧边按钮在 ({click_ref_x}, {click_ref_y}) ± 200 范围内，点击 ({cx}, {cy})")
                self.tap_with_offset(cx, cy, offset=10)
                time.sleep(1)
            else:
                print(f"    [跳过] 侧边按钮在 ({click_ref_x}, {click_ref_y}) ± 200 范围外 ({cx}, {cy})，不点击")
        else:
            print(f"    侧边按钮未找到，跳过")

        # 在检察官异常特殊区域内点击异常图标筛选
        print("\n" + "=" * 50)
        print("步骤: 在异常区域点击异常图标筛选")
        print("=" * 50)

        x1, y1, x2, y2 = self.inspector_special_region
        screenshot = self.capture_and_cache()
        if screenshot is not None:
            region_screenshot = screenshot[y1:y2, x1:x2]
            result = self.find_template('anomaly_filter_btn', screenshot=region_screenshot)
            if result:
                actual_x = result['center_x'] + x1
                actual_y = result['center_y'] + y1
                print(f"    [点击] 异常筛选按钮 at ({result['center_x']}, {result['center_y']}) → 实际({actual_x}, {actual_y})")
                self.tap_with_offset(actual_x, actual_y, offset=5)
                time.sleep(1)
            else:
                print(f"    异常筛选按钮未找到")
        else:
            print(f"    获取截图失败")

    def step_check_in_anomaly_space(self) -> bool:
        """检测是否已在异常空间中（通过敌对怪物+一键锁定判断）

        Returns:
            True: 已在异常空间中，可以跳过跃迁直接进入战斗
            False: 不在异常空间中，需要继续跃迁
        """
        print("\n" + "=" * 50)
        print("步骤: 检测是否在异常空间中")
        print("=" * 50)

        screenshot = self.capture_and_cache()
        if screenshot is None:
            print(f"    获取截图失败")
            return False

        # 检测敌对怪物
        monsters_found = self.check_monsters(screenshot)
        print(f"    敌对怪物检测: {'有' if monsters_found else '无'}")

        # 检测一键锁定
        one_key_lock_result = self.find_template('one_key_lock', screenshot)
        one_key_lock_found = one_key_lock_result is not None
        print(f"    一键锁定检测: {'有' if one_key_lock_found else '无'}")

        # 有敌对怪物或一号锁定之一，说明已经在异常空间中
        if monsters_found or one_key_lock_found:
            print(f"    [判断] 已在异常空间中（敌对怪物:{monsters_found}, 一键锁定:{one_key_lock_found}），跳过跃迁")
            return True

        print(f"    [判断] 不在异常空间中，需要跃迁")
        return False

    def step_click_anomaly(self) -> bool:
        """匹配并点击异常空间（优先级: 大型 > 检察官 > 中型 > 小型，同坐标异常cluster屏蔽）"""
        print("\n" + "=" * 50)
        print("步骤: 匹配异常空间")
        print("=" * 50)

        # 异常模板列表（用于收集匹配）
        anomaly_priority = [
            ('anomaly_angel_large', '天使大型异常'),
            ('anomaly_angel_inspector', '高级天使检察官异常'),
            ('anomaly_angel_medium', '天使中型异常'),
            ('anomaly_angel_small', '天使小型异常'),
        ]

        screenshot = self.load_screenshot()

        # 收集所有异常匹配
        all_matches = []
        for template_name, chinese_name in anomaly_priority:
            results = self.find_all_template(template_name, screenshot)
            for r in results:
                all_matches.append({
                    'type': template_name,
                    'chinese_name': chinese_name,
                    'x': int(r['center_x']),
                    'y': int(r['center_y']),
                    'confidence': r.get('confidence', 0)
                })

        if not all_matches:
            print(f"    未找到任何异常模板")
            return False

        print(f"    找到异常共 {len(all_matches)} 个")

        # 聚类：同坐标附近的归为同一cluster，以置信度最高的为代表点
        clusters = {}  # key: cluster_id, value: {'rep': (x,y), 'members': [matches]}
        for m in all_matches:
            placed = False
            for cid, cluster in clusters.items():
                rep_x, rep_y = cluster['rep']
                dist = ((m['x'] - rep_x) ** 2 + (m['y'] - rep_y) ** 2) ** 0.5
                if dist <= self.anomaly_position_threshold:
                    cluster['members'].append(m)
                    if m['confidence'] > next(mm['confidence'] for mm in cluster['members']):
                        cluster['rep'] = (m['x'], m['y'])
                    placed = True
                    break
            if not placed:
                clusters[len(clusters)] = {'rep': (m['x'], m['y']), 'members': [m]}
        print(f"    聚类后共 {len(clusters)} 个独立异常")

        # 过滤：只排除与上次点击位置重复的cluster（用代表点比较）
        valid_clusters = []
        for cid, cluster in clusters.items():
            rep_x, rep_y = cluster['rep']
            excluded = False
            for px, py in self.last_anomaly_positions:
                dist = ((rep_x - px) ** 2 + (rep_y - py) ** 2) ** 0.5
                if dist <= self.anomaly_position_threshold:
                    print(f"        cluster ({rep_x}, {rep_y}) 与上次位置重复({dist:.0f}px)，跳过")
                    excluded = True
                    break
            if not excluded:
                valid_clusters.append(cluster)

        if not valid_clusters:
            print(f"    所有异常都与历史位置重复")
            return False

        # 从每个有效cluster取出置信度最高的member组成有效匹配列表
        valid_matches = []
        for cluster in valid_clusters:
            best_member = max(cluster['members'], key=lambda x: x['confidence'])
            valid_matches.append(best_member)

        # 按上次异常类型决定本次优先级
        # 规则：大型→小型， 中型→大型， 小型→检察官/中型， 检察官→大型
        import random
        random.shuffle(valid_matches)  # 先打乱

        last_type = self.last_anomaly_type
        if last_type == 'anomaly_angel_large':
            # 大型 → 优先小型
            priority_order = ['anomaly_angel_small', 'anomaly_angel_medium', 'anomaly_angel_large', 'anomaly_angel_inspector']
        elif last_type == 'anomaly_angel_medium':
            # 中型 → 优先大型
            priority_order = ['anomaly_angel_large', 'anomaly_angel_small', 'anomaly_angel_medium', 'anomaly_angel_inspector']
        elif last_type == 'anomaly_angel_small':
            # 小型 → 优先检察官/中型
            priority_order = ['anomaly_angel_inspector', 'anomaly_angel_medium', 'anomaly_angel_small', 'anomaly_angel_large']
        elif last_type == 'anomaly_angel_inspector':
            # 检察官 → 优先大型
            priority_order = ['anomaly_angel_large', 'anomaly_angel_medium', 'anomaly_angel_small', 'anomaly_angel_inspector']
        else:
            # 首次/未知 → 默认大/中/小随机，检察官最高
            normal_types = ['anomaly_angel_large', 'anomaly_angel_medium', 'anomaly_angel_small']
            random.shuffle(normal_types)
            priority_order = ['anomaly_angel_inspector'] + normal_types

        best = None
        for template_name in priority_order:
            for m in valid_matches:
                if m['type'] == template_name:
                    best = m
                    break
            if best:
                break

        if not best:
            print(f"    未找到有效异常")
            return False

        clicked_x, clicked_y = best['x'], best['y']
        print(f"    选择 {best['chinese_name']} ({clicked_x}, {clicked_y}), 置信度:{best['confidence']:.3f}")
        self.tap_with_offset(clicked_x, clicked_y, offset=10)

        # 更新追踪状态 - 本次跃迁用（失败重试）+ 历史列表（跨跃迁持久化）
        self.last_anomaly_type = best['type']
        self.last_anomaly_positions = [(clicked_x, clicked_y)]
        self.all_anomaly_positions.append((clicked_x, clicked_y))

        # 检察官异常需要特殊处理流程
        if best['type'] == 'anomaly_angel_inspector':
            self.inspector_mode = True
            print(f"    [检察官异常] 进入特殊流程")
        else:
            self.inspector_mode = False

        print(f"    等待 {self.anomaly_click_wait} 秒...")
        time.sleep(self.random_wait(self.anomaly_click_wait))
        return True

    def run_inspector_special_flow(self):
        """检察官异常特殊流程：跃迁落地后点击侧边图标→加速轨道→激活→等待44s"""
        print("\n" + "=" * 50)
        print("[检察官异常] 执行特殊流程")
        print("=" * 50)

        x1, y1, x2, y2 = self.get_scaled_region(self.inspector_special_region)
        print(f"    检测区域: ({x1}, {y1}) - ({x2}, {y2})")

        # 1. 在右侧区域查找侧边图标并点击
        print(f"    [1/4] 查找加速轨道侧边图标...")
        screenshot = self.load_screenshot()
        if screenshot is not None:
            region = screenshot[y1:y2, x1:x2]
            result = self.matcher.find_template(region, self.templates['accelerate_track_side_icon'], threshold=0.7)
            if result:
                cx = int(result['center_x']) + x1
                cy = int(result['center_y']) + y1
                print(f"        找到侧边图标 at ({cx}, {cy})")
                self.tap_with_offset(cx, cy, offset=10)
                time.sleep(self.random_wait(1))
            else:
                print(f"        未找到侧边图标")

        # 2. 查找并点击加速轨道（全图搜索，模板较大）
        print(f"    [2/4] 查找加速轨道...")
        screenshot = self.load_screenshot()
        if screenshot is not None:
            result = self.matcher.find_template(screenshot, self.templates['accelerate_track_icon'], threshold=0.7)
            if result:
                cx = int(result['center_x'])
                cy = int(result['center_y'])
                print(f"        找到加速轨道 at ({cx}, {cy})")
                self.tap_with_offset(cx, cy, offset=10)
                time.sleep(self.random_wait(1))
            else:
                print(f"        未找到加速轨道")

        # 3. 查找并点击激活按钮
        print(f"    [3/4] 查找激活按钮...")
        activate_result = self.find_template('activate_btn')
        if activate_result:
            print(f"        找到激活按钮 at ({activate_result['center_x']}, {activate_result['center_y']})")
            self.tap_with_offset(activate_result['center_x'], activate_result['center_y'], offset=10)
        else:
            print(f"        未找到激活按钮")

        # 3.5 等待4秒后重新查找并点击异常图标筛选
        print(f"    [3.5/5] 等待4秒后点击异常图标筛选...")
        time.sleep(self.random_wait(4))
        x1, y1, x2, y2 = self.inspector_special_region
        screenshot = self.capture_and_cache()
        if screenshot is not None:
            region_screenshot = screenshot[y1:y2, x1:x2]
            filter_result = self.find_template('anomaly_filter_btn', screenshot=region_screenshot)
            if filter_result:
                actual_x = filter_result['center_x'] + x1
                actual_y = filter_result['center_y'] + y1
                print(f"        异常筛选按钮 at ({filter_result['center_x']}, {filter_result['center_y']}) → 实际({actual_x}, {actual_y})")
                self.tap_with_offset(actual_x, actual_y, offset=5)
            else:
                print(f"        异常筛选按钮未找到")
        time.sleep(1)

        # 4. 等待44秒后开始正式战斗流程
        wait_time = 44
        print(f"    [5/5] 等待 {wait_time} 秒后开始战斗...")
        time.sleep(self.random_wait(wait_time))

        print(f"    [检察官异常] 特殊流程完成")

    def step_click_warp(self) -> bool:
        """匹配并点击跃迁（简化版，由主流程控制时序）"""
        result = self.find_template('warp_btn')
        if result:
            self.tap_with_offset(result['center_x'], result['center_y'], offset=10)
            return True
        return False

    def click_one_key_lock_once(self):
        """点击一键锁定（带间隔检测）"""
        current_time = time.time()

        if current_time - self.last_one_key_lock_time < self.one_key_lock_interval:
            return False

        result = self.find_template('one_key_lock')
        if result:
            self.tap_with_offset(result['center_x'], result['center_y'], offset=10)
            self.last_one_key_lock_time = current_time
            return True

        return False

    def click_equipment_once(self):
        """点击装备（防御模块、综合火控系统、旗舰掠能器）- 每个跃迁只点一次"""
        if self.equipment_clicked_this_warp:
            return

        clicked_positions = []

        def add_clicked_position(x, y):
            clicked_positions.append((int(x), int(y)))

        def click_with_verify(cx: int, cy: int, screenshot=None) -> bool:
            self.tap_with_offset(cx, cy, offset=10)
            add_clicked_position(cx, cy)

            if self.verify_equipment_activated(cx, cy, screenshot):
                return True
            else:
                self.tap_with_offset(cx, cy, offset=10)
                if self.verify_equipment_activated(cx, cy, screenshot):
                    return True
                return False

        screenshot = self.load_screenshot()

        # 综合火控系统 - 找到几个就点几个
        fire_results = self.find_all_template('fire_control')
        # 防御模块 - 优先点击（使用统一状态管理）
        defense_results = self.find_all_template('defense_module')
        print(f"[装备] 防御模块 x{len(defense_results)}")
        for r in defense_results:
            cx, cy = int(r['center_x']), int(r['center_y'])
            print(f"[装备]   点击 ({cx}, {cy})")
            self.click_defense_module(cx, cy)
            time.sleep(1)
            # 检测环形进度条进度
            ss2 = self.capture_and_cache()
            if ss2 is not None:
                prog = self.detect_ring_progress(ss2, cx, cy)
                if prog >= 0:
                    print(f"[装备]   防御模块进度: {prog*100:.0f}%")
            time.sleep(1)

        # 综合火控系统
        print(f"[装备] 综合火控系统 x{len(fire_results)}")
        for r in fire_results:
            cx, cy = int(r['center_x']), int(r['center_y'])
            print(f"[装备]   点击 ({cx}, {cy})")
            click_with_verify(cx, cy, screenshot)
            time.sleep(1)
            ss4 = self.capture_and_cache()
            if ss4 is not None:
                prog = self.detect_ring_progress(ss4, cx, cy)
                if prog >= 0:
                    print(f"[装备]   火控系统进度: {prog*100:.0f}%")
            time.sleep(1)

        # 旗舰掠能器 - 找到几个就点几个
        cap_results = self.find_all_template('cap_energy')
        print(f"[装备] 旗舰掠能器 x{len(cap_results)}")
        for r in cap_results:
            cx, cy = int(r['center_x']), int(r['center_y'])
            print(f"[装备]   点击 ({cx}, {cy})")
            click_with_verify(cx, cy, screenshot)
            time.sleep(1)
            ss3 = self.capture_and_cache()
            if ss3 is not None:
                prog = self.detect_ring_progress(ss3, cx, cy)
                if prog >= 0:
                    print(f"[装备]   掠能器进度: {prog*100:.0f}%")
            time.sleep(1)

        # 打捞器 - 跃迁落地后首次点击
        salvager_results = self.find_all_template('salvager')
        print(f"[装备] 打捞器 x{len(salvager_results)}")
        for r in salvager_results:
            cx, cy = int(r['center_x']), int(r['center_y'])
            print(f"[装备]   点击 ({cx}, {cy})")
            self.tap_with_offset(cx, cy, offset=10)
            time.sleep(1)

        self.equipment_clicked_this_warp = True

    def check_enemies(self, screenshot=None) -> tuple:
        """检查是否有敌对目标（红色预检测+模板匹配）"""
        if screenshot is None:
            screenshot = self.load_screenshot()
        if screenshot is None:
            return False, []

        # 裁剪到检测区域
        x1, y1, x2, y2 = self.get_scaled_region(self.enemy_detection_region)
        region = screenshot[y1:y2, x1:x2]

        # 敌对模板分组（同一类敌人可能多个模板）
        enemy_template_groups = {
            '挑战隐匿': ['enemy_1'],
            '暴风级守卫': ['enemy_2'],
            '狂暴级战列舰': ['enemy_3'],
            '猎犬级护卫': ['enemy_4'],
            '探索级护卫': ['enemy_5'],
            '龙卷风级巡洋': ['enemy_cruiser', 'enemy_cruiser2'],
            '台风级战列舰': ['enemy_battleship'],
            '刺客级巡洋': ['enemy_assassin'],
            '长尾鲛级护卫': ['enemy_shark'],
            '死亡漩涡级战列舰': ['enemy_vortex'],
            '赛那波级巡洋': ['enemy_sinabo'],
            '德拉米尔级护卫': ['enemy_delamir'],
            '工业级涅鲁斯': ['industrial_nerus'],
            '工业级伊米卡': ['industrial_imika'],
            '工业级座头鲸': ['industrial_zuotouyu'],
        }

        found_enemies = []

        # 按组进行模板匹配，每组只保留置信度最高的匹配
        for group_name, template_names in enemy_template_groups.items():
            best_match = None
            for template_name in template_names:
                if template_name not in self.templates:
                    continue
                path = self.templates[template_name]
                if not os.path.exists(path):
                    continue

                results = self.matcher.find_all_matches(region, path, threshold=self.ENEMY_TEMPLATE_THRESHOLD)
                for r in results:
                    conf = r.get('confidence', 0)
                    if best_match is None or conf > best_match['confidence']:
                        best_match = {
                            'type': group_name,
                            'x': int(r['center_x']) + x1,
                            'y': int(r['center_y']) + y1,
                            'confidence': conf
                        }

            if best_match is not None:
                found_enemies.append(best_match)

        # 按置信度排序
        found_enemies.sort(key=lambda e: e['confidence'], reverse=True)

        return len(found_enemies) > 0, found_enemies

    def _check_red_in_region(self, region, threshold_ratio=0.01) -> bool:
        """检测区域内是否有红色像素（敌对目标预检测）

        Args:
            region: BGR格式图像区域
            threshold_ratio: 红色像素占比阈值

        Returns:
            True if red pixels exceed threshold
        """
        if region is None or region.size == 0:
            return False

        h, w = region.shape[:2]
        total_pixels = w * h

        # HSV红色检测
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 | mask2
        red_count = np.count_nonzero(red_mask)
        red_ratio = red_count / total_pixels

        print(f"[红色检测] 像素:{red_count}, 占比:{red_ratio:.4f} ({red_ratio*100:.2f}%), 阈值:{threshold_ratio}")

        return red_ratio >= threshold_ratio

    def check_danger(self, screenshot=None) -> Tuple[bool, dict]:
        """检测危险（使用本地玩家栏OCR检测）

        Returns:
            (是否检测到危险, {'x': x, 'y': y, 'count': 玩家数量})
        """
        if screenshot is None:
            screenshot = self.capture_and_cache()
        if screenshot is None:
            return False, {}

        # 使用本地玩家栏检测（定位+OCR）
        danger_found, danger_info = self.check_local_player_enemies(screenshot)
        return danger_found, danger_info

    def check_monsters(self, screenshot=None) -> bool:
        """检测怪物（复用战斗检测，有任意敌对目标即为有怪）"""
        has_enemies, _ = self.check_enemies(screenshot)
        return has_enemies

    def locate_local_player_bar(self, screenshot=None) -> bool:
        """定位本地玩家栏检测区域（只在开始和跃迁时调用）

        Returns:
            True if located successfully
        """
        if screenshot is None:
            screenshot = self.capture_and_cache()
        if screenshot is None:
            return False

        bar_result = self.find_template('local_player_detect', screenshot=screenshot)
        if not bar_result:
            return False

        self.local_player_bar_position = (bar_result['center_x'], bar_result['center_y'])
        # 存储本地模板左上角，用于计算敌对标签检测区域
        bx, by, _, _ = bar_result['bbox']
        self.local_player_bar_top_left = (bx, by)
        return True

    def check_local_player_enemies(self, screenshot=None) -> Tuple[bool, dict]:
        """检测本地玩家栏中的敌对玩家（使用缓存位置，每30秒OCR检测一次）

        Returns:
            (是否检测到敌对, {'x': x, 'y': y, 'confidence': conf})
        """
        if screenshot is None:
            screenshot = self.capture_and_cache()
        if screenshot is None:
            return False, {}

        current_time = time.time()

        # 如果没有缓存的位置，需要先定位
        if self.local_player_bar_position is None:
            if not self.locate_local_player_bar(screenshot):
                return False, {}
            self.last_local_player_check_time = current_time
            return False, {}  # 刚定位完，这次不检测

        bar_x, bar_y = self.local_player_bar_position

        # 每30秒进行一次OCR检测
        if current_time - self.last_local_player_check_time >= self.local_player_check_interval:
            self.last_local_player_check_time = current_time

            # 在缓存位置附近搜索数字
            h, w = screenshot.shape[:2]
            search_radius_x = 300
            search_radius_y = 60

            x1 = max(0, bar_x - search_radius_x)
            y1 = max(0, bar_y - search_radius_y)
            x2 = min(w, bar_x + search_radius_x)
            y2 = min(h, bar_y + search_radius_y)

            region = screenshot[y1:y2, x1:x2]

            if self.ocr.available:
                results = self.ocr.recognize_text(region)
                numbers = []
                min_confidence = 0.7  # OCR识别置信度阈值
                for res in results:
                    confidence = res.get('confidence', 0)
                    if confidence < min_confidence:
                        continue  # 跳过低置信度结果
                    text = res.get('text', '')
                    import re
                    digits = re.findall(r'\d+', text)
                    for d in digits:
                        numbers.append((d, confidence))

                if numbers:
                    max_num = max([int(d[0]) for d in numbers if d[0].isdigit()])
                    if max_num > 0:
                        return True, {'x': bar_x, 'y': bar_y, 'count': max_num}

            return False, {}
        else:
            return False, {}

    def cache_equipment_positions(self, screenshot=None) -> bool:
        """缓存装备区域模板坐标（启动时和每10分钟刷新一次）

        Returns:
            True if caching successful
        """
        if screenshot is None:
            screenshot = self.capture_and_cache()
        if screenshot is None:
            return False

        print(f"    [装备缓存] 开始缓存装备坐标...")

        # 要缓存的装备模板
        equipment_templates = ['defense_module', 'fire_control', 'cap_energy']

        cached_count = 0
        for name in equipment_templates:
            # 使用 find_all_template 获取所有匹配位置
            results = self.find_all_template(name, screenshot=screenshot)
            if results:
                positions = [(int(r['center_x']), int(r['center_y'])) for r in results]
                self.equipment_cache[name] = positions
                print(f"    [装备缓存] {name}: {positions}")
                cached_count += 1
            else:
                print(f"    [装备缓存] {name}: 未找到")

        self.last_equipment_cache_time = time.time()
        print(f"    [装备缓存] 完成，已缓存 {cached_count}/{len(equipment_templates)} 种装备")
        return cached_count > 0

    def get_cached_equipment(self, name: str, index: int = 0) -> Optional[Tuple[int, int]]:
        """获取缓存的装备坐标（如果缓存过期则重新缓存）

        Args:
            name: 装备名称
            index: 返回第index个缓存的位置（默认返回第一个）

        Returns:
            (x, y) 装备坐标，或 None
        """
        current_time = time.time()

        # 检查缓存是否需要刷新
        if current_time - self.last_equipment_cache_time >= self.equipment_cache_interval:
            print(f"    [装备缓存] 缓存已过期（{self.equipment_cache_interval}秒），重新缓存...")
            self.cache_equipment_positions(  )

        # 返回缓存的坐标
        if name in self.equipment_cache and self.equipment_cache[name]:
            positions = self.equipment_cache[name]
            if index < len(positions):
                return positions[index]
            # 如果索引超出范围，返回第一个
            return positions[0] if positions else None
        return None

    def get_all_cached_equipment(self, name: str) -> List[Tuple[int, int]]:
        """获取缓存的所有装备坐标

        Returns:
            [(x, y), ...] 所有缓存的装备坐标列表
        """
        current_time = time.time()

        # 检查缓存是否需要刷新
        if current_time - self.last_equipment_cache_time >= self.equipment_cache_interval:
            print(f"    [装备缓存] 缓存已过期（{self.equipment_cache_interval}秒），重新缓存...")
            self.cache_equipment_positions()

        if name in self.equipment_cache:
            return self.equipment_cache[name]
        return []
  
    def get_defense_module_state(self) -> dict:
        """获取母舰防御模块状态

        Returns:
            {'state': 'activated'/'restart_delay'/'ready', 'elapsed': 秒数, 'remaining': 剩余秒数}
        """
        if self.defense_module_last_click_time == 0:
            return {'state': 'ready', 'elapsed': 0, 'remaining': 0, 'can_warp': True, 'info': '未点击/可点击'}

        elapsed = time.time() - self.defense_module_last_click_time
        total = self.defense_module_total_cycle
        activation = self.defense_module_activation_time
        restart_delay = self.defense_module_restart_delay

        if elapsed < activation:
            remaining = activation - elapsed
            state_info = f"激活中({remaining:.0f}s后可跃迁)"
            return {'state': 'activated', 'elapsed': elapsed, 'remaining': remaining, 'can_warp': False, 'info': state_info}
        elif elapsed < total:
            remaining = total - elapsed
            state_info = f"重启延迟({remaining:.0f}s后可重新点击)"
            return {'state': 'restart_delay', 'elapsed': elapsed, 'remaining': remaining, 'can_warp': True, 'info': state_info}
        else:
            state_info = "可点击"
            return {'state': 'ready', 'elapsed': elapsed, 'remaining': 0, 'can_warp': True, 'info': state_info}

    def verify_equipment_activated(self, cx: int, cy: int, screenshot=None) -> bool:
        """验证装备是否启动成功（模板验证）"""
        time.sleep(self.equip_verify_wait)

        if screenshot is None:
            screenshot = self.load_screenshot()
        if screenshot is None:
            return False

        if 'equip_status_bar' in self.templates:
            path = self.templates['equip_status_bar']
            if os.path.exists(path):
                results = self.matcher.find_all_matches(screenshot, path, threshold=self.equip_success_confidence)

                best_conf = 0
                for r in results:
                    sx, sy = int(r['center_x']), int(r['center_y'])
                    dist = ((sx - cx) ** 2 + (sy - cy) ** 2) ** 0.5
                    if dist <= self.equip_verify_radius:
                        if r['confidence'] > best_conf:
                            best_conf = r['confidence']

                if best_conf > 0:
                    return True

        return False

    def run_combat_loop(self):
        """战斗循环 - 多线程独立运行"""
        print("\n" + "=" * 50)
        print(">>> 进入战斗循环 <<<")
        print("=" * 50)

        import threading

        # 共享状态
        self.combat_running = True
        self.combat_should_exit = False
        self.equipment_triggered = threading.Event()
        self.equipment_triggered.clear()
        self.one_key_lock_cooldown = False
        self.one_key_lock_cooldown_until = 0  # 冷却结束时间戳
        self.one_key_lock_cooldown_time = 12  # 一键锁定冷却时间（秒）
        self.enemy_check_interval = 10  # 敌对目标检测间隔（秒）
        self.last_enemy_check_time = 0  # 上次敌对检测时间
        self.enemy_check_found_last = False  # 上次敌对检测是否找到目标
        self.one_key_lock_check_interval = 5  # 一键锁定检测间隔（秒）
        self.last_one_key_lock_check_time = 0  # 上次一键锁定检测时间
        self.no_target_consecutive_rounds = 0  # 连续没有目标的轮数
        self.no_lock_consecutive_rounds = 0  # 连续一键锁定未检测到的轮数
        self.last_enemy_count = 0  # 上次检测的敌人数
        self.enemy_count_unchanged_rounds = 0  # 敌对数量连续相同的轮数
        self.last_exit_check_time = 0  # 上次退出检查时间（独立计时器）
        self.warp_cycle_counter = 0  # 跃迁周期计数器，每跃迁一次递增

        # 本地玩家栏定位（只在开始和跃迁时定位一次）
        self.local_player_bar_position = None  # 检测区域的屏幕坐标 (x, y)
        self.local_player_bar_top_left = None  # 本地模板左上角（用于计算敌对标签检测区域）
        self.local_player_check_interval = 30  # 本地玩家栏OCR检测间隔（秒）
        self.last_local_player_check_time = 0  # 上次OCR检测时间

        # 装备区域坐标缓存（固定装备位置，10分钟刷新一次）
        self.equipment_cache_interval = 600  # 装备缓存刷新间隔（秒）= 10分钟
        self.last_equipment_cache_time = 0  # 上次缓存更新时间
        self.equipment_cache = {}  # 装备坐标缓存 {name: [(x, y), ...]}
        self.armor_repair_start_time = 0  # 旗舰装甲维修器开启时间
        self.last_cap_energy_click_time = 0  # 上次掠能器点击时间

        # 母舰防御模块计时
        self.defense_module_activation_time = 111  # 激活时间（秒）- 不能跃迁
        self.defense_module_restart_delay = 61  # 重启延迟（秒）- 可以跃迁
        self.defense_module_total_cycle = 172  # 总周期（秒）

        last_warp_cycle = [-1]  # 上次处理的跃迁周期

        # 状态检测
        self.last_status_check_time = 0  # 上次状态检测时间
        self.armor_repair_needed = False  # 是否需要执行装甲维修
        self.armor_repair_clicked = False  # 装甲维修是否已点击开启
        self.first_lock_after_warp = True  # 跃迁后首次锁定标志（首次锁定才触发装备点击）

        # 检察官异常防御模块追踪
        self.inspector_defense_clicked = 0  # 已点击次数
        self.inspector_defense_timers = []  # 下次可点击时间列表
        self.inspector_first_lock_time = 0  # 首次锁定时间

        # 装备进度检测
        self.progress_check_interval = 10  # 进度检测间隔（秒）
        self.last_progress_check_time = 0  # 上次进度检测时间

        def is_position_in_list(x, y, pos_list, threshold=20):
            """检查位置是否已在列表中"""
            for px, py in pos_list:
                if abs(x - px) <= threshold and abs(y - py) <= threshold:
                    return True
            return False

        def equipment_timer_thread():
            """装备计时线程 - 独立计时，按顺序执行装备点击"""

            while self.combat_running:
                # 等待装备触发信号
                if self.equipment_triggered.wait(timeout=0.5):
                    print(f"[装备线程] 收到触发信号，等待2秒...")
                    time.sleep(self.random_wait(2))
                    self.equipment_triggered.clear()
                else:
                    current_loop_time = time.time()

                    # === 防御模块定期点击：172秒周期（落地后首次点击由主循环触发）===
                    if self.defense_module_last_click_time > 0:
                        elapsed = current_loop_time - self.defense_module_last_click_time
                        if elapsed >= 172:
                            defense_positions_list = self.get_all_cached_equipment('defense_module')
                            if not defense_positions_list:
                                # 缓存为空时直接搜索
                                results = self.find_all_template('defense_module')
                                if results:
                                    defense_positions_list = [(int(r['center_x']), int(r['center_y'])) for r in results]
                            if defense_positions_list:
                                cx, cy = defense_positions_list[0]
                                if self.defense_module_activated:
                                    # 激活状态 → 点击关闭（60s冷却）
                                    self.click_defense_module(cx, cy)
                                    print(f"[装备] 防御模块172s到，点击关闭")
                                else:
                                    # 关闭状态 → 点击激活（重新开始172s）
                                    self.click_defense_module(cx, cy)
                                    print(f"[装备] 防御模块172s到，点击激活")

                    # === 装甲维修自动开关（基于红色检测）===
                    # 有红色 → 启动；无红色 → 关闭
                    if self.armor_repair_needed and not self.armor_repair_clicked:
                        # 需要启动装甲维修
                        armor_result = self.find_template('armor_repair')
                        if armor_result:
                            cx, cy = int(armor_result['center_x']), int(armor_result['center_y'])
                            self.tap_with_offset(cx, cy, offset=10)
                            self.armor_repair_clicked = True
                            self.armor_repair_start_time = current_loop_time
                    elif not self.armor_repair_needed and self.armor_repair_clicked:
                        # 需要关闭装甲维修（无红色但开着）
                        armor_result = self.find_template('armor_repair')
                        if armor_result:
                            cx, cy = int(armor_result['center_x']), int(armor_result['center_y'])
                            self.tap_with_offset(cx, cy, offset=10)
                            self.armor_repair_clicked = False

                    # === 装备进度和防御模块定期检测：每10秒检测一次 ===
                    if current_loop_time - self.last_progress_check_time >= self.progress_check_interval:
                        self.last_progress_check_time = current_loop_time

                        # 防御模块状态倒计时
                        if self.defense_module_last_click_time > 0:
                            defense_state = self.get_defense_module_state()
                            print(f"[防御模块] {defense_state['info']}")

                        # 装备进度检测
                        if self.equipment_cache:
                            ss = self.capture_and_cache()
                            if ss is not None:
                                for eq_name, positions in self.equipment_cache.items():
                                    if not positions:
                                        continue
                                    cx, cy = positions[0]
                                    prog = self.detect_ring_progress(ss, cx, cy)
                                    if prog >= 0:
                                        print(f"[装备进度] {eq_name}: {prog*100:.0f}%")

        # 启动装备计时线程
        timer_thread = threading.Thread(target=equipment_timer_thread, daemon=True)
        timer_thread.start()

        # 状态检测线程 - 独立计时，每30秒检测一次状态
        def status_check_thread():
            """状态检测线程 - 每30秒检测损失/健康状态"""
            while self.combat_running:
                time.sleep(self.status_check_interval)

                if not self.combat_running:
                    break

                screenshot = self.load_screenshot()
                if screenshot is None:
                    continue

                # 在指定区域检测红色像素（装甲维修状态检测）
                x1, y1, x2, y2 = self.get_scaled_region(self.status_detection_region)
                region = screenshot[y1:y2, x1:x2]

                # 检测是否有红色（损失状态：R高,G低,B低）
                import numpy as np
                bgr = region.astype(np.float32)
                # 红色特征：R > 150, G < 100, B < 100
                red_mask = (bgr[:, :, 2] > 150) & (bgr[:, :, 1] < 100) & (bgr[:, :, 0] < 100)
                red_ratio = np.sum(red_mask) / red_mask.size if red_mask.size > 0 else 0

                # 有红色 → 装甲维修应该是开启状态；无红色 → 应关闭
                if red_ratio > 0.01:  # 超过1%红色像素
                    self.armor_repair_needed = True
                    self.armor_repair_clicked = False
                else:
                    self.armor_repair_needed = False
                    self.armor_repair_clicked = False

        status_thread = threading.Thread(target=status_check_thread, daemon=True)
        status_thread.start()

        # 主循环 - 只检测一键锁定
        # 初始化：定位本地玩家栏
        init_screenshot = self.capture_and_cache()
        self.locate_local_player_bar(init_screenshot)

        while True:
            # 检查是否需要退出
            if self.combat_should_exit:
                self.combat_running = False
                self.equipment_triggered.set()
                timer_thread.join(timeout=2)
                status_thread.join(timeout=2)
                return False

            screenshot = self.capture_and_cache()

            # 检查敌对玩家标签：检测到危险（玩家数量>0）则撤退
            danger_found, danger_info = self.check_danger(screenshot)
            if danger_found:
                print(f"[警告] 敌对玩家(count={danger_info.get('count', 0)})，紧急撤退")
                self.combat_running = False
                self.equipment_triggered.set()
                timer_thread.join(timeout=2)
                status_thread.join(timeout=2)
                self.emergency_retreat()
                return True

            # 敌对标签模板检测（以本地模板左上角为起点，尺寸611x1134）
            if screenshot is None:
                screenshot = self.capture_and_cache()
            if screenshot is not None and self.local_player_bar_top_left is not None:
                bx, by = self.local_player_bar_top_left
                # 以本地模板左上角为右上角（往右延伸），尺寸611x11134
                x1 = int(bx + self.scale_x * 0)   # 实际上就是bx（模板右边起点）
                y1 = int(by)
                x2 = int(x1 + 611 * self.scale_x)
                y2 = int(min(y1 + 1134 * self.scale_y, self.actual_height))
                region = screenshot[y1:y2, x1:x2]
                tag_result = self.matcher.find_template(region, self.templates['enemy_player_tag'], threshold=self.enemy_tag_threshold)
                if tag_result:
                    # 严格验证颜色：匹配中心点必须是红色（敌对标签特征）
                    import numpy as np
                    h, w = region.shape[:2]
                    cx_in_region = int(tag_result['center_x'])
                    cy_in_region = int(tag_result['center_y'])
                    if 0 <= cy_in_region < h and 0 <= cx_in_region < w:
                        pixel_bgr = region[cy_in_region, cx_in_region]
                        pixel_rgb = np.array([pixel_bgr[2], pixel_bgr[1], pixel_bgr[0]])
                        dist = np.sqrt(np.sum((pixel_rgb - np.array(self.danger_color)) ** 2))
                        color_ok = dist <= self.danger_color_tolerance * np.sqrt(3)
                    else:
                        color_ok = False
                    if not color_ok:
                        print(f"[调试] 敌对标签模板匹配但颜色不符({dist:.1f})，跳过")
                    else:
                        cx = int(tag_result['center_x']) + x1
                        cy = int(tag_result['center_y']) + y1
                        print(f"[警告] 敌对标签 detected({tag_result['confidence']:.3f}) at ({cx},{cy})，紧急撤退")
                        self.combat_running = False
                        self.equipment_triggered.set()
                        timer_thread.join(timeout=2)
                        status_thread.join(timeout=2)
                        self.emergency_retreat()
                        return True

            current_time = time.time()

            # 每15秒检测敌对目标
            if current_time - self.last_enemy_check_time >= self.enemy_check_interval:
                screenshot = self.capture_and_cache()
                _, enemies_list = self.check_enemies(screenshot)
                self.last_enemy_check_time = current_time
                if enemies_list:
                    if len(enemies_list) == 1:
                        e = enemies_list[0]
                        print(f"[敌对] x1 坐标({e['x']},{e['y']}) 置信度:{e['confidence']:.2f}")
                    else:
                        print(f"[敌对] x{len(enemies_list)}")

            # 每5秒检测一键锁定
            if current_time - self.last_one_key_lock_check_time >= self.one_key_lock_check_interval:
                # 冷却期间不检测
                if not (self.one_key_lock_cooldown and current_time < self.one_key_lock_cooldown_until):
                    screenshot = self.capture_and_cache()
                    one_key_lock_result = self.find_template('one_key_lock', screenshot)
                    one_key_lock_found = one_key_lock_result is not None
                    self.last_one_key_lock_check_time = current_time

                    if one_key_lock_found:
                        if self.click_one_key_lock_once():
                            if not self.equipment_cache:
                                self.cache_equipment_positions()
                            # 只有跃迁后首次锁定才触发装备点击
                            if self.first_lock_after_warp:
                                print(f"[装备触发] 跃迁#{self.warp_cycle_counter}首次锁定，触发装备")
                                self.equipment_triggered.set()
                                self.click_equipment_once()  # 落地点击：火控+掠能器+防御模块（仅一次）
                                self.first_lock_after_warp = False
                            else:
                                print(f"[锁定] 非首次锁定，跳过装备触发")
                            self.one_key_lock_cooldown = True
                            self.one_key_lock_cooldown_until = time.time() + self.one_key_lock_cooldown_time
                            # 首次锁定，重置掠能器计时
                            self.last_cap_energy_click_time = 0

            # 掠能器每28秒随机点击一个（跃迁落地后持续工作，直到跃迁）
            if self.combat_running and self.equipment_cache:
                if current_time - self.last_cap_energy_click_time >= 28:
                    self.last_cap_energy_click_time = current_time
                    cap_positions_list = self.get_all_cached_equipment('cap_energy')
                    if cap_positions_list:
                        import random
                        cx, cy = random.choice(cap_positions_list)
                        print(f"[装备] 掠能器续期 ({cx}, {cy})")
                        self.tap_with_offset(cx, cy, offset=10)

            # 15秒周期结束时判断是否需要退出
            exit_elapsed = current_time - self.last_exit_check_time
            if exit_elapsed >= self.enemy_check_interval:
                self.last_exit_check_time = current_time
                screenshot = self.capture_and_cache()

                # 检测一键锁定
                one_key_lock_result = self.find_template('one_key_lock', screenshot)
                has_lock = one_key_lock_result is not None

                # 模板匹配检测敌人
                has_enemies, enemies_list = self.check_enemies(screenshot)
                current_enemy_count = len(enemies_list)

                # 输出敌对目标类型统计
                if enemies_list:
                    counts = {}
                    for e in enemies_list:
                        counts[e['type']] = counts.get(e['type'], 0) + 1
                    summary = ', '.join(f"{t} x{n}" for t, n in counts.items())
                    print(f"[敌对检测] {summary}")

                # 追踪一键锁定未检测到连续轮数
                if not has_lock:
                    self.no_lock_consecutive_rounds += 1
                else:
                    self.no_lock_consecutive_rounds = 0

                # 追踪敌对数量不变连续轮数
                if current_enemy_count == self.last_enemy_count and current_enemy_count > 0:
                    self.enemy_count_unchanged_rounds += 1
                else:
                    self.enemy_count_unchanged_rounds = 0
                self.last_enemy_count = current_enemy_count

                # 退出条件：2轮无目标(无敌人且无锁) 或 (3轮一键锁定未检测到 且 3轮敌对数量不变)
                should_exit = False
                exit_reason = ""
                if not has_enemies and not has_lock:
                    self.no_target_consecutive_rounds += 1
                    if self.no_target_consecutive_rounds >= 2:
                        should_exit = True
                        exit_reason = f"2轮无目标(敌:{has_enemies},锁:{has_lock})"
                else:
                    self.no_target_consecutive_rounds = 0

                # 3轮无锁定但有敌对目标 → 点击敌对 → 集中火力
                if self.no_lock_consecutive_rounds >= 3 and current_enemy_count > 0:
                    if enemies_list:
                        e = enemies_list[0]
                        print(f"[激活尝试] 3轮无锁定但有敌对，点击敌对({e['x']},{e['y']})")
                        self.tap_with_offset(e['x'], e['y'], offset=10)
                        time.sleep(self.random_wait(1))
                        ss = self.capture_and_cache()
                        cf = self.find_template('concentrated_fire', ss)
                        if cf:
                            print(f"[集中火力] 找到集中火力，点击 ({cf['center_x']},{cf['center_y']})")
                            self.tap_with_offset(cf['center_x'], cf['center_y'], offset=10)
                            self.no_lock_consecutive_rounds = 0
                            self.enemy_count_unchanged_rounds = 0
                            continue
                        else:
                            print(f"[集中火力] 未找到集中火力")
                    # 如果没找到敌人或集中火力，继续走后续逻辑（可能被遮挡等）

                # 敌对数量连续N轮不变时，点击集中火力
                if self.enemy_count_unchanged_rounds >= 3 and current_enemy_count > 0:
                    ss = self.capture_and_cache()
                    cf = self.find_template('concentrated_fire', ss)
                    if cf:
                        print(f"[集中火力] 敌对数量{current_enemy_count}连续{self.enemy_count_unchanged_rounds}轮不变，点击集中火力({cf['center_x']},{cf['center_y']})")
                        self.tap_with_offset(cf['center_x'], cf['center_y'], offset=10)
                        self.enemy_count_unchanged_rounds = 0
                        self.no_lock_consecutive_rounds = 0
                        continue

                if self.no_lock_consecutive_rounds >= 4 and self.enemy_count_unchanged_rounds >= 4 and current_enemy_count >= 10:
                    # 尝试点击敌对位置 + 集中火力
                    if enemies_list:
                        e = enemies_list[0]
                        print(f"[集中火力] 尝试点击敌对位置 ({e['x']}, {e['y']})")
                        self.tap_with_offset(e['x'], e['y'], offset=10)
                        time.sleep(self.random_wait(1))
                        ss = self.capture_and_cache()
                        cf = self.find_template('concentrated_fire', ss)
                        if cf:
                            print(f"[集中火力] 找到集中火力，点击 ({cf['center_x']}, {cf['center_y']})")
                            self.tap_with_offset(cf['center_x'], cf['center_y'], offset=10)
                            # 重置计数器，继续刷怪
                            self.no_lock_consecutive_rounds = 0
                            self.enemy_count_unchanged_rounds = 0
                            self.no_target_consecutive_rounds = 0
                            continue
                        else:
                            print(f"[集中火力] 未找到集中火力，退出")
                    should_exit = True
                    exit_reason = f"4轮无锁定且敌对数量不变({current_enemy_count})"

                if should_exit:
                    print(f"[退出] {exit_reason}，跃迁...")

                    # 防御模块激活中则点击取消
                    defense_state = self.get_defense_module_state()
                    if not defense_state['can_warp']:
                        defense_results = self.find_all_template('defense_module')
                        if defense_results:
                            r = defense_results[0]
                            cx, cy = int(r['center_x']), int(r['center_y'])
                            self.tap_with_offset(cx, cy, offset=10)

                    if self.inspector_mode:
                        x1, y1, x2, y2 = self.get_scaled_region(self.inspector_special_region)
                        screenshot = self.capture_and_cache()
                        if screenshot is not None:
                            region = screenshot[y1:y2, x1:x2]
                            result = self.matcher.find_template(region, self.templates['anomaly_filter_btn'], threshold=0.7)
                            if result:
                                cx = int(result['center_x']) + x1
                                cy = int(result['center_y']) + y1
                                self.tap_with_offset(cx, cy, offset=10)
                                time.sleep(2)
                        self.inspector_mode = False

                    self.combat_running = False
                    self.equipment_triggered.set()
                    timer_thread.join(timeout=2)
                    status_thread.join(timeout=2)
                    return False

            time.sleep(1)

    def emergency_retreat(self):
        """紧急撤退：(74,486) -> 2s -> (239,760) -> 2s -> (1271,638) -> 2s -> (74,486) -> 确认防御关闭并稳定启动"""
        print("\n" + "=" * 50)
        print(">>> 紧急撤退程序 <<<")
        print("=" * 50)

        # 1. 点击 (74, 486)
        print(f"    [1/4] 点击 (74, 486)")
        self.tap_with_offset(74, 486, offset=10)
        time.sleep(2)

        # 2. 点击 (239, 760)
        print(f"    [2/4] 点击 (239, 760)")
        self.tap_with_offset(239, 760, offset=10)
        time.sleep(2)

        # 3. 点击 (1271, 638)
        print(f"    [3/4] 点击 (1271, 638)")
        self.tap_with_offset(1271, 638, offset=10)
        time.sleep(2)

        # 4. 点击 (74, 486)
        print(f"    [4/4] 点击 (74, 486)")
        self.tap_with_offset(74, 486, offset=10)

        # 5. 确保防御模块关闭
        print(f"    [5/6] 检查防御模块状态...")
        defense_state = self.get_defense_module_state()
        print(f"        防御模块状态: {defense_state['info']}")
        if self.defense_module_activated:
            defense_results = self.find_all_template('defense_module')
            if defense_results:
                cx, cy = int(defense_results[0]['center_x']), int(defense_results[0]['center_y'])
                self.click_defense_module(cx, cy)
                time.sleep(1)

        # 6. 重置防御模块计时，确保稳定启动
        print(f"    [6/6] 重置防御模块计时")
        self.defense_module_last_click_time = 0
        self.defense_module_activated = False
        self.defense_module_deactivate_cooldown = 0  # 紧急撤退清空冷却

        print(f"\n    紧急撤退完成，刷怪循环结束！")

    def run_loop(self, max_loops: int = 999):
        """主循环"""
        print("\n" + "#" * 60)
        print("# EVE Bot 刷怪脚本 v4")
        print("#" * 60)

        # 检查设备
        devices = self.adb.get_devices()
        if not devices:
            print("错误: 未检测到设备")
            return
        print(f"已连接设备: {devices}\n")

        loop_count = 0

        while loop_count < max_loops:
            loop_count += 1
            print(f"\n[{self.ts()}] {'='*60}")
            print(f"[{self.ts()}] >>> 第 {loop_count} 轮开始")
            print(f"[{self.ts()}] {'='*60}")

            # ===== 步骤1: 检测是否在站内 =====
            print(f"\n[{self.ts()}] [步骤1/5] 检测是否在空间站内...")
            screenshot = self.load_screenshot()
            if self.is_in_station(screenshot):
                # 在站内，离站
                self.step_undock()
            else:
                print(f"    当前在太空，跳过离站步骤")

            # ===== 步骤2: 点击侧边开关 =====
            print(f"\n[{self.ts()}] [步骤2/5] 点击侧边界面开关...")
            self.step_toggle_sidebar()

            # 步骤2.5: 如果防御模块已激活，点击关闭后才能跃迁
            if self.defense_module_activated:
                defense_result = self.find_template('defense_module')
                if defense_result:
                    self.click_defense_module(defense_result['center_x'], defense_result['center_y'])
                    time.sleep(self.random_wait(2))

            # ===== 步骤2.8: 滑动屏幕后再匹配异常 =====
            print(f"    [{self.ts()}] [滑动] (2730,1120) → (2725,811) 模拟人工滑动...")
            self.swipe_with_human(2730, 1120, 2725, 811, duration_ms=500)
            time.sleep(self.random_wait(2))

            # ===== 步骤3: 匹配异常并点击 =====
            print(f"\n[{self.ts()}] [步骤3/5] 匹配异常空间...")
            anomaly_found = self.step_click_anomaly()
            if not anomaly_found:
                print(f"    [{self.ts()}] [警告] 未找到异常模板，尝试点击异常图标筛选...")
                x1, y1, x2, y2 = self.inspector_special_region
                screenshot = self.capture_and_cache()
                if screenshot is not None:
                    region_screenshot = screenshot[y1:y2, x1:x2]
                    filter_result = self.find_template('anomaly_filter_btn', screenshot=region_screenshot)
                    if filter_result:
                        actual_x = filter_result['center_x'] + x1
                        actual_y = filter_result['center_y'] + y1
                        print(f"    [{self.ts()}] [筛选] 点击异常筛选 ({actual_x}, {actual_y})")
                        self.tap_with_offset(actual_x, actual_y, offset=5)
                        time.sleep(self.random_wait(2))
                        # 重试匹配异常
                        anomaly_found = self.step_click_anomaly()
                if not anomaly_found:
                    print(f"    [{self.ts()}] [警告] 未找到异常模板，截图保存以便排查")
                    screenshot = self.load_screenshot()
                    if screenshot is not None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_path = os.path.join(self.screenshot_dir, f"no_anomaly_{timestamp}.png")
                        cv2.imwrite(save_path, screenshot)
                        print(f"    截图已保存: {save_path}")
                    print(f"    等待5秒后重试...")
                    time.sleep(self.random_wait(5))
                    continue

            # ===== 步骤4: 点击跃迁 =====
            print(f"\n[{self.ts()}] [步骤4/5] 匹配跃迁按钮...")

            # 点击异常后等待跃迁按钮出现（最多20秒，超过则重新匹配异常）
            warp_found = False
            warp_wait_start = time.time()
            max_warp_wait = 20  # 超过20秒则重新选择异常
            while time.time() - warp_wait_start < max_warp_wait:
                result = self.find_template('warp_btn')
                if result:
                    warp_found = True
                    time.sleep(self.random_wait(1))
                    print(f"    找到跃迁按钮，点击 ({result['center_x']}, {result['center_y']})")
                    self.tap_with_offset(result['center_x'], result['center_y'], offset=10)
                    self.equipment_clicked_this_warp = False  # 重置装备点击状态
                    self.warp_cycle_counter += 1  # 跃迁周期递增
                    print(f"    [跃迁] 进入新跃迁周期 #{self.warp_cycle_counter}")
                    break
                elapsed = time.time() - warp_wait_start
                print(f"    未找到跃迁按钮，已等待 {elapsed:.0f}s，继续等待...")
                time.sleep(self.random_wait(3))

            if not warp_found:
                print(f"    [警告] 等待{ max_warp_wait}秒仍未找到跃迁按钮，清空位置重新选择异常")
                self.last_anomaly_positions = []
                self.last_anomaly_type = None
                continue

            # 3. 点击惯性稳定器（跃迁后等待2秒）
            time.sleep(self.random_wait(2))
            inertia_result = self.find_template('inertia')
            if inertia_result:
                print(f"    找到惯性稳定器，点击 ({inertia_result['center_x']}, {inertia_result['center_y']})")
                self.tap_with_offset(inertia_result['center_x'], inertia_result['center_y'], offset=10)

            # 4. 跃迁前检查防御模块（如果距上次点击超过171秒，提前点击）
            if self.defense_module_last_click_time > 0:
                elapsed = time.time() - self.defense_module_last_click_time
                if elapsed >= 171:
                    defense_positions = self.get_all_cached_equipment('defense_module')
                    if defense_positions:
                        cx, cy = defense_positions[0]
                        print(f"    [跃迁前] 防御模块距上次点击{elapsed:.0f}s，执行点击 ({cx}, {cy})")
                        self.click_defense_module(cx, cy)

            # 5. 等待44秒（OCR已暂时屏蔽）
            print(f"    等待 {self.warp_wait} 秒...")
            time.sleep(self.random_wait(self.warp_wait))

            # 检察官异常特殊流程：点击加速轨道和激活按钮
            if self.inspector_mode:
                self.run_inspector_special_flow()

            # ===== 步骤5: 战斗循环 =====
            print(f"\n[{self.ts()}] [步骤5/5] 进入战斗循环...")
            # 跃迁后重置本地玩家栏位置和装备缓存，下次战斗开始时会重新定位和缓存
            self.local_player_bar_position = None
            self.equipment_cache = {}  # 清空装备缓存，落地首次锁定时会重新缓存
            should_exit = self.run_combat_loop()
            if should_exit:
                print(f"\n[{self.ts()}] <<< 刷怪循环已结束 >>>")
                return  # 紧急撤退或用户停止，退出整个循环

            # 退出战斗循环（无目标/无锁定），清空异常位置，重新选择新异常
            print(f"\n[{self.ts()}] <<< 第 {loop_count} 轮结束，跃迁新异常 >>>")
            self.last_anomaly_positions = []
            # 不重置 last_anomaly_type，保留以便下次按规则选择不同类型
            continue

        print(f"\n达到最大循环次数 {max_loops}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='EVE Bot 刷怪脚本 v4')
    parser.add_argument('--device', '-d', help='设备ID')
    parser.add_argument('--loops', '-l', type=int, default=999, help='最大循环次数')

    args = parser.parse_args()

    script = RatFarmV2(args.device)
    script.run_loop(max_loops=args.loops)


if __name__ == "__main__":
    main()
