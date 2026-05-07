# -*- coding: utf-8 -*-
"""
EVE Bot 模板模块

提供游戏界面模板图片的名称映射和查找功能。
模板文件存放在 mods/ 目录下，按类型分类组织。

模板命名规范：
- 异常空间: anomaly_*.png
- 装备按钮: equip_*.png
- 敌对目标: enemy_*.png / ship_*.png
- 状态显示: status_*.png
- UI按钮: *.png

使用方法：
    from mods import TEMPLATE_NAMES, get_template_path
    path = get_template_path('one_key_lock')
"""

import os
import glob
import cv2
import numpy
from typing import Optional, Dict, List, Tuple

# -----------------------------------------------------------------------------
# 模板名称映射 (拼音名 -> 中文名)
# -----------------------------------------------------------------------------

TEMPLATE_NAMES = {
    # ==================== 异常空间模板 ====================
    # 用于刷怪时识别不同类型的异常空间
    'anomaly_angel_large': '天使大型异常',
    'anomaly_angel_medium': '天使中型异常',
    'anomaly_angel_inspector': '高级天使检察官异常',
    'anomaly_angel_small': '天使小型异常',
    'anomaly_angel_base': '天使基地',
    'anomaly_corps_station2': '军团哨站2',

    # ==================== 检察官特殊按钮 ====================
    # 高级天使检察官异常需要点击的特殊按钮
    'accelerate_track_icon': '加速轨道图标',
    'activate_btn': '激活按钮',
    'anomaly_filter_btn': '异常图标筛选',

    # ==================== 装备按钮模板 ====================
    # 旗舰装备的开关按钮
    'equip_cap_Energy': '旗舰掠能器',
    'equip_fire_control': '综合火控系统',
    'equip_inertia': '惯性稳定器',
    'equip_defense_module': '母舰防御模块',
    'equip_armor_repair': '旗舰装甲维修器',
    'equip_status_bar': '运行状态条',

    # ==================== 敌对目标模板 ====================
    # 用于检测屏幕上的敌对舰船（今日标注，存于 mods/new_enemies/）
    'enemy_1': '挑战隐匿',
    'enemy_2': '暴风级守卫',
    'enemy_3': '狂暴级战列舰',
    'enemy_4': '猎犬级护卫',
    'enemy_5': '探索级护卫',
    'enemy_cruiser': '龙卷风级巡洋',
    'enemy_cruiser2': '龙卷风级站巡',
    'enemy_battleship': '台风级战列舰',
    'enemy_assassin': '刺客级巡洋',
    'enemy_shark': '长尾鲛级护卫',
    'enemy_vortex': '死亡漩涡级战列舰',
    'enemy_sinabo': '赛那波级巡洋',
    'enemy_delamir': '德拉米尔级护卫',
    # 工业船（战斗检测）
    'industrial_nerus': '工业级涅鲁斯',
    'industrial_imika': '工业级伊米卡',
    'industrial_zuotouyu': '工业级座头鲸',

    # ==================== 战术按钮 ====================
    'one_key_lock': '一键锁定',
    'space_tactic': '空天战术',

    # ==================== 太空站外界面按钮 ====================
    'approach_btn': '接近按钮',
    'warp_btn': '跃迁按钮',
    'close_btn': '关闭按钮',

    # ==================== 侧边栏和存点 ====================
    'sidebar_toggle': '侧边界面张开/隐藏',
    'save_point_00': '存点00',
    'set_destination': '设置为终点',
    'navigate_to_destination': '导航到终点',
    'save_location': '存点位置',

    # ==================== 站内界面按钮 ====================
    # 空间站内的各种功能入口
    'warehouse': '仓库',
    'ship_config': '舰船配置',
    'wallet': '钱包',
    'market_entrance': '市场入口',
    'control_building': '控制建筑',
    'activity_btn': '活动按钮',
    'insurance_btn': '保险按钮',
    'deep_sleep': '深眠之境',
    'management_btn': '管理按钮',
    'industry_btn': '工业按钮',
    'repair_station': '维修站',
    'clone_center': '克隆中心',
    'shrink_local_ui': '缩小本地界面',
    'mail_box': '邮箱',
    'emote_btn': '表情按钮',
    'local_count': '本地人数',

    # ==================== 状态模板 ====================
    # 用于检测舰船状态（损失/健康）
    'status_loss': '损失状态',
    'status_health': '健康状态',
}


# -----------------------------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------------------------

def get_template_path(name: str) -> Optional[str]:
    """
    通过名称获取模板文件路径

    Args:
        name: 模板名称（拼音或中文）

    Returns:
        模板文件完整路径，不存在返回None
    """
    mods_dir = os.path.join(os.path.dirname(__file__))

    # 如果是拼音名，直接使用
    if name in TEMPLATE_NAMES:
        pinyin = name
    else:
        # 通过中文名查找拼音
        pinyin = None
        for p, c in TEMPLATE_NAMES.items():
            if c == name:
                pinyin = p
                break
        if pinyin is None:
            return None

    # 构建完整路径
    path = os.path.join(mods_dir, f'{pinyin}.png')
    if os.path.exists(path):
        return path
    return None


def list_templates() -> List[Tuple[str, str, bool]]:
    """
    列出所有可用模板

    Returns:
        列表，每项为 (拼音名, 中文名, 文件是否存在)
    """
    mods_dir = os.path.join(os.path.dirname(__file__))
    result = []
    for pinyin, chinese in TEMPLATE_NAMES.items():
        path = os.path.join(mods_dir, f'{pinyin}.png')
        result.append((pinyin, chinese, os.path.exists(path)))
    return result


def load_template(name: str) -> Optional[numpy.ndarray]:
    """
    加载模板图片为numpy数组

    Args:
        name: 模板名称

    Returns:
        OpenCV图像数组，失败返回None
    """
    path = get_template_path(name)
    if path is None:
        return None
    try:
        with open(path, 'rb') as f:
            data = f.read()
            return cv2.imdecode(
                numpy.frombuffer(data, numpy.uint8),
                cv2.IMREAD_COLOR
            )
    except Exception:
        return None
