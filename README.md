# EVE Bot 刷怪脚本 v4

## 项目简介

基于模板匹配的 EVE Online 安卓模拟器自动化刷怪脚本，支持多分辨率自动适配。

**功能特性：**
- 自动离站/进站检测
- 智能异常空间识别与选择（优先级：大型 > 中型 > 检察官 > 小型）
- 自动化战斗流程（一键锁定、装备管理）
- 防御模块智能管理（172秒周期，60秒冷却）
- 危险玩家检测与紧急撤退
- 多线程并行处理（战斗循环、装备计时、状态检测）
- 自动分辨率适配（参考分辨率 3200x2136）

---

## 目录结构

```
evebot/
├── rat_farm_v2.py           # 刷怪脚本主程序（v4版本）
├── adb_controller.py        # ADB连接和设备控制
├── screen_capture.py        # 屏幕截图和图像处理
├── template_matcher.py      # 模板匹配功能
├── ocr_recognizer.py        # OCR文本识别（已禁用）
├── annotate_tool.py         # 屏幕标注工具（独立程序）
├── requirements.txt        # Python依赖列表
├── mods/                    # 游戏模板图片目录
│   ├── anomaly_*.png        # 异常空间模板（4种类型）
│   ├── equip_*.png         # 装备按钮模板
│   ├── enemy_*.png         # 敌对目标模板
│   └── *.png              # 其他UI模板
├── screenshots/            # 截图保存目录
├── dataset/                # 机器学习数据集（YOLO格式）
├── records/                # 运行记录
├── templates/              # 测试模板
└── README.md              # 项目说明文档
```

---

## 模块说明

### 1. rat_farm_v2.py - 主程序

**核心类**: `RatFarmV2`

**主要方法**:
- `run_loop()` - 主循环入口
- `run_combat_loop()` - 战斗循环
- `step_undock()` - 离站流程
- `step_click_anomaly()` - 异常空间匹配与选择
- `step_click_warp()` - 跃迁流程
- `emergency_retreat()` - 紧急撤退
- `click_equipment_once()` - 装备点击（防御/火控/掠能器/打捞器）

**核心逻辑**:
```
主循环 (run_loop):
  步骤1 → 检测站内/站外 → 站内则离站
  步骤2 → 点击侧边界面开关
  步骤3 → 匹配异常空间（自动排除重复位置）
  步骤4 → 点击跃迁按钮
  步骤5 → 进入战斗循环
```

### 2. adb_controller.py - ADB控制

**核心功能**:
- 设备连接与管理
- 屏幕截图 (screencap)
- 点击/滑动模拟
- 分辨率获取

**主要方法**:
- `get_devices()` - 获取设备列表
- `screenshot()` - 获取屏幕截图
- `tap(x, y)` - 点击坐标
- `swipe(x1, y1, x2, y2, duration)` - 滑动操作
- `get_resolution()` - 获取设备分辨率

### 3. screen_capture.py - 屏幕采集

**核心功能**:
- ADB截图采集
- 截图缓存机制（减少重复截图）
- 分辨率自动适配
- 图像预处理

**主要方法**:
- `capture()` - 采集截图
- `capture_and_cache()` - 采集并缓存截图

### 4. template_matcher.py - 模板匹配

**核心功能**:
- OpenCV模板匹配
- 多模板批量匹配
- 置信度阈值控制
- 坐标自动缩放

**主要方法**:
- `match()` - 单模板匹配
- `match_all()` - 多模板匹配
- `find_best_match()` - 查找最佳匹配

### 5. annotate_tool.py - 标注工具

**独立运行的图形界面工具**:
- 加载截图或ADB实时画面
- 框选区域标注
- 保存为JSON格式
- 支持截图链记录

---

## 战斗流程详解

### 战斗循环 (run_combat_loop)

**多线程架构**:

```
┌─────────────────────────────────────────────┐
│              主循环线程                       │
│  - 检测敌对目标                              │
│  - 检测一键锁定                              │
│  - 判断退出条件                              │
└─────────────────────────────────────────────┘
          ↑ 触发信号 ↓
┌─────────────────────────────────────────────┐
│            装备计时线程                      │
│  - 防御模块管理（172秒周期）                 │
│  - 综合火控系统                             │
│  - 旗舰掠能器续期                           │
│  - 打捞器点击                               │
│  - 装甲维修（状态驱动）                     │
└─────────────────────────────────────────────┘
          ↑ 定时检测 ↓
┌─────────────────────────────────────────────┐
│           状态检测线程                      │
│  - 损失/健康状态检测                         │
│  - 防御模块倒计时显示                        │
│  - 装备进度检测                             │
└─────────────────────────────────────────────┘
```

### 退出条件

| 条件 | 说明 |
|------|------|
| 2轮无目标 | 连续2轮无敌人且无一键锁定 → 跃迁新异常 |
| 3轮无锁定且敌对不变 | 连续3轮检测不到锁定 且 3轮敌人数量相同 |
| 敌对玩家 | 检测到红色敌人标签 → 紧急撤退 |

---

## 装备管理详解

### 防御模块 (Defense Module)

**状态机制**: 切换模式

| 当前状态 | 点击效果 | 后续状态 |
|---------|---------|---------|
| 激活中 | 关闭 | 60秒冷却 |
| 关闭中 | 激活 | 172秒自动关闭 |

**时间线示例**:
```
00:00 - 跃迁落地 → 激活防御模块（倒计时172秒）
02:52 - 172秒到 → 自动关闭（冷却60秒）
03:05 - 跃迁新异常 → 重新激活
```

### 综合火控系统 (Fire Control)

- 跃迁落地后首次锁定触发
- 每个位置点击一次
- 不需要周期性维护

### 旗舰掠能器 (Cap Energy)

- 首次激活后每28秒随机点击一个
- 持续工作直到跃迁
- 用于维持电容能量

### 打捞器 (Salvager) 🆕

- 跃迁落地后首次锁定触发
- 每个位置点击一次
- 用于自动打捞战场残骸

### 装甲维修器 (Armor Repair)

- 状态驱动：检测到红色警告才开启
- 开启60秒后自动关闭
- 用于自动修复装甲损伤

### 装备点击流程

```
首次点击一键锁定
  ↓
触发装备信号
  ↓
等待2秒（避免太快点击）
  ↓
按顺序点击：
  1. 防御模块（所有位置）
  2. 综合火控系统（所有位置）
  3. 旗舰掠能器（所有位置）
  4. 打捞器（所有位置）🆕
```

---

## 模板列表

### 异常空间模板

| 文件名 | 说明 | 优先级 |
|--------|------|--------|
| `anomaly_angel_large.png` | 天使大型异常 | 1（最高） |
| `anomaly_angel_medium.png` | 天使中型异常 | 2 |
| `anomaly_angel_inspector.png` | 天使检察官异常 | 3 |
| `anomaly_angel_small.png` | 天使小型异常 | 4（最低） |

### 装备模板

| 文件名 | 说明 | 点击策略 |
|--------|------|---------|
| `equip_defense_module.png` | 母舰防御模块 | 172秒周期 |
| `equip_fire_control.png` | 综合火控系统 | 首次激活 |
| `equip_cap_Energy.png` | 旗舰掠能器 | 28秒续期 |
| `equip_armor_repair.png` | 旗舰装甲维修器 | 状态驱动 |
| `equip_inertia.png` | 惯性稳定器 | 跃迁后1次 |
| `equip_salvager.png` | 打捞器 | 首次激活 🆕 |

### 敌对目标模板

| 模板名 | 文件 | 说明 |
|--------|------|------|
| enemy_frigate | `enemy_frigate.png` | 敌对护卫舰 |
| enemy_cruiser | `enemy_cruiser.png` | 敌对巡洋舰 |
| enemy_battleship | `enemy_battleship.png` | 敌对战列舰 |
| enemy_battlecruiser | `enemy_battlecruiser.png` | 敌对战列巡洋舰 |
| enemy_1 | `enemy_1.png` | 挑战隐匿 |
| enemy_2 | `enemy_2.png` | 暴风级守卫 |
| enemy_3 | `enemy_3.png` | 狂暴级战列舰 |
| enemy_4 | `enemy_4.png` | 猎犬级护卫 |
| enemy_5 | `enemy_5.png` | 探索级护卫 |
| enemy_assassin | `enemy_assassin.png` | 刺客级巡洋 |
| enemy_shark | `enemy_shark.png` | 长尾鲛级护卫 |
| enemy_vortex | `enemy_vortex.png` | 死亡漩涡级战列舰 |
| enemy_sinabo | `enemy_sinabo.png` | 赛那波级巡洋 |
| enemy_delamir | `enemy_delamir.png` | 德拉米尔级护卫 |

### 工业目标（危险检测）

| 模板名 | 文件 | 说明 |
|--------|------|------|
| industrial_nerus | `industrial_nerus.png` | 工业级涅鲁斯 |
| industrial_zuotouyu | `industrial_zuotouyu.png` | 工业级座头鲸 |
| industrial_imika | `industrial_imika.png` | 工业级伊米卡 |

### 状态模板

| 文件名 | 说明 |
|--------|------|
| `status_loss.png` | 损失状态 |
| `status_health.png` | 健康状态 |
| `enemy_player_tag.png` | 敌对玩家标签 |

### 其他UI模板

| 文件名 | 说明 |
|--------|------|
| `undock_btn.png` | 离站按钮 |
| `warp_btn.png` | 跃迁按钮 |
| `one_key_lock.png` | 一键锁定 |
| `sidebar_toggle.png` | 侧边界面开关 |
| `anomaly_filter_btn.png` | 异常筛选按钮 |
| `activate_btn.png` | 激活按钮 |
| `accelerate_track_icon.png` | 加速轨道图标 |

---

## 配置参数

在 `rat_farm_v2.py` 的 `__init__` 中调整：

### 时间参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `undock_wait` | 25秒 | 离站等待时间 |
| `anomaly_click_wait` | 3秒 | 点击异常后等待 |
| `warp_wait` | 44秒 | 跃迁后等待 |
| `one_key_lock_interval` | 10秒 | 一键锁定最小间隔 |
| `no_target_timeout` | 15秒 | 无目标超时 |
| `equip_click_interval` | 3秒 | 装备点击间隔 |

### 检测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enemy_check_interval` | 15秒 | 敌对检测间隔 |
| `status_check_interval` | 30秒 | 状态检测间隔 |
| `progress_check_interval` | 10秒 | 装备进度检测间隔 |
| `one_key_lock_check_interval` | 5秒 | 一键锁定检测间隔 |
| `one_key_lock_cooldown_time` | 12秒 | 一键锁定冷却时间 |

### 防御模块参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `defense_module_total_cycle` | 172秒 | 总周期 |
| `defense_module_activation_time` | 111秒 | 激活时间 |
| `defense_module_restart_delay` | 61秒 | 可重启延迟 |
| `defense_module_deactivate_cooldown` | 60秒 | 关闭后冷却 |

### 掠能器参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cap_energy_interval` | 28秒 | 掠能器续期间隔 |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**依赖列表**:
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- Pillow >= 10.0.0

### 2. 配置ADB

```bash
# 连接设备
adb devices

# 如果是无线连接
adb tcpip 5555
adb connect <设备IP>:5555
```

### 3. 运行脚本

```bash
python rat_farm_v2.py
```

**可选参数**:
- `-d, --device`: 指定设备ID
- `-l, --loops`: 最大循环次数（默认999）

### 4. 运行标注工具

```bash
python annotate_tool.py
```

---

## 分辨率适配

脚本使用 **3200 x 2136** 作为参考分辨率，自动适配其他分辨率。

### 支持的分辨率

| 分辨率 | 缩放比例 |
|--------|---------|
| 720p (1280×720) | (0.400, 0.337) |
| 1080p (1920×1080) | (0.600, 0.506) |
| 1440p (2560×1440) | (0.800, 0.674) |
| 4K (3840×2160) | (1.200, 1.011) |
| 参考 (3200×2136) | (1.000, 1.000) |

### 模板制作建议

- 使用参考分辨率 3200x2136 制作模板
- 模拟器分辨率建议设置为 3200x2136
- 模板格式：PNG，支持透明度

---

## 注意事项

1. **分辨率**: 模板基于 3200x2136，自动缩放
2. **首次运行**: 自动检测设备分辨率
3. **ADB连接**: 无线连接需开启 TCP 模式
4. **匹配阈值**: 默认0.65，敌对0.75
5. **防御模块**: 激活中不能跃迁
6. **紧急撤退**: 检测敌对玩家自动触发

---

## 故障排除

| 问题 | 解决方案 |
|------|---------|
| 未找到异常模板 | 检查设备分辨率，确认模板文件存在 |
| 跃迁超时 | 增加 `warp_wait`，检查网络 |
| 一键锁定不触发装备 | 检查 `first_lock_after_warp` 重置 |
| 脚本运行慢 | 检查截图频率设置 |

---

## 更新日志

### v4.0.1 (最新)
- 移除 PaddleOCR 依赖（兼容 Python 3.14）
- 修复 OpenCV setLogLevel 兼容性问题
- 添加打捞器 (Salvager) 支持
- 添加打捞器模板 `equip_salvager.png`

### v4.0
- 集成多线程屏幕同步
- 优化装备点击策略
- 实现防御模块状态管理
- 增加紧急撤退机制
- 支持多分辨率自动适配
- 增加检察官异常特殊处理