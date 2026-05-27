# -*- coding: utf-8 -*-
"""
YOLO数据集准备工具

功能：
1. 从截图中使用模板匹配自动生成标注
2. 通过模板合成生成多尺度训练数据
3. 输出YOLO格式数据集
4. 自动划分训练集/验证集

使用方法：
    python prepare_yolo_dataset.py --mode auto     # 自动从截图标注
    python prepare_yolo_dataset.py --mode synthetic  # 合成生成训练数据
    python prepare_yolo_dataset.py --mode both       # 两者都执行
"""

import os
import sys
import cv2
import json
import shutil
import random
import argparse
import numpy as np
from glob import glob
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.dirname(__file__))
from template_matcher import TemplateMatcher, imread_unicode

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'yolo_dataset')
SCREENSHOT_DIR = os.path.join(BASE_DIR, 'screenshots')

TEMPLATE_CLASSES = {
    'undock_btn': {'path': 'mods/undock_btn.png', 'group': 'ui'},
    'sidebar_toggle': {'path': 'mods/sidebar_toggle.png', 'group': 'ui'},
    'anomaly_angel_large': {'path': 'mods/anomaly_angel_large.png', 'group': 'anomaly'},
    'anomaly_angel_medium': {'path': 'mods/anomaly_angel_medium.png', 'group': 'anomaly'},
    'anomaly_angel_small': {'path': 'mods/anomaly_angel_small.png', 'group': 'anomaly'},
    'anomaly_angel_inspector': {'path': 'mods/anomaly_angel_inspector.png', 'group': 'anomaly'},
    'warp_btn': {'path': 'mods/warp_btn.png', 'group': 'ui'},
    'inertia': {'path': 'mods/equip_inertia.png', 'group': 'equip'},
    'one_key_lock': {'path': 'mods/one_key_lock.png', 'group': 'equip'},
    'defense_module': {'path': 'mods/equip_defense_module.png', 'group': 'equip'},
    'fire_control': {'path': 'mods/equip_fire_control.png', 'group': 'equip'},
    'cap_energy': {'path': 'mods/equip_cap_Energy.png', 'group': 'equip'},
    'salvager': {'path': 'mods/equip_salvager.png', 'group': 'equip'},
    'armor_repair': {'path': 'mods/equip_armor_repair.png', 'group': 'equip'},
    'activate_btn': {'path': 'mods/activate_btn.png', 'group': 'ui'},
    'anomaly_filter_btn': {'path': 'mods/anomaly_filter_btn.png', 'group': 'ui'},
    'concentrated_fire': {'path': 'mods/concentrated_fire.png', 'group': 'equip'},
    'accelerate_track_side_icon': {'path': 'mods/accelerate_track_side_icon.png', 'group': 'anomaly'},
    'accelerate_track_icon': {'path': 'mods/accelerate_track_icon.png', 'group': 'anomaly'},
    'enemy_1': {'path': 'mods/new_enemies/enemy_1.png', 'group': 'enemy'},
    'enemy_2': {'path': 'mods/new_enemies/enemy_2.png', 'group': 'enemy'},
    'enemy_3': {'path': 'mods/new_enemies/enemy_3.png', 'group': 'enemy'},
    'enemy_4': {'path': 'mods/new_enemies/enemy_4.png', 'group': 'enemy'},
    'enemy_5': {'path': 'mods/new_enemies/enemy_5.png', 'group': 'enemy'},
    'enemy_cruiser': {'path': 'mods/new_enemies/enemy_cruiser.png', 'group': 'enemy'},
    'enemy_cruiser2': {'path': 'mods/new_enemies/enemy_cruiser2.png', 'group': 'enemy'},
    'enemy_battleship': {'path': 'mods/new_enemies/enemy_battleship.png', 'group': 'enemy'},
    'enemy_assassin': {'path': 'mods/new_enemies/enemy_assassin.png', 'group': 'enemy'},
    'enemy_shark': {'path': 'mods/new_enemies/enemy_shark.png', 'group': 'enemy'},
    'enemy_vortex': {'path': 'mods/new_enemies/enemy_vortex.png', 'group': 'enemy'},
    'enemy_sinabo': {'path': 'mods/new_enemies/enemy_sinabo.png', 'group': 'enemy'},
    'enemy_delamir': {'path': 'mods/new_enemies/enemy_delamir.png', 'group': 'enemy'},
    'industrial_nerus': {'path': 'mods/new_enemies/industrial_nerus.png', 'group': 'enemy'},
    'industrial_zuotouyu': {'path': 'mods/new_enemies/industrial_zuotouyu.png', 'group': 'enemy'},
    'industrial_imika': {'path': 'mods/new_enemies/industrial_imika.png', 'group': 'enemy'},
    'save_point_00': {'path': 'mods/save_point_00.png', 'group': 'ui'},
    'set_destination': {'path': 'mods/set_destination.png', 'group': 'ui'},
    'navigate_to_destination': {'path': 'mods/navigate_to_destination.png', 'group': 'ui'},
    'equip_status_bar': {'path': 'mods/equip_status_bar.png', 'group': 'equip'},
    'status_loss': {'path': 'mods/status_loss.png', 'group': 'status'},
    'status_health': {'path': 'mods/status_health.png', 'group': 'status'},
    'enemy_player_tag': {'path': 'mods/enemy_player_tag.png', 'group': 'enemy'},
    'local_player_detect': {'path': 'mods/local_player_detect.png', 'group': 'enemy'},
}


class YOLODatasetBuilder:
    """YOLO数据集构建器"""

    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = output_dir
        self.matcher = TemplateMatcher()
        self.class_names = sorted(TEMPLATE_CLASSES.keys())
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}

        os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)

    def generate_yaml(self):
        """生成YOLO数据集配置文件"""
        yaml_content = f"""# EVE Bot YOLO Dataset Configuration
path: {self.output_dir.replace(os.sep, '/')}
train: images/train
val: images/val

nc: {len(self.class_names)}
names: {self.class_names}
"""
        yaml_path = os.path.join(self.output_dir, 'data.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        print(f"[配置] 已生成: {yaml_path}")
        return yaml_path

    def auto_annotate_screenshots(
        self,
        screenshot_dir: str = SCREENSHOT_DIR,
        threshold: float = 0.7,
        multi_scale: bool = True
    ) -> int:
        """
        从截图自动生成标注

        Args:
            screenshot_dir: 截图目录
            threshold: 匹配阈值
            multi_scale: 是否使用多尺度匹配

        Returns:
            生成的标注数量
        """
        screenshots = glob(os.path.join(screenshot_dir, '*.png'))
        screenshots += glob(os.path.join(screenshot_dir, '*.jpg'))

        if not screenshots:
            print(f"[警告] 截图目录为空: {screenshot_dir}")
            return 0

        total_labels = 0
        for idx, screenshot_path in enumerate(screenshots):
            screenshot = imread_unicode(screenshot_path)
            if screenshot is None:
                continue

            h, w = screenshot.shape[:2]
            labels = []

            for class_name, info in TEMPLATE_CLASSES.items():
                template_path = os.path.join(BASE_DIR, info['path'])
                if not os.path.exists(template_path):
                    continue

                if multi_scale:
                    result = self.matcher.find_template_multiscale(
                        screenshot, template_path,
                        threshold=threshold,
                        scale_range=(0.3, 2.0),
                        scale_steps=20
                    )
                else:
                    result = self.matcher.find_template(
                        screenshot, template_path,
                        threshold=threshold
                    )

                if result:
                    cx = result['center_x'] / w
                    cy = result['center_y'] / h
                    bbox = result['bbox']
                    bw = bbox[2] / w
                    bh = bbox[3] / h

                    cls_id = self.class_to_id[class_name]
                    labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if labels:
                split = 'val' if random.random() < 0.15 else 'train'
                img_name = f"auto_{idx:04d}.png"
                lbl_name = f"auto_{idx:04d}.txt"

                img_path = os.path.join(self.output_dir, 'images', split, img_name)
                lbl_path = os.path.join(self.output_dir, 'labels', split, lbl_name)

                shutil.copy2(screenshot_path, img_path)
                with open(lbl_path, 'w') as f:
                    f.write('\n'.join(labels))

                total_labels += len(labels)
                print(f"  [{idx+1}/{len(screenshots)}] {os.path.basename(screenshot_path)} -> {len(labels)} 标注")

        print(f"[自动标注] 完成，共 {total_labels} 个标注")
        return total_labels

    def generate_synthetic_data(
        self,
        screenshot_dir: str = SCREENSHOT_DIR,
        samples_per_screenshot: int = 15,
        scale_range: Tuple[float, float] = (0.4, 2.0),
        max_templates_per_image: int = 8
    ) -> int:
        """
        通过模板合成生成训练数据

        将模板以不同尺度、旋转、透明度粘贴到截图上，自动生成标注

        Args:
            screenshot_dir: 截图目录
            samples_per_screenshot: 每张截图生成的样本数
            scale_range: 缩放范围
            max_templates_per_image: 每张图最多粘贴的模板数

        Returns:
            生成的样本数量
        """
        screenshots = glob(os.path.join(screenshot_dir, '*.png'))
        screenshots += glob(os.path.join(screenshot_dir, '*.jpg'))

        if not screenshots:
            print(f"[警告] 截图目录为空: {screenshot_dir}")
            return 0

        templates = {}
        for class_name, info in TEMPLATE_CLASSES.items():
            template_path = os.path.join(BASE_DIR, info['path'])
            if os.path.exists(template_path):
                tpl = imread_unicode(template_path)
                if tpl is not None:
                    templates[class_name] = tpl

        if not templates:
            print("[警告] 没有可用的模板")
            return 0

        total_samples = 0
        template_list = list(templates.items())

        for s_idx, screenshot_path in enumerate(screenshots):
            bg = imread_unicode(screenshot_path)
            if bg is None:
                continue

            for sample_idx in range(samples_per_screenshot):
                canvas = bg.copy()
                h, w = canvas.shape[:2]
                labels = []

                num_templates = random.randint(1, max_templates_per_image)
                chosen = random.choices(template_list, k=num_templates)

                for class_name, tpl in chosen:
                    scale = random.uniform(*scale_range)
                    th, tw = tpl.shape[:2]
                    new_w = int(tw * scale)
                    new_h = int(th * scale)

                    if new_w >= w or new_h >= h:
                        continue
                    if new_w < 5 or new_h < 5:
                        continue

                    scaled = cv2.resize(tpl, (new_w, new_h))

                    angle = random.uniform(-10, 10)
                    M = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), angle, 1.0)
                    cos = abs(M[0, 0])
                    sin = abs(M[0, 1])
                    rot_w = int(new_h * sin + new_w * cos)
                    rot_h = int(new_h * cos + new_w * sin)
                    M[0, 2] += (rot_w - new_w) / 2
                    M[1, 2] += (rot_h - new_h) / 2
                    rotated = cv2.warpAffine(scaled, M, (rot_w, rot_h),
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=(0, 0, 0, 0))

                    max_x = w - rot_w
                    max_y = h - rot_h
                    if max_x <= 0 or max_y <= 0:
                        continue
                    px = random.randint(0, max_x)
                    py = random.randint(0, max_y)

                    if rotated.shape[2] == 4:
                        alpha = rotated[:, :, 3] / 255.0
                        for c in range(3):
                            canvas[py:py+rot_h, px:px+rot_w, c] = (
                                alpha * rotated[:, :, c] +
                                (1 - alpha) * canvas[py:py+rot_h, px:px+rot_w, c]
                            ).astype(np.uint8)
                    else:
                        canvas[py:py+rot_h, px:px+rot_w] = rotated

                    cx = (px + rot_w / 2) / w
                    cy = (py + rot_h / 2) / h
                    bw = rot_w / w
                    bh = rot_h / h
                    cls_id = self.class_to_id[class_name]
                    labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                if labels:
                    split = 'val' if random.random() < 0.15 else 'train'
                    img_name = f"syn_{s_idx:03d}_{sample_idx:02d}.png"
                    lbl_name = f"syn_{s_idx:03d}_{sample_idx:02d}.txt"

                    img_path = os.path.join(self.output_dir, 'images', split, img_name)
                    lbl_path = os.path.join(self.output_dir, 'labels', split, lbl_name)

                    cv2.imwrite(img_path, canvas)
                    with open(lbl_path, 'w') as f:
                        f.write('\n'.join(labels))

                    total_samples += 1

            print(f"  [{s_idx+1}/{len(screenshots)}] {os.path.basename(screenshot_path)} -> {samples_per_screenshot} 合成样本")

        print(f"[合成数据] 完成，共 {total_samples} 个样本")
        return total_samples

    def add_manual_screenshot(
        self,
        image_path: str,
        labels: List[Dict]
    ) -> bool:
        """
        添加手动标注的截图

        Args:
            image_path: 截图路径
            labels: 标注列表 [{'class_name': str, 'bbox': (cx, cy, bw, bh)}, ...]
                    bbox 格式: (center_x_ratio, center_y_ratio, width_ratio, height_ratio)

        Returns:
            是否成功
        """
        img = cv2.imread(image_path)
        if img is None:
            return False

        h, w = img.shape[:2]
        yolo_labels = []
        for label in labels:
            cls_name = label['class_name']
            if cls_name not in self.class_to_id:
                continue
            cx, cy, bw, bh = label['bbox']
            cls_id = self.class_to_id[cls_name]
            yolo_labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        split = 'val' if random.random() < 0.15 else 'train'
        name = os.path.splitext(os.path.basename(image_path))[0]
        img_name = f"manual_{name}.png"
        lbl_name = f"manual_{name}.txt"

        img_path = os.path.join(self.output_dir, 'images', split, img_name)
        lbl_path = os.path.join(self.output_dir, 'labels', split, lbl_name)

        shutil.copy2(image_path, img_path)
        with open(lbl_path, 'w') as f:
            f.write('\n'.join(yolo_labels))

        return True

    def get_dataset_stats(self) -> Dict:
        """获取数据集统计信息"""
        stats = {
            'train_images': len(glob(os.path.join(self.output_dir, 'images', 'train', '*'))),
            'val_images': len(glob(os.path.join(self.output_dir, 'images', 'val', '*'))),
            'train_labels': len(glob(os.path.join(self.output_dir, 'labels', 'train', '*.txt'))),
            'val_labels': len(glob(os.path.join(self.output_dir, 'labels', 'val', '*.txt'))),
            'total_classes': len(self.class_names),
            'class_names': self.class_names
        }

        class_counts = {}
        for lbl_file in glob(os.path.join(self.output_dir, 'labels', '**', '*.txt'), recursive=True):
            with open(lbl_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f'unknown_{cls_id}'
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        stats['class_counts'] = class_counts
        return stats


def main():
    parser = argparse.ArgumentParser(description='YOLO数据集准备工具')
    parser.add_argument('--mode', choices=['auto', 'synthetic', 'both'], default='both',
                        help='数据生成模式')
    parser.add_argument('--output', default=OUTPUT_DIR, help='输出目录')
    parser.add_argument('--screenshots', default=SCREENSHOT_DIR, help='截图目录')
    parser.add_argument('--threshold', type=float, default=0.7, help='模板匹配阈值')
    parser.add_argument('--samples', type=int, default=15, help='每张截图合成样本数')
    args = parser.parse_args()

    builder = YOLODatasetBuilder(args.output)
    builder.generate_yaml()

    print("\n" + "=" * 60)
    print("YOLO 数据集准备工具")
    print("=" * 60)
    print(f"输出目录: {args.output}")
    print(f"截图目录: {args.screenshots}")
    print(f"类别数量: {len(builder.class_names)}")
    print(f"模式: {args.mode}")
    print("=" * 60)

    if args.mode in ('auto', 'both'):
        print("\n[阶段1] 自动标注截图...")
        builder.auto_annotate_screenshots(args.screenshots, args.threshold)

    if args.mode in ('synthetic', 'both'):
        print("\n[阶段2] 合成训练数据...")
        builder.generate_synthetic_data(args.screenshots, args.samples)

    print("\n" + "=" * 60)
    stats = builder.get_dataset_stats()
    print("数据集统计:")
    print(f"  训练图片: {stats['train_images']}")
    print(f"  验证图片: {stats['val_images']}")
    print(f"  训练标注: {stats['train_labels']}")
    print(f"  验证标注: {stats['val_labels']}")
    print(f"  类别数量: {stats['total_classes']}")
    if stats['class_counts']:
        print("\n  各类别标注数量:")
        for name, count in sorted(stats['class_counts'].items(), key=lambda x: -x[1]):
            print(f"    {name}: {count}")
    print("=" * 60)
    print("\n数据集已准备完成！")
    print(f"配置文件: {os.path.join(args.output, 'data.yaml')}")
    print(f"\n下一步: python train_yolo.py")


if __name__ == '__main__':
    main()
