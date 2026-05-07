# -*- coding: utf-8 -*-
"""EVE Bot 屏幕标注工具

功能：
1. 加载截图显示
2. 鼠标框选区域
3. 输入区域名称/功能
4. 保存标注结果
5. 从ADB获取实时截图
6. 截图连贯关系标注（界面跳转链）
"""

import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import json
from datetime import datetime
import sys
sys.path.insert(0, os.path.dirname(__file__))
from adb_controller import ADBController


class AnnotationTool:
    def __init__(self, screenshot_path=None):
        self.root = tk.Tk()
        self.root.title("EVE Bot 屏幕标注工具")
        self.root.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}")
        self.root.state('zoomed')

        self.screenshot_path = screenshot_path
        self.annotations = []  # [(x1, y1, x2, y2, name), ...]
        self.current_rect = None
        self.start_x = 0
        self.start_y = 0
        self.show_list = True
        self.view_coords_mode = False  # 查看坐标模式

        # ADB相关
        self.adb = None
        self.adb_device_id = None
        self.screen_size = None  # 实际屏幕尺寸 (width, height)
        self.adb_connected = False

        # 截图连贯关系
        self.screenshot_chain = []  # [{'screenshot': path, 'description': str, 'resolution': (w,h)}, ...]

        self.setup_ui()
        self.load_screenshot()

    def setup_ui(self):
        # 顶部工具栏
        toolbar = tk.Frame(self.root, bg='#2196F3', height=50)
        toolbar.pack(fill=tk.X, side=tk.TOP)

        tk.Label(toolbar, text="EVE标注", bg='#2196F3', fg='white',
                font=('Microsoft YaHei', 14, 'bold')).pack(side=tk.LEFT, padx=10)

        tk.Button(toolbar, text="加载截图", command=self.load_screenshot_dialog,
                bg='#1976D2', fg='white', relief='flat', padx=10).pack(side=tk.LEFT, padx=5)
        tk.Button(toolbar, text="ADB截屏", command=self.capture_from_adb,
                bg='#9C27B0', fg='white', relief='flat', padx=10).pack(side=tk.LEFT, padx=5)
        tk.Button(toolbar, text="保存", command=self.save_annotations,
                bg='#4CAF50', fg='white', relief='flat', padx=10).pack(side=tk.LEFT, padx=5)
        tk.Button(toolbar, text="保存链", command=self.save_screenshot_chain,
                bg='#00BCD4', fg='white', relief='flat', padx=10).pack(side=tk.LEFT, padx=5)
        tk.Button(toolbar, text="清除", command=self.clear_annotations,
                bg='#f44336', fg='white', relief='flat', padx=10).pack(side=tk.LEFT, padx=5)
        tk.Button(toolbar, text="切换列表", command=self.toggle_list,
                bg='#FF9800', fg='white', relief='flat', padx=10).pack(side=tk.RIGHT, padx=5)
        tk.Button(toolbar, text="查看链", command=self.show_chain_dialog,
                bg='#795548', fg='white', relief='flat', padx=10).pack(side=tk.RIGHT, padx=5)
        self.coords_btn = tk.Button(toolbar, text="查看坐标", command=self.toggle_coords_mode,
                bg='#607D8B', fg='white', relief='flat', padx=10)
        self.coords_btn.pack(side=tk.RIGHT, padx=5)

        # 状态栏 - 显示截图尺寸和设备尺寸
        self.status_bar = tk.Label(self.root, text="未加载截图", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 主显示区域 - Canvas带滚动
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas_frame = tk.Frame(self.main_frame, bg='#333')
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        hscroll = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        vscroll = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)

        self.canvas = tk.Canvas(self.canvas_frame, bg='#1a1a1a', cursor='crosshair',
                               xscrollcommand=hscroll.set,
                               yscrollcommand=vscroll.set)
        hscroll.config(command=self.canvas.xview)
        vscroll.config(command=self.canvas.yview)
        hscroll.pack(side=tk.BOTTOM, fill=tk.X)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # 底部标注列表（可折叠）
        self.list_frame = tk.Frame(self.root, bg='white', height=120)
        self.list_frame.pack(fill=tk.X, side=tk.BOTTOM)

        tk.Label(self.list_frame, text="已标注 (双击修改):", font=('Microsoft YaHei', 9, 'bold'),
                bg='white').pack(anchor='w', padx=10, pady=2)

        scrollbar = tk.Scrollbar(self.list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(self.list_frame, font=('Consolas', 9), height=5,
                                 yscrollcommand=scrollbar.set)
        self.listbox.pack(fill=tk.X, padx=10, pady=2)
        scrollbar.config(command=self.listbox.yview)

        self.listbox.bind('<Double-Button-1>', self.on_list_double_click)

    def toggle_list(self):
        self.show_list = not self.show_list
        if self.show_list:
            self.list_frame.pack(fill=tk.X, side=tk.BOTTOM)
        else:
            self.list_frame.pack_forget()

    def toggle_coords_mode(self):
        """切换查看坐标模式"""
        self.view_coords_mode = not self.view_coords_mode
        if self.view_coords_mode:
            self.coords_btn.config(bg='#4CAF50', text="退出坐标")
            self.canvas.config(cursor='crosshair')
            self.status_bar.config(text="查看坐标模式: 点击屏幕查看坐标")
        else:
            self.coords_btn.config(bg='#607D8B', text="查看坐标")
            self.canvas.config(cursor='crosshair')
            self.status_bar.config(text="已退出坐标查看模式")

    def _init_adb(self):
        """初始化ADB连接"""
        if self.adb is None:
            self.adb = ADBController(self.adb_device_id)
            # 尝试获取设备屏幕尺寸
            self.screen_size = self.adb.get_screen_size()
            self.adb_connected = self.adb.adb_path is not None

    def capture_from_adb(self):
        """从ADB设备获取实时截图"""
        self._init_adb()

        if not self.adb_connected:
            # 尝试获取已连接设备
            devices = self.adb.get_devices()
            if devices:
                self.adb_device_id = devices[0][0]
                self.adb = ADBController(self.adb_device_id)
                self.adb_connected = self.adb.adb_path is not None

        if not self.adb_connected:
            # 让用户输入设备IP
            ip = simpledialog.askstring("ADB连接", "请输入设备IP地址:")
            if ip:
                if self.adb.connect(ip):
                    self.adb_device_id = f"{ip}:5555"
                    self.adb = ADBController(self.adb_device_id)
                    self.screen_size = self.adb.get_screen_size()
                    self.adb_connected = True
                else:
                    messagebox.showerror("连接失败", f"无法连接到 {ip}")
                    return

        if self.adb_connected:
            # 使用快速截图
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(os.path.dirname(__file__), "screenshots")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"adb_capture_{timestamp}.png")

            result = self.adb.screenshot_to_file(save_path)
            if result:
                # 获取实际截图尺寸
                img = cv2.imread(save_path)
                if img is not None:
                    actual_h, actual_w = img.shape[:2]
                    self.screen_size = (actual_w, actual_h)

                    self.screenshot_path = save_path
                    self.annotations = []
                    self.load_screenshot()
                    self.update_listbox()

                    # 添加到链
                    desc = simpledialog.askstring("截图描述", "这是哪个界面？（用于记录连贯关系）\n例如: 主界面、点击菜单后、购买物品后")
                    if desc:
                        self.screenshot_chain.append({
                            'screenshot': save_path,
                            'description': desc,
                            'resolution': (actual_w, actual_h)
                        })

                    self.status_bar.config(text=f"ADB截图已捕获 | 实际尺寸: {actual_w}x{actual_h}")
                else:
                    messagebox.showerror("错误", "截图读取失败")
            else:
                messagebox.showerror("错误", "ADB截图失败")

    def load_screenshot(self):
        if not self.screenshot_path or not os.path.exists(self.screenshot_path):
            return

        img = cv2.imread(self.screenshot_path)
        if img is None:
            messagebox.showerror("错误", f"无法加载图片: {self.screenshot_path}")
            return

        self.original_h, self.original_w = img.shape[:2]

        # 检查与实际屏幕尺寸是否匹配
        size_info = ""
        if self.screen_size:
            sw, sh = self.screen_size
            if sw != self.original_w or sh != self.original_h:
                size_info = f" | 尺寸不匹配: 标注{self.original_w}x{self.original_h} vs 屏幕{sw}x{sh}"
            else:
                size_info = f" | 尺寸匹配: {sw}x{sh}"

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 根据窗口可用空间计算实际显示尺寸（完整显示图片）
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        # 如果Canvas还没渲染，使用初始值
        if canvas_w <= 1:
            canvas_w = self.root.winfo_screenwidth() - 20
        if canvas_h <= 1:
            canvas_h = self.root.winfo_screenheight() - 200

        # 计算缩放比例，确保图片完整显示
        scale_to_fit = min(canvas_w / self.original_w, canvas_h / self.original_h, 1.0)
        self.display_w = int(self.original_w * scale_to_fit)
        self.display_h = int(self.original_h * scale_to_fit)

        # 更新canvas图片（高质量缩放）
        self.photo = ImageTk.PhotoImage(
            Image.fromarray(img_rgb).resize((self.display_w, self.display_h), Image.LANCZOS)
        )

        self.canvas.config(scrollregion=(0, 0, self.display_w, self.display_h))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

        self.redraw_annotations()
        self.status_bar.config(text=f"显示: {self.display_w}x{self.display_h} | 原始: {self.original_w}x{self.original_h}{size_info}")

    def load_screenshot_dialog(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(title="选择截图",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if path:
            self.screenshot_path = path
            self.annotations = []
            # 尝试从文件名推断尺寸
            img = cv2.imread(path)
            if img is not None:
                self.original_h, self.original_w = img.shape[:2]
                self.screen_size = (self.original_w, self.original_h)
            self.load_screenshot()
            self.update_listbox()

    def on_click(self, event):
        self.start_x = event.x
        self.start_y = event.y

        # 如果是查看坐标模式，直接输出坐标
        if self.view_coords_mode:
            scale_x = self.original_w / self.display_w
            scale_y = self.original_h / self.display_h
            x = int(event.x * scale_x)
            y = int(event.y * scale_y)
            # 输出到状态栏和控制台
            coord_text = f"坐标: ({x}, {y})"
            self.status_bar.config(text=coord_text)
            print(coord_text)
            return

        if self.current_rect:
            self.canvas.delete(self.current_rect)

    def on_drag(self, event):
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline='#00FF00', width=2, dash=(5, 5))

    def on_release(self, event):
        if abs(event.x - self.start_x) < 10 or abs(event.y - self.start_y) < 10:
            self.canvas.delete(self.current_rect)
            self.current_rect = None
            return

        scale_x = self.original_w / self.display_w
        scale_y = self.original_h / self.display_h

        x1 = int(min(self.start_x, event.x) * scale_x)
        y1 = int(min(self.start_y, event.y) * scale_y)
        x2 = int(max(self.start_x, event.x) * scale_x)
        y2 = int(max(self.start_y, event.y) * scale_y)

        # 验证坐标是否在范围内
        x1 = max(0, min(x1, self.original_w - 1))
        y1 = max(0, min(y1, self.original_h - 1))
        x2 = max(0, min(x2, self.original_w))
        y2 = max(0, min(y2, self.original_h))

        name = simpledialog.askstring("标注区域", f"区域: ({x1}, {y1}) - ({x2}, {y2})\n尺寸: {x2-x1}x{y2-y1}\n请输入功能名称:")
        if name:
            self.annotations.append((x1, y1, x2, y2, name))
            self.update_listbox()
            self.redraw_annotations()

        self.canvas.delete(self.current_rect)
        self.current_rect = None

    def redraw_annotations(self):
        if not hasattr(self, 'original_w'):
            return

        scale_x = self.display_w / self.original_w
        scale_y = self.display_h / self.original_h

        colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
        for i, (x1, y1, x2, y2, name) in enumerate(self.annotations):
            color = colors[i % len(colors)]
            dx1, dy1 = int(x1 * scale_x), int(y1 * scale_y)
            dx2, dy2 = int(x2 * scale_x), int(y2 * scale_y)
            self.canvas.create_rectangle(dx1, dy1, dx2, dy2,
                    outline=color, width=2, tags=f'rect_{i}')
            self.canvas.create_text((dx1 + 5), (dy1 + 5),
                    text=name, fill=color, anchor='nw', font=('Arial', 10, 'bold'))

    def update_listbox(self):
        self.listbox.delete(0, tk.END)
        for i, (x1, y1, x2, y2, name) in enumerate(self.annotations):
            self.listbox.insert(tk.END, f"{i+1}. ({x1}, {y1}) - ({x2}, {y2}) | {name}")

    def on_list_double_click(self, event):
        selection = self.listbox.curselection()
        if selection:
            idx = selection[0]
            name = simpledialog.askstring("修改名称", "新的功能名称:",
                    initialvalue=self.annotations[idx][4])
            if name:
                x1, y1, x2, y2, _ = self.annotations[idx]
                self.annotations[idx] = (x1, y1, x2, y2, name)
                self.update_listbox()
                self.redraw_annotations()

    def clear_annotations(self):
        if messagebox.askyesno("确认", "清除所有标注?"):
            self.annotations = []
            self.update_listbox()
            self.load_screenshot()

    def save_annotations(self):
        if not self.annotations:
            messagebox.showwarning("警告", "没有标注可保存")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.screenshot_path:
            base_dir = os.path.dirname(self.screenshot_path)
            name = os.path.splitext(os.path.basename(self.screenshot_path))[0]
        else:
            base_dir = os.path.join(os.path.dirname(__file__), "screenshots")
            name = timestamp

        json_path = os.path.join(base_dir, f"annotations_{name}.json")

        data = {
            'screenshot': self.screenshot_path,
            'resolution': {'width': self.original_w, 'height': self.original_h},
            'annotations': [
                {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'name': name}
                for x1, y1, x2, y2, name in self.annotations
            ],
            'timestamp': timestamp
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 保存带标注的截图
        if self.screenshot_path and os.path.exists(self.screenshot_path):
            img = cv2.imread(self.screenshot_path)
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
            for i, (x1, y1, x2, y2, name) in enumerate(self.annotations):
                color = colors[i % len(colors)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, name, (x1, max(y1 - 10, 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            img_path = os.path.join(base_dir, f"annotated_{name}.png")
            cv2.imwrite(img_path, img)

        messagebox.showinfo("保存成功", f"标注已保存到:\n{json_path}")
        print(f"已保存: {json_path}")

    def save_screenshot_chain(self):
        """保存截图连贯关系"""
        if not self.screenshot_chain:
            messagebox.showwarning("警告", "截图链为空，请先使用ADB截屏功能添加截图")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join(os.path.dirname(__file__), "screenshots")
        os.makedirs(base_dir, exist_ok=True)
        json_path = os.path.join(base_dir, f"screenshot_chain_{timestamp}.json")

        # 构建连贯关系数据
        chain_data = {
            'chain': [],
            'timestamp': timestamp
        }

        for i, item in enumerate(self.screenshot_chain):
            chain_data['chain'].append({
                'step': i + 1,
                'screenshot': item['screenshot'],
                'description': item['description'],
                'resolution': item['resolution']
            })

        # 保存为可读的流程说明
        flow_path = os.path.join(base_dir, f"flow_{timestamp}.txt")
        with open(flow_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("截图连贯关系 / Screenshot Continuity Flow\n")
            f.write("=" * 60 + "\n\n")
            for i, item in enumerate(self.screenshot_chain):
                f.write(f"[{i+1}] {item['description']}\n")
                f.write(f"    文件: {os.path.basename(item['screenshot'])}\n")
                f.write(f"    尺寸: {item['resolution'][0]}x{item['resolution'][1]}\n\n")

            f.write("-" * 60 + "\n")
            f.write("界面跳转关系:\n")
            for i in range(len(self.screenshot_chain) - 1):
                curr = self.screenshot_chain[i]
                next_item = self.screenshot_chain[i + 1]
                f.write(f"  {curr['description']} -> {next_item['description']}\n")

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(chain_data, f, ensure_ascii=False, indent=2)

        messagebox.showinfo("保存成功", f"截图链已保存:\n{json_path}\n{flow_path}")
        print(f"已保存截图链: {json_path}")

    def show_chain_dialog(self):
        """显示截图链对话框"""
        if not self.screenshot_chain:
            messagebox.showinfo("截图链", "截图链为空")
            return

        chain_win = tk.Toplevel(self.root)
        chain_win.title("截图连贯关系")
        chain_win.geometry("600x400")

        tk.Label(chain_win, text="当前截图链:", font=('Microsoft YaHei', 12, 'bold')).pack(anchor='w', padx=10, pady=5)

        text = scrolledtext.ScrolledText(chain_win, font=('Consolas', 10), height=20)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        for i, item in enumerate(self.screenshot_chain):
            text.insert(tk.END, f"[{i+1}] {item['description']}\n", 'bold')
            text.insert(tk.END, f"    文件: {os.path.basename(item['screenshot'])}\n")
            text.insert(tk.END, f"    尺寸: {item['resolution'][0]}x{item['resolution'][1]}\n\n")

        text.tag_config('bold', font=('Consolas', 10, 'bold'))

        btn_frame = tk.Frame(chain_win)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(btn_frame, text="添加当前截图到链", command=self.add_current_to_chain,
                 bg='#9C27B0', fg='white').pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="清空链", command=self.clear_chain,
                 bg='#f44336', fg='white').pack(side=tk.LEFT, padx=5)

    def add_current_to_chain(self):
        """将当前截图添加到链"""
        if self.screenshot_path:
            img = cv2.imread(self.screenshot_path)
            if img is not None:
                h, w = img.shape[:2]
                desc = simpledialog.askstring("截图描述", "这是哪个界面？")
                if desc:
                    self.screenshot_chain.append({
                        'screenshot': self.screenshot_path,
                        'description': desc,
                        'resolution': (w, h)
                    })
                    messagebox.showinfo("添加成功", f"已添加到链: {desc}")
            else:
                messagebox.showerror("错误", "无法读取当前截图")
        else:
            messagebox.showwarning("警告", "当前没有加载截图")

    def clear_chain(self):
        if messagebox.askyesno("确认", "清空截图链?"):
            self.screenshot_chain = []
            messagebox.showinfo("已清空", "截图链已清空")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    app = AnnotationTool(path)
    app.run()
