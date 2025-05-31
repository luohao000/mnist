import tkinter as tk
from tkinter import ttk, messagebox
from torchvision import datasets, transforms
import random
import matplotlib
matplotlib.use('TkAgg')  # 确保使用正确的后端
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class MNISTImageLoader:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MNIST图片加载器")
        self.root.geometry("800x700")
        
        # 设置窗口关闭协议
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 初始化变量
        self.train_dataset = None
        self.test_dataset = None
        self.current_dataset = None
        self.train_size = 0
        self.test_size = 0
        self.dataset_type = "训练集"
        
        # 添加标签索引缓存
        self.train_label_indices = {}
        self.test_label_indices = {}
        self.current_label_indices = {}
        
        # 加载MNIST数据集
        self.load_mnist_data()
        
        # 当前显示的图片信息
        self.current_image = None
        self.current_label = None
        self.current_index = 0
        
        # 创建界面
        if self.current_dataset is not None:
            self.setup_ui()
        else:
            messagebox.showerror("错误", "无法加载数据集，程序将退出")
            self.root.destroy()
            return
        
    def load_mnist_data(self):
        """加载MNIST数据集"""
        try:
            # 数据变换（只转换为张量）
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            
            # 加载训练集和测试集
            self.train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            self.test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
            
            # 获取数据集大小
            self.train_size = len(self.train_dataset)
            self.test_size = len(self.test_dataset)
            
            # 默认使用训练集
            self.current_dataset = self.train_dataset
            self.dataset_type = "训练集"
            
            # 预建立标签索引缓存
            print("正在建立标签索引缓存...")
            self._build_label_indices()
            
            print(f"MNIST数据集加载成功!")
            print(f"训练集大小: {self.train_size}")
            print(f"测试集大小: {self.test_size}")
            
        except Exception as e:
            print(f"加载MNIST数据集失败: {str(e)}")
            messagebox.showerror("错误", f"加载MNIST数据集失败:\n{str(e)}")
            self.current_dataset = None
            
    def _build_label_indices(self):
        """建立标签索引缓存"""
        try:
            # 为训练集建立索引
            for digit in range(10):
                self.train_label_indices[digit] = []
                self.test_label_indices[digit] = []
            
            # 遍历训练集
            for i in range(len(self.train_dataset)):
                _, label = self.train_dataset[i]
                self.train_label_indices[label].append(i)
            
            # 遍历测试集
            for i in range(len(self.test_dataset)):
                _, label = self.test_dataset[i]
                self.test_label_indices[label].append(i)
            
            # 设置当前标签索引
            self.current_label_indices = self.train_label_indices
            
            print("标签索引缓存建立完成!")
            
        except Exception as e:
            print(f"建立标签索引失败: {str(e)}")
            
    def setup_ui(self):
        """设置用户界面"""
        # 标题
        title_label = tk.Label(self.root, text="MNIST手写数字图片加载器", 
                              font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # 数据集选择框架
        dataset_frame = tk.Frame(self.root)
        dataset_frame.pack(pady=10)
        
        tk.Label(dataset_frame, text="数据集选择:", font=('Arial', 12)).pack(side=tk.LEFT)
        
        self.dataset_var = tk.StringVar(value="训练集")
        dataset_combo = ttk.Combobox(dataset_frame, textvariable=self.dataset_var,
                                   values=["训练集", "测试集"], state="readonly", width=10)
        dataset_combo.pack(side=tk.LEFT, padx=5)
        dataset_combo.bind("<<ComboboxSelected>>", self.change_dataset)
        
        # 数据集信息
        self.info_label = tk.Label(self.root, 
                                  text=f"当前: {self.dataset_type} (共 {len(self.current_dataset)} 张图片)",
                                  font=('Arial', 10))
        self.info_label.pack()
        
        # 控制按钮框架
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=15)
        
        # 随机加载按钮
        random_btn = tk.Button(control_frame, text="随机加载", command=self.load_random_image,
                              bg='blue', fg='white', font=('Arial', 11), width=10)
        random_btn.pack(side=tk.LEFT, padx=5)
        
        # 上一张按钮
        prev_btn = tk.Button(control_frame, text="上一张", command=self.load_previous_image,
                            bg='orange', fg='white', font=('Arial', 11), width=10)
        prev_btn.pack(side=tk.LEFT, padx=5)
        
        # 下一张按钮
        next_btn = tk.Button(control_frame, text="下一张", command=self.load_next_image,
                            bg='green', fg='white', font=('Arial', 11), width=10)
        next_btn.pack(side=tk.LEFT, padx=5)
        
        # 指定索引加载框架
        index_frame = tk.Frame(self.root)
        index_frame.pack(pady=10)
        
        tk.Label(index_frame, text="指定索引:", font=('Arial', 10)).pack(side=tk.LEFT)
        
        self.index_var = tk.StringVar()
        index_entry = tk.Entry(index_frame, textvariable=self.index_var, width=10)
        index_entry.pack(side=tk.LEFT, padx=5)
        
        load_index_btn = tk.Button(index_frame, text="加载", command=self.load_by_index,
                                  bg='purple', fg='white', font=('Arial', 10))
        load_index_btn.pack(side=tk.LEFT, padx=5)
        
        # 数字筛选框架
        filter_frame = tk.Frame(self.root)
        filter_frame.pack(pady=10)
        
        tk.Label(filter_frame, text="筛选数字:", font=('Arial', 10)).pack(side=tk.LEFT)
        
        self.filter_var = tk.StringVar(value="全部")
        filter_combo = ttk.Combobox(filter_frame, textvariable=self.filter_var,
                                  values=["全部", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                                  state="readonly", width=8)
        filter_combo.pack(side=tk.LEFT, padx=5)
        
        filter_btn = tk.Button(filter_frame, text="随机加载该数字", command=self.load_filtered_random,
                              bg='red', fg='white', font=('Arial', 10))
        filter_btn.pack(side=tk.LEFT, padx=5)
        
        # 图片显示区域
        self.setup_image_display()
        
        # 图片信息显示
        info_frame = tk.Frame(self.root)
        info_frame.pack(pady=10)
        
        self.image_info_label = tk.Label(info_frame, text="请加载图片", 
                                        font=('Arial', 12), fg='blue')
        self.image_info_label.pack()
        
        # 像素值显示区域
        pixel_frame = tk.Frame(self.root)
        pixel_frame.pack(pady=5)
        
        tk.Label(pixel_frame, text="像素值范围:", font=('Arial', 10)).pack(side=tk.LEFT)
        self.pixel_info_label = tk.Label(pixel_frame, text="", font=('Arial', 10))
        self.pixel_info_label.pack(side=tk.LEFT, padx=5)
        
        # 加载第一张图片
        self.load_image_by_index(0)
        
    def setup_image_display(self):
        """设置图片显示区域"""
        # 创建图片显示框架
        image_frame = tk.Frame(self.root, relief=tk.SUNKEN, borderwidth=2)
        image_frame.pack(pady=15)
        
        # 使用matplotlib显示图片（更好的灰度显示效果）
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_title("MNIST图片", fontsize=14)
        self.ax.axis('off')
        
        # 将matplotlib图嵌入tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, image_frame)
        self.canvas.get_tk_widget().pack()
        
    def change_dataset(self, event=None):
        """切换数据集"""
        dataset_type = self.dataset_var.get()
        
        if dataset_type == "训练集":
            self.current_dataset = self.train_dataset
            self.dataset_type = "训练集"
            self.current_label_indices = self.train_label_indices
        else:
            self.current_dataset = self.test_dataset
            self.dataset_type = "测试集"
            self.current_label_indices = self.test_label_indices
            
        # 更新信息
        self.info_label.config(text=f"当前: {self.dataset_type} (共 {len(self.current_dataset)} 张图片)")
        
        # 重置索引并加载第一张图片
        self.current_index = 0
        self.load_image_by_index(0)
        
    def load_image_by_index(self, index):
        """根据索引加载图片"""
        try:
            if 0 <= index < len(self.current_dataset):
                # 获取图片和标签
                image, label = self.current_dataset[index]
                
                # 转换为numpy数组用于显示
                image_np = image.squeeze().numpy()  # 移除batch维度
                
                # 更新当前图片信息
                self.current_image = image_np
                self.current_label = label
                self.current_index = index
                
                # 显示图片
                self.display_image(image_np, label, index)
                
            else:
                messagebox.showwarning("警告", f"索引超出范围! 有效范围: 0 - {len(self.current_dataset)-1}")
                
        except Exception as e:
            messagebox.showerror("错误", f"加载图片失败:\n{str(e)}")
            
    def display_image(self, image_np, label, index):
        """显示图片"""
        # 清除之前的图片
        self.ax.clear()
        
        # 显示图片（灰度图，白色背景黑色数字）
        self.ax.imshow(image_np, cmap='gray', vmin=0, vmax=1)
        self.ax.set_title(f"Label: {label} (Index: {index})", fontsize=16, fontweight='bold')
        self.ax.axis('off')
        
        # 刷新画布
        self.canvas.draw()
        
        # 更新图片信息
        pixel_min = image_np.min()
        pixel_max = image_np.max()
        pixel_mean = image_np.mean()
        
        self.image_info_label.config(
            text=f"Label: {label} | Index: {index} | 数据集: {self.dataset_type}"
        )
        
        self.pixel_info_label.config(
            text=f"最小值: {pixel_min:.3f} | 最大值: {pixel_max:.3f} | 平均值: {pixel_mean:.3f}"
        )
        
    def load_random_image(self):
        """加载随机图片"""
        random_index = random.randint(0, len(self.current_dataset) - 1)
        self.load_image_by_index(random_index)
        
    def load_previous_image(self):
        """加载上一张图片"""
        if self.current_index > 0:
            self.load_image_by_index(self.current_index - 1)
        else:
            messagebox.showinfo("提示", "已经是第一张图片!")
            
    def load_next_image(self):
        """加载下一张图片"""
        if self.current_index < len(self.current_dataset) - 1:
            self.load_image_by_index(self.current_index + 1)
        else:
            messagebox.showinfo("提示", "已经是最后一张图片!")
            
    def load_by_index(self):
        """根据用户输入的索引加载图片"""
        try:
            index = int(self.index_var.get())
            self.load_image_by_index(index)
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字索引!")
            
    def load_filtered_random(self):
        """加载指定数字的随机图片"""
        filter_digit = self.filter_var.get()
        
        if filter_digit == "全部":
            self.load_random_image()
            return
            
        try:
            target_digit = int(filter_digit)
            
            # 使用缓存的索引
            if target_digit in self.current_label_indices and self.current_label_indices[target_digit]:
                matching_indices = self.current_label_indices[target_digit]
                # 随机选择一个匹配的索引
                random_index = random.choice(matching_indices)
                self.load_image_by_index(random_index)
                print(f"加载数字 {target_digit}，索引: {random_index}")
            else:
                messagebox.showinfo("提示", f"在{self.dataset_type}中未找到数字 {target_digit}")
                
        except ValueError:
            messagebox.showerror("错误", "无效的数字选择!")
        except Exception as e:
            print(f"筛选加载错误: {str(e)}")
            messagebox.showerror("错误", f"筛选加载失败:\n{str(e)}")
            
    def setup_image_display(self):
        """设置图片显示区域"""
        # 创建图片显示框架
        image_frame = tk.Frame(self.root, relief=tk.SUNKEN, borderwidth=2)
        image_frame.pack(pady=15)
        
        # 使用matplotlib显示图片（更好的灰度显示效果）
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_title("MNIST图片", fontsize=14)
        self.ax.axis('off')
        
        # 将matplotlib图嵌入tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, image_frame)
        self.canvas.get_tk_widget().pack()
        
    def on_closing(self):
        """窗口关闭时的清理工作"""
        try:
            # 关闭matplotlib图形
            if hasattr(self, 'fig'):
                plt.close(self.fig)
            
            # 关闭所有matplotlib图形
            plt.close('all')
            
            # 销毁tkinter窗口
            self.root.quit()
            self.root.destroy()
            
        except Exception as e:
            print(f"关闭程序时出错: {str(e)}")
        finally:
            # 强制退出
            import sys
            sys.exit(0)
            
    def run(self):
        """运行程序"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()
        except Exception as e:
            print(f"程序运行出错: {str(e)}")
            self.on_closing()

if __name__ == "__main__":
    app = MNISTImageLoader()
    app.run()