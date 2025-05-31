import torch
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
from PIL import Image, ImageDraw
import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 模型调用
from mnist_cnn import CNN, channel_nums

class CNNRecognizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CNN手写数字识别器")
        self.root.geometry("500x650")
          # 加载模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN(channel_nums=channel_nums)
        try:
            self.model.load_state_dict(torch.load("cnn_model.pth", map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            model_status = f"CNN模型已加载 (设备: {self.device})"
        except FileNotFoundError:
            model_status = "警告: 未找到 cnn_model.pth 文件"
            messagebox.showwarning("模型文件", "未找到 cnn_model.pth 文件，请先训练CNN模型")
        
        # 标题
        title_label = tk.Label(self.root, text="CNN手写数字识别器", 
                              font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # 模型状态
        status_label = tk.Label(self.root, text=model_status, 
                               font=('Arial', 10), fg='green' if '已加载' in model_status else 'red')
        status_label.pack()
        
        # 创建画布
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(pady=20)
        
        tk.Label(canvas_frame, text="在黑色画布上绘制数字:", font=('Arial', 12)).pack()
        self.canvas = tk.Canvas(canvas_frame, width=280, height=280, bg='black', 
                               relief=tk.SUNKEN, borderwidth=2)
        self.canvas.pack(pady=10)
        
        # 绑定鼠标事件
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)
        
        # 创建PIL图像用于保存
        self.image = Image.new('L', (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
        
        # 控制按钮
        button_frame1 = tk.Frame(self.root)
        button_frame1.pack(pady=10)
        
        clear_btn = tk.Button(button_frame1, text="清除画布", command=self.clear_canvas,
                             bg='red', fg='white', font=('Arial', 11), width=12)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        recognize_btn = tk.Button(button_frame1, text="识别数字", command=self.recognize_digit,
                                 bg='blue', fg='white', font=('Arial', 11), width=12)
        recognize_btn.pack(side=tk.LEFT, padx=5)
        
        save_btn = tk.Button(button_frame1, text="保存张量", command=self.save_tensor,
                            bg='green', fg='white', font=('Arial', 11), width=12)
        save_btn.pack(side=tk.LEFT, padx=5)
        
        # 第二行按钮
        button_frame2 = tk.Frame(self.root)
        button_frame2.pack(pady=5)
        
        preview_btn = tk.Button(button_frame2, text="预览28x28", command=self.preview_tensor,
                               bg='purple', fg='white', font=('Arial', 11), width=12)
        preview_btn.pack(side=tk.LEFT, padx=5)
        
        test_btn = tk.Button(button_frame2, text="测试CNN", command=self.test_model,
                            bg='orange', fg='white', font=('Arial', 11), width=12)
        test_btn.pack(side=tk.LEFT, padx=5)
        
        # 预测结果显示
        result_frame = tk.Frame(self.root)
        result_frame.pack(pady=15)
        
        tk.Label(result_frame, text="CNN识别结果:", font=('Arial', 12, 'bold')).pack()
        self.result_label = tk.Label(result_frame, text="请绘制数字后点击识别", 
                                    font=('Arial', 24, 'bold'), fg='blue',
                                    relief=tk.SUNKEN, borderwidth=2, width=20, height=2)
        self.result_label.pack(pady=5)
        
        # 概率分布显示
        prob_frame = tk.Frame(self.root)
        prob_frame.pack(pady=10)
        
        tk.Label(prob_frame, text="各数字概率:", font=('Arial', 10)).pack()
        self.prob_text = tk.Text(prob_frame, height=3, width=60, font=('Arial', 9))
        self.prob_text.pack()
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("CNN识别器准备就绪")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.brush_size = 15
        
    def paint(self, event):
        # 在tkinter画布上绘制
        x1, y1 = (event.x - self.brush_size), (event.y - self.brush_size)
        x2, y2 = (event.x + self.brush_size), (event.y + self.brush_size)
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
        
        # 在PIL图像上绘制
        self.draw.ellipse([x1, y1, x2, y2], fill=255)
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new('L', (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="请绘制数字后点击识别")
        self.prob_text.delete(1.0, tk.END)
        self.status_var.set("画布已清除")
        
    def convert_to_mnist_format(self):
        # 调整图像大小到28x28
        resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # 转换为numpy数组
        tensor = np.array(resized, dtype=np.float32)
        
        # 归一化到0-1范围
        tensor = tensor / 255.0
        
        # 如果需要标准化（与训练时保持一致）
        # tensor = (tensor - 0.1307) / 0.3081  # MNIST标准化参数
        
        return tensor
        
    def recognize_digit(self):
        try:
            # 转换为MNIST格式
            tensor = self.convert_to_mnist_format()
            
            # CNN需要保持2D图像结构，形状为 [batch_size, channels, height, width]
            torch_tensor = torch.tensor(tensor).unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, 28, 28]
            
            # 进行预测
            with torch.no_grad():
                output = self.model(torch_tensor)
                probabilities = torch.softmax(output, dim=1)
                prediction = torch.argmax(output, dim=1).item()
                confidence = probabilities[0][prediction].item()
                
            # 显示结果
            self.result_label.config(text=f"CNN识别: {prediction}\n置信度: {confidence:.2%}")
            
            # 显示所有概率
            prob_text = "各数字概率: "
            for i in range(10):
                prob = probabilities[0][i].item()
                prob_text += f"{i}:{prob:.1%} "
            
            self.prob_text.delete(1.0, tk.END)
            self.prob_text.insert(1.0, prob_text)
            
            self.status_var.set(f"CNN识别完成 - 预测数字: {prediction} (置信度: {confidence:.1%})")
            
        except Exception as e:
            messagebox.showerror("识别错误", f"CNN识别过程中出现错误:\n{str(e)}")
            self.status_var.set("CNN识别失败")
            
    def preview_tensor(self):
        tensor = self.convert_to_mnist_format()
        
        # 创建预览窗口
        preview_window = tk.Toplevel(self.root)
        preview_window.title("CNN输入预览 - 28x28")
        preview_window.geometry("350x400")
        
        # 放大显示28x28图像
        preview_canvas = tk.Canvas(preview_window, width=280, height=280, bg='white')
        preview_canvas.pack(pady=10)
        
        # 绘制放大的像素
        pixel_size = 10
        for i in range(28):
            for j in range(28):
                intensity = int(tensor[i, j] * 255)
                color = f'#{intensity:02x}{intensity:02x}{intensity:02x}'
                x1, y1 = j * pixel_size, i * pixel_size
                x2, y2 = x1 + pixel_size, y1 + pixel_size
                preview_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=color)
        
        # 显示张量信息
        info_label = tk.Label(preview_window, 
                             text=f"CNN输入形状: [1, 1, {tensor.shape[0]}, {tensor.shape[1]}]\n"
                                  f"像素值范围: {tensor.min():.3f} - {tensor.max():.3f}",
                             font=('Arial', 10))
        info_label.pack()
        
    def save_tensor(self):
        tensor = self.convert_to_mnist_format()
        
        # 直接保存在当前目录下
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cnn_input_{timestamp}.npy"
        
        np.save(filename, tensor)
        self.status_var.set(f"CNN输入张量已保存到: {filename}")
        messagebox.showinfo("保存成功", f"28x28张量已保存到:\n{filename}")
        
    def test_model(self):
        try:
            # 在新窗口中显示测试进度
            test_window = tk.Toplevel(self.root)
            test_window.title("CNN模型测试")
            test_window.geometry("350x250")
            
            progress_label = tk.Label(test_window, text="正在加载MNIST测试数据...", 
                                    font=('Arial', 12))
            progress_label.pack(pady=20)
            
            progress_bar = ttk.Progressbar(test_window, mode='indeterminate')
            progress_bar.pack(pady=10)
            progress_bar.start()
            
            result_text = tk.Text(test_window, height=10, width=40)
            result_text.pack(pady=10)
            
            # 更新界面
            test_window.update()
            
            # 加载MNIST测试数据（与训练时保持一致的预处理）
            transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))  # 如果训练时使用了标准化
            ])
            test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            
            progress_label.config(text="正在测试CNN模型准确率...")
            test_window.update()
            
            # 测试模型
            self.model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(test_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    # CNN不需要展平，保持 [batch_size, 1, 28, 28] 形状
                    output = self.model(data)
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
                    
                    if batch_idx % 20 == 0:
                        progress_label.config(text=f"CNN测试进度: {batch_idx * 64}/{len(test_dataset)}")
                        test_window.update()
            
            accuracy = correct / total
            
            progress_bar.stop()
            progress_label.config(text="CNN测试完成!")
            
            result_info = f"""CNN模型测试结果:
            
模型类型: 卷积神经网络 (CNN)
总测试样本: {total}
正确预测: {correct}
测试准确率: {accuracy:.4f} ({accuracy*100:.2f}%)

模型性能: {'优秀' if accuracy > 0.98 else '良好' if accuracy > 0.95 else '一般'}

CNN优势: 
- 空间特征提取
- 平移不变性
- 参数共享
"""
            
            result_text.insert(1.0, result_info)
            self.status_var.set(f"CNN模型测试完成 - 准确率: {accuracy:.2%}")
            
        except Exception as e:
            messagebox.showerror("测试错误", f"CNN模型测试过程中出现错误:\n{str(e)}")
            
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = CNNRecognizer()
    app.run()