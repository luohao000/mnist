import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 可调参数
batch_size = 64
epochs = 10
lr = 0.001

channel_nums = [32, 32]  # 中间的通道数，卷积池化 2 次

class CNN(nn.Module):
    def __init__(self, channel_nums=channel_nums):
        super(CNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, channel_nums[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channel_nums[0], channel_nums[1], kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)  # 2x2池化，尺寸减半

        # 全连接层
        self.fc1 = nn.Linear(channel_nums[1] * 7 * 7, 64)  # 7x7是池化后特征图的尺寸
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        # 第一个卷积块：卷积 -> 激活 -> 池化
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 展平
        x = x.view(-1, channel_nums[1] * 7 * 7)
        
        # 全连接层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x


def main():
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
      # 数据加载及预处理
    transform = transforms.Compose([
        # transforms.ToTensor() 会把PIL图片或numpy数组转换为PyTorch张量，
        # 并且把像素值从[0,255]缩放到[0,1]的float32类型，形状变为 (1, 28, 28)
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 模型、损失、优化器
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_accuracy = 0.0  # 用于保存最佳准确率
    best_model_path = "cnn_model.pth"  # 保存最佳模型的路径

    print("Starting CNN training...")
    
    # 训练
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pred = output.argmax(dim=1)
            correct_train += (pred == target).sum().item()
            total_train += target.size(0)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        # 测试
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
                
        test_accuracy = correct / total
        
        print(f"Test Accuracy: {test_accuracy:.4f}")

        # 保存最佳模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with accuracy: {best_accuracy:.4f}")

    print(f"Training complete. Best accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()