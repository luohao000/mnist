import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 可调参数
# 28*28 = 784
layer_sizes = [784, 64, 64, 10]  # 输入层+隐藏层+输出层，每层神经元数量
batch_size = 64
epochs = 10
lr = 0.001


class FlexibleFNN(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # 支持输入为 (batch_size, 1, 28, 28) 的张量，自动展平
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        return self.net(x)


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
    model = FlexibleFNN(layer_sizes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_accuracy = 0.0  # 用于保存最佳准确率
    best_model_path = "fnn_model.pth"  # 保存最佳模型的路径

    print("Starting FNN training...")

    # 训练
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
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
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with accuracy: {best_accuracy:.4f}")

    print(f"Training complete. Best accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    main()
