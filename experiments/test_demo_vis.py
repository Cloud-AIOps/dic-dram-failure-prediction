import torch
import torch.nn as nn
from torchsummary import summary
from torchviz import make_dot

# 定义一个简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# 创建模型实例
model = SimpleCNN()
model.to('cuda')
# 使用torchsummary库来查看模型摘要
summary(model, (3, 32, 32))  # 输入图片大小为(3, 32, 32)

# 使用torchviz库来可视化模型图
x = torch.randn(1, 3, 32, 32)
x = x.to('cuda')
y = model(x)
dot = make_dot(y, params=dict(model.named_parameters()))
dot.render("simple_cnn_model", format="png")  # 将模型图保存为PNG文件
