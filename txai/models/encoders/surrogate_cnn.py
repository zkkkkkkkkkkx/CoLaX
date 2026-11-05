"""
从 surrogate_data_based_machine_learning 项目迁移的CNN架构

原项目位置：surrogate_data_based_machine_learning-main/.../01_create_fit_evaluate_save_dl_model.py
原框架：TensorFlow/Keras
迁移框架：PyTorch

包含三个模型：
1. SurrogateCNN: 基础CNN模型（2层卷积）
2. ThreeHeadCNN: 三头CNN模型
3. SurrogateLSTM: LSTM模型（单向，单层）
"""

from torch import nn
import torch
import torch.nn.functional as F


class SurrogateCNN(nn.Module):
    """
    从surrogate项目迁移的基础CNN模型
    
    原TensorFlow架构：
        Conv1D(64, 3, activation='relu')
        Conv1D(64, 3, activation='relu')
        Dropout(0.5)
        MaxPooling1D()
        Flatten()
        Dense(128, activation='relu')
        Dense(1, activation='sigmoid')  # 二分类
    
    迁移后的PyTorch架构：
        - 支持多类分类（n_classes）
        - 使用AdaptiveAvgPool1d适应不同序列长度
        - 兼容Colax的输入格式
    """
    def __init__(self, d_inp, n_classes, dim=64):
        super().__init__()
        self.d_inp = d_inp
        self.n_classes = n_classes
        self.dim = dim
        
        # 卷积编码器（对应原项目的2层Conv1D）
        self.encoder = nn.Sequential(
            nn.Conv1d(d_inp, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool1d(1),  # 适应不同序列长度
            nn.Flatten(),
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x, _times, get_embedding=False, captum_input=False, show_sizes=False):
        """
        前向传播
        
        输入格式：
            - captum_input=True: (batch, time, channels)
            - captum_input=False: (time, batch, channels) 或 (batch, time, channels)
        输出：
            - get_embedding=False: (batch, n_classes)
            - get_embedding=True: (output, embedding)
        """
        # 处理输入维度（兼容Colax的输入格式）
        if captum_input:
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            # (batch, time, channels) -> (batch, channels, time)
            x = x.permute(0, 2, 1)
        else:
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            # (time, batch, channels) -> (batch, channels, time)
            x = x.permute(1, 2, 0)

        # 确保序列长度足够（至少4，以便MaxPool1d能工作）
        if x.shape[-1] < 4:
            x = F.pad(x, (0, 4 - x.shape[-1]), mode="constant", value=0)

        # 编码
        embedding = self.encoder(x)
        
        # 分类
        out = self.classifier(embedding)

        if get_embedding:
            return out, embedding
        else:
            return out


class ThreeHeadCNN(nn.Module):
    """
    从surrogate项目迁移的三头CNN模型
    
    原架构：三个并行的CNN分支（kernel_size=3,5,11），合并后分类
    特点：能够捕获不同时间尺度的特征
    """
    def __init__(self, d_inp, n_classes):
        super().__init__()
        self.d_inp = d_inp
        self.n_classes = n_classes
        
        # Head 1: kernel_size=3, filters=32
        self.head1 = nn.Sequential(
            nn.Conv1d(d_inp, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool1d(kernel_size=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # Head 2: kernel_size=5, filters=32
        self.head2 = nn.Sequential(
            nn.Conv1d(d_inp, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool1d(kernel_size=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # Head 3: kernel_size=11, filters=64
        self.head3 = nn.Sequential(
            nn.Conv1d(d_inp, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool1d(kernel_size=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # 合并层
        self.classifier = nn.Sequential(
            nn.Linear(32 + 32 + 64, 128),  # 合并三个头的输出
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x, _times, get_embedding=False, captum_input=False, show_sizes=False):
        """前向传播"""
        # 输入格式转换
        if captum_input:
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            x = x.permute(0, 2, 1)
        else:
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            x = x.permute(1, 2, 0)

        # 确保序列长度足够
        if x.shape[-1] < 4:
            x = F.pad(x, (0, 4 - x.shape[-1]), mode="constant", value=0)

        # 三个分支并行处理
        h1 = self.head1(x)  # (batch, 32)
        h2 = self.head2(x)  # (batch, 32)
        h3 = self.head3(x)  # (batch, 64)
        
        # 合并
        merged = torch.cat([h1, h2, h3], dim=1)  # (batch, 128)
        
        # 分类
        out = self.classifier(merged)

        if get_embedding:
            return out, merged
        else:
            return out


class SurrogateLSTM(nn.Module):
    """
    从surrogate项目迁移的LSTM模型
    
    原架构：单向LSTM(128单元，1层) -> Dropout -> Dense(128) -> Dense(1, sigmoid)
    特点：简单高效，适合序列分类
    """
    def __init__(self, d_inp, n_classes, dim=128):
        super().__init__()
        self.d_inp = d_inp
        self.n_classes = n_classes
        self.dim = dim
        
        # LSTM编码器（单向，单层）
        self.encoder = nn.LSTM(
            d_inp,
            dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,  # 单向LSTM（与原项目一致）
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x, _times, get_embedding=False, captum_input=False, show_sizes=False):
        """前向传播"""
        # 输入格式转换
        if not captum_input:
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            # (time, batch, channels) -> (batch, time, channels)
            x = x.permute(1, 0, 2)
        elif len(x.shape) == 2:
            x = x.unsqueeze(0)

        # LSTM编码
        embedding, _ = self.encoder(x)  # (batch, time, dim)
        
        # 取最后一个时间步的输出（与原项目一致）
        embedding = embedding[:, -1, :]  # (batch, dim)
        
        # 分类
        out = self.classifier(embedding)

        if get_embedding:
            return out, embedding
        else:
            return out


# 使用示例
if __name__ == "__main__":
    print("测试 SurrogateCNN 模型...")
    model = SurrogateCNN(d_inp=1, n_classes=4, dim=64)
    x = torch.randn(100, 32, 1)  # (time=100, batch=32, channels=1)
    output = model(x, None)
    print(f"SurrogateCNN 输出形状: {output.shape}")  # (32, 4)
    
    print("\n测试 ThreeHeadCNN 模型...")
    model = ThreeHeadCNN(d_inp=1, n_classes=4)
    output = model(x, None)
    print(f"ThreeHeadCNN 输出形状: {output.shape}")  # (32, 4)
    
    print("\n测试 SurrogateLSTM 模型...")
    model = SurrogateLSTM(d_inp=1, n_classes=4, dim=128)
    output = model(x, None)
    print(f"SurrogateLSTM 输出形状: {output.shape}")  # (32, 4)
    
    print("\n✅ 所有模型测试通过！")



































