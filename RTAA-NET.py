import pandas as pd
import numpy as np
import requests

from math import radians, sin, cos, sqrt, atan2
from scipy.signal import savgol_filter
from typing import List, Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from sklearn.metrics import f1_score, jaccard_score
import time



# ====================== Enhanced Embedded Data ======================
def generate_synthetic_data(base_data, num_samples=200):
    new_data = []
    patterns = [
        {'speed_delta': 2, 'flags': [1, 0, 0, 0, 0, 0]},  # Acceleration
        {'speed_delta': -3, 'flags': [0, 1, 0, 0, 0, 0]},  # Braking
        {'speed_delta': 0, 'flags': [0, 0, 1, 1, 0, 0]},  # Neutral slide
        {'speed_delta': 5, 'flags': [0, 0, 0, 0, 1, 0]},  # Fatigue
    ]

    for i in range(num_samples):
        pattern = patterns[i % len(patterns)]
        modified = base_data[i % len(base_data)].copy()
        modified["time"] = f"2023-01-01 {8 + i // 60:02d}:{i % 60:02d}:00"  # 修正时间格式
        modified["speed"] = np.clip(modified["speed"] + pattern['speed_delta'], 0, 120)
        modified["latitude"] += np.random.uniform(-0.001, 0.001)
        modified["longitude"] += np.random.uniform(-0.001, 0.001)

        flags = ['isRapidlySpeedup', 'isRapidlySlowdown', 'isNeutralSlide',
                 'isOverspeed', 'isFatigueDriving', 'isHthrottleStop']
        for flag, value in zip(flags, pattern['flags']):
            modified[flag] = value

        new_data.append(modified)
    return new_data


# ====================== GeoAnalyzer 类定义 ======================
class GeoAnalyzer:
    def __init__(self, data: List[Dict]):
        self.df = self._prepare_dataframe(data)
        self._elevation_data = None

    def _prepare_dataframe(self, data: List[Dict]) -> pd.DataFrame:
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'])
        return df

    def _batch_get_elevation(self) -> bool:
        try:
            locations = [{"latitude": row.latitude, "longitude": row.longitude}
                         for _, row in self.df.iterrows()]
            response = requests.post("https://api.open-elevation.com/api/v1/lookup",
                                     json={"locations": locations},
                                     timeout=10)
            if response.status_code == 200:
                self.df['elevation'] = [r['elevation'] for r in response.json()['results']]
                return True
        except Exception as e:
            print(f"高程API请求失败: {str(e)}，启用模拟数据")
            self._generate_synthetic_elevation()
            return False

    def _generate_synthetic_elevation(self):
        base = 4.5
        noise = np.random.normal(0, 0.3, len(self.df))
        self.df['elevation'] = (base + np.cumsum(noise)).round(2)

    def calculate_terrain_features(self):
        if not self._batch_get_elevation():
            print("注意：当前使用模拟高程数据")

        # 计算坡度
        self.df['slope'] = self.df['elevation'].diff().fillna(0) * 100

        # 计算曲率
        self.df['curvature'] = self.df['latitude'].diff().fillna(0) + self.df['longitude'].diff().fillna(0)
        return self

    @property
    def results(self):
        return self.df


# ==================== 数据预处理 ====================
def preprocess_data(raw_data: List[Dict]) -> Dict[str, np.ndarray]:
    analyzer = GeoAnalyzer(raw_data).calculate_terrain_features()
    df = analyzer.results

    # 特征工程
    driver_features = df[['isRapidlySpeedup', 'isRapidlySlowdown',
                          'isNeutralSlide', 'isFatigueDriving', 'isHthrottleStop']].values
    vehicle_features = df[['speed', 'direction']].values
    topo_features = df[['slope', 'curvature']].values

    # 标签生成
    labels = np.random.randint(0, 6, size=len(df))  # 6个类别

    # 时间序列窗口
    seq_length = 5
    samples = []
    for i in range(len(df) - seq_length):
        samples.append((
            driver_features[i:i + seq_length],
            vehicle_features[i:i + seq_length],
            topo_features[i:i + seq_length],
            labels[i + seq_length]
        ))

    return {
        'driver_status': np.array([s[0] for s in samples]),
        'vehicle_status': np.array([s[1] for s in samples]),
        'topographic': np.array([s[2] for s in samples]),
        'labels': np.array([s[3] for s in samples])
    }


# ==================== 模型架构 ====================
class EnhancedAttention(nn.Module):
    # def __init__(self, input_dim):
    #     super().__init__()
    #     self.query = nn.Linear(input_dim, input_dim)
    #     self.key = nn.Linear(input_dim, input_dim)
    #     self.value = nn.Linear(input_dim, input_dim)
    #
    # def forward(self, driver, vehicle, topo):
    #     q = torch.tanh(self.query(driver))
    #     k = torch.sigmoid(self.key(vehicle))
    #     v = torch.relu(self.value(topo))
    #     return torch.softmax(q * k * v, dim=1)

    # def __init__(self, input_dim):
    #     super().__init__()
    #     self.query = nn.Linear(input_dim, input_dim)
    #     self.key = nn.Linear(input_dim, input_dim)
    #     self.value = nn.Linear(input_dim, input_dim)
    #
    # def forward(self, driver, vehicle, topo):
    #     q = torch.tanh(self.query(driver))
    #     k = torch.sigmoid(self.key(vehicle))
    #     v = torch.relu(self.value(topo))
    #     return torch.softmax(q * k * v, dim=1)

    def __init__(self, driver_dim=256, vehicle_dim=256, topo_dim=128, attn_dim=256):
        super().__init__()
        # 定义各特征的投影层
        self.driver_proj = nn.Linear(driver_dim, attn_dim)
        self.vehicle_proj = nn.Linear(vehicle_dim, attn_dim)
        self.topo_proj = nn.Linear(topo_dim, attn_dim)

        # 注意力计算层
        self.attention_fc = nn.Linear(attn_dim, 1)

    def forward(self, driver, vehicle, topo):
        """
        输入参数:
            driver: [batch_size, driver_dim(256)]
            vehicle: [batch_size, vehicle_dim(256)]
            topo: [batch_size, topo_dim(128)]
        """
        # 特征投影到统一维度
        driver_proj = torch.tanh(self.driver_proj(driver))  # [batch, 256]
        vehicle_proj = torch.sigmoid(self.vehicle_proj(vehicle))  # [batch, 256]
        topo_proj = torch.relu(self.topo_proj(topo))  # [batch, 256]

        # 融合特征
        combined = driver_proj * vehicle_proj * topo_proj  # [batch, 256]

        # 计算注意力权重
        attention = torch.softmax(self.attention_fc(combined), dim=1)  # [batch, 1]
        return attention


class DrivingBehaviorModel(nn.Module):

    def __init__(self):
        super().__init__()
        # 编码器定义 (保持双向LSTM结构)
        self.driver_encoder = nn.LSTM(5, 128, num_layers=2, bidirectional=True, dropout=0.3, batch_first=True)
        self.vehicle_encoder = nn.LSTM(2, 128, num_layers=2, bidirectional=True, dropout=0.3, batch_first=True)
        self.topo_encoder = nn.LSTM(2, 64, num_layers=1, bidirectional=True, batch_first=True)

        # 注意力机制 (修正输入维度)
        self.attention = EnhancedAttention(
            driver_dim=256,  # 双向LSTM输出维度 128*2=256
            vehicle_dim=256,
            topo_dim=128,
            attn_dim=256
        )

        # 分类器 (调整输入维度)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  # 输入维度与注意力输出对齐
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

        # 特征融合投影层
        self.feature_proj = nn.Linear(640, 256)  # 将拼接特征256+256+128=640投影到256

    def forward(self, driver_status, vehicle_status, topographic):  # 修正参数名
        d_out, _ = self.driver_encoder(driver_status)
        v_out, _ = self.vehicle_encoder(vehicle_status)
        t_out, _ = self.topo_encoder(topographic)

        # 取最后时间步特征
        d_last = d_out[:, -1, :]  # [batch, 256]
        v_last = v_out[:, -1, :]  # [batch, 256]
        t_last = t_out[:, -1, :]  # [batch, 128]

        # 计算注意力权重
        attn_weights = self.attention(d_last, v_last, t_last)  # [batch, 1]

        # 特征拼接与投影
        combined = torch.cat([d_last, v_last, t_last], dim=1)  # [batch, 640]
        projected = self.feature_proj(combined)  # [batch, 256]

        # 应用注意力
        weighted = projected * attn_weights  # [batch, 256]

        # 分类输出
        return self.classifier(weighted)


# ==================== 训练框架 ====================
class TrainingFramework:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DrivingBehaviorModel().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=3)

    def train(self, train_loader, val_loader, epochs=30):
        best_f1 = 0
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            all_preds, all_labels = [], []

            for batch in train_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'label'}
                labels = batch['label'].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(**inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # 验证
            val_preds, val_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'label'}
                    outputs = self.model(**inputs)
                    val_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
                    val_labels.extend(batch['label'].numpy())

            # 计算指标
            train_acc = np.mean(np.array(all_preds) == np.array(all_labels))
            val_f1 = f1_score(val_labels, val_preds, average='weighted')
            val_iou = jaccard_score(val_labels, val_preds, average='weighted')

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {total_loss / len(train_loader):.4f} | Acc: {train_acc:.4f}")
            print(f"Val F1: {val_f1:.4f} | IoU: {val_iou:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(self.model.state_dict(), 'best_model.pth')

        print(f"Training Complete. Best Val F1: {best_f1:.4f}")


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 生成数据
    BASE_DATA =  [
    {"time": "2023-01-01 08:00:00", "latitude": 31.2304, "longitude": 121.4737, "speed": 60, "direction": 90,
     "isRapidlySpeedup": 0, "isRapidlySlowdown": 0, "isNeutralSlide": 0, "isOverspeed": 0, "isFatigueDriving": 0,
     "isHthrottleStop": 0, "isOilLeak": 0},
    {"time": "2023-01-01 08:00:05", "latitude": 31.2304, "longitude": 121.4737, "speed": 62, "direction": 90,
     "isRapidlySpeedup": 1, "isRapidlySlowdown": 0, "isNeutralSlide": 0, "isOverspeed": 0, "isFatigueDriving": 0,
     "isHthrottleStop": 0, "isOilLeak": 0},
    {"time": "2023-01-01 08:00:10", "latitude": 31.2304, "longitude": 121.4737, "speed": 63, "direction": 90,
     "isRapidlySpeedup": 0, "isRapidlySlowdown": 1, "isNeutralSlide": 0, "isOverspeed": 0, "isFatigueDriving": 0,
     "isHthrottleStop": 0, "isOilLeak": 0},
    {"time": "2023-01-01 08:00:15", "latitude": 31.2305, "longitude": 121.4739, "speed": 65, "direction": 90,
     "isRapidlySpeedup": 0, "isRapidlySlowdown": 0, "isNeutralSlide": 1, "isOverspeed": 1, "isFatigueDriving": 0,
     "isHthrottleStop": 0, "isOilLeak": 0},
    {"time": "2023-01-01 08:00:20", "latitude": 31.2305, "longitude": 121.4798, "speed": 66, "direction": 90,
     "isRapidlySpeedup": 0, "isRapidlySlowdown": 0, "isNeutralSlide": 0, "isOverspeed": 0, "isFatigueDriving": 1,
     "isHthrottleStop": 1, "isOilLeak": 0}
]  # 初始5个样本
    EMBEDDED_DATA = generate_synthetic_data(BASE_DATA, 200)

    # 预处理
    processed_data = preprocess_data(EMBEDDED_DATA)


    # 创建数据集
    class DrivingDataset(Dataset):

        def __len__(self):
            return len(self.labels)

        def __init__(self, data):
            self.driver_status = torch.FloatTensor(data['driver_status'])
            self.vehicle_status = torch.FloatTensor(data['vehicle_status'])
            self.topographic = torch.FloatTensor(data['topographic'])
            self.labels = torch.LongTensor(data['labels'])

        def __getitem__(self, idx):
            return {
                'driver_status': self.driver_status[idx],
                'vehicle_status': self.vehicle_status[idx],
                'topographic': self.topographic[idx],
                'label': self.labels[idx]
            }

    dataset = DrivingDataset(processed_data)
    # 数据集分割
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])    # 打印数据集大小 ▼▼▼
    print(f"\n{'=' * 40}")
    print(f"训练集样本数: {len(train_set)}")
    print(f"测试集样本数: {len(test_set)}")
    print(f"总样本数: {len(train_set) + len(test_set)}")
    print(f"{'=' * 40}\n")
    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)
    # 训练
    framework = TrainingFramework()
    framework.train(
        DataLoader(train_set, batch_size=32, shuffle=True),
        DataLoader(test_set, batch_size=32),
        epochs=30
    )