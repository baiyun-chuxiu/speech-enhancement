import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from pytorch_msssim import SSIM

# ===================== 可配置参数 =====================
NOISE_ROOT = "/Users/Downloads/dataset/movenoise_head/dataset_headmovement"  # 带噪声数据集根目录
CLEAN_DIR = "/Users/Downloads/dataset/movenoise_head/dataset_clean"  # 干净图片目录
OUTPUT_DIR = "/Users/Downloads/output/movenoise_body/denoised_results"  # 输出目录
MODEL_PATH = "/Users/Downloads/output/movenoise_body/denoise_model.pth"  # 模型保存路径
IMAGE_SIZE = (256, 256)  # 输入图像尺寸
BATCH_SIZE = 16  # 批处理大小
EPOCHS = 10  # 训练轮次
LEARNING_RATE = 1e-4  # 初始学习率
WEIGHT_DECAY = 1e-5  # 权重衰减
GRAD_CLIP = 1.0  # 梯度裁剪阈值
# =====================================================

# 设备配置
device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)


# 自定义数据集类
class DenoisingDataset(Dataset):
    def __init__(self, noise_dir, clean_dir, transform=None):
        self.noise_dir = noise_dir
        self.clean_dir = clean_dir
        self.transform = transform
        self.file_pairs = []

        # 构建文件对映射
        for filename in os.listdir(noise_dir):
            if filename.endswith(".png"):
                base_name = "_".join(filename.split("_")[:-1]) + ".png"
                clean_path = os.path.join(clean_dir, base_name)
                if os.path.exists(clean_path):
                    self.file_pairs.append((
                        os.path.join(noise_dir, filename),
                        clean_path
                    ))

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        noise_path, clean_path = self.file_pairs[idx]
        noise_img = Image.open(noise_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        if self.transform:
            noise_img = self.transform(noise_img)
            clean_img = self.transform(clean_img)

        return noise_img, clean_img


# 数据预处理
transform = transforms.Compose([
    transforms.RandomCrop(IMAGE_SIZE, padding=4),
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio),
            nn.ReLU(),
            nn.Linear(in_channels // ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x).flatten(1))
        max = self.fc(self.max_pool(x).flatten(1))
        return x * self.sigmoid(avg + max).view(-1, x.size(1), 1, 1)


# 改进的U-Net++模型
class UNetPlusPlus(nn.Module):
    def __init__(self):
        super().__init__()

        # 编码器
        self.encoder1 = self._block(3, 64)
        self.encoder2 = self._block(64, 128)
        self.encoder3 = self._block(128, 256)
        self.encoder4 = self._block(256, 512)
        self.pool = nn.MaxPool2d(2)

        # 瓶颈层
        self.bottleneck = self._block(512, 1024)

        # 解码器
        self.decoder4 = self._up_block(1024, 512)
        self.decoder3 = self._up_block(512, 256)
        self.decoder2 = self._up_block(256, 128)
        self.decoder1 = self._up_block(128, 64)

        # 注意力
        self.att4 = ChannelAttention(512)
        self.att3 = ChannelAttention(256)
        self.att2 = ChannelAttention(128)

        # 输出层
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def _up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        # 编码
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        # 瓶颈
        bn = self.bottleneck(self.pool(e4))

        # 解码
        d4 = self.decoder4(bn)
        d4 = torch.cat([self.att4(e4), d4], 1)

        d3 = self.decoder3(d4)
        d3 = torch.cat([self.att3(e3), d3], 1)

        d2 = self.decoder2(d3)
        d2 = torch.cat([self.att2(e2), d2], 1)

        d1 = self.decoder1(d2)
        d1 = torch.cat([e1, d1], 1)

        return self.final(d1)


# 混合损失函数
class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIM(data_range=1.0, size_average=True)

    def forward(self, pred, target):
        return 0.7 * self.l1(pred, target) + 0.3 * (1 - self.ssim(pred, target))


# 训练函数
def train_model(model, train_loader, val_loader):
    criterion = HybridLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=2, factor=0.5)
    best_val = float('inf')

    print(f"\n{'=' * 40}")
    print(f"Starting training for {EPOCHS} epochs")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Initial LR: {LEARNING_RATE}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"{'=' * 40}\n")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            train_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                avg_loss = train_loss / 10
                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] | Batch [{batch_idx + 1}/{len(train_loader)}] | Loss: {avg_loss:.4f}")
                train_loss = 0.0

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}] Summary")
        print(f"Val Loss: {avg_val:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), MODEL_PATH)
            print("Saved new best model!")


# 测试处理
def process_test_set(model, test_loader, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)

            # 后处理
            img = outputs.squeeze().cpu().numpy().transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)
            img = cv2.filter2D(img, -1, sharpen_kernel)

            # 保存
            filename = os.path.basename(test_loader.dataset.file_pairs[i][0])
            Image.fromarray(img).save(os.path.join(output_dir, filename))


if __name__ == "__main__":
    # 初始化数据集
    train_set = DenoisingDataset(os.path.join(NOISE_ROOT, "train"), CLEAN_DIR, transform)
    val_set = DenoisingDataset(os.path.join(NOISE_ROOT, "val"), CLEAN_DIR, transform)
    test_set = DenoisingDataset(os.path.join(NOISE_ROOT, "test"), CLEAN_DIR, transform)

    # 创建数据加载器
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, 1, shuffle=False)

    # 创建模型
    model = UNetPlusPlus().to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    train_model(model, train_loader, val_loader)

    # 测试
    print("\nProcessing test set...")
    model.load_state_dict(torch.load(MODEL_PATH))
    process_test_set(model, test_loader, OUTPUT_DIR)
    print(f"\nDenoised results saved to: {OUTPUT_DIR}")