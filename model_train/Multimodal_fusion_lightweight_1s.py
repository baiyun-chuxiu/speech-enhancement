import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.optim.lr_scheduler import ReduceLROnPlateau
import re
from tqdm import tqdm
import cv2
import random
import time

# 适配不同版本的 AMP (混合精度训练)
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


set_seed(42)

# 设备配置
device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
print(f"使用设备: {device}")

# 路径配置
TRAIN_DIR = '/Volumes/train'
TEST_DIR = '/Volumes/test'
OUTPUT_DIR = '/Volumes/model_1s/results_video2audio_轻量化test1_1s'  # 修改输出路径以区分

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'test'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'visualizations'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'models'), exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, 'training_log.txt')


def print_log(*args, **kwargs):
    print(*args, **kwargs)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        print(*args, **kwargs, file=f)


# Dataset 类 (修改了路径和解析规则)
class AudioVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment=False):
        self.root_dir = root_dir
        # 修改为 1s 和 30 的文件夹
        self.audiopic_dir = os.path.join(root_dir, 'audiopic1s')
        self.videodisplace_dir = os.path.join(root_dir, 'videodisplace30')
        self.audiopicorigin_dir = os.path.join(root_dir, 'audiopicorigin1s')
        self.transform = transform
        self.augment = augment
        self.file_pairs = self._find_file_pairs()
        self.video_mean, self.video_std = self._compute_video_stats()

    def _find_file_pairs(self):
        file_pairs = []
        if not os.path.exists(self.audiopic_dir): return []
        audiopic_files = [f for f in os.listdir(self.audiopic_dir) if f.endswith('.png')]

        for audiopic_file in audiopic_files:
            # 增加对 _z 的解析
            match = re.search(r'audiopic1s_(\d+)_(\d+)_(\d+)\.png', audiopic_file)
            if match:
                x, y, z = match.groups()
                videodisplace_file = f'videodisplace30_{x}_{y}_{z}.npy'
                audiopicorigin_file = f'audiopicorigin1s_{x}_{y}_{z}.png'

                if (os.path.exists(os.path.join(self.videodisplace_dir, videodisplace_file)) and
                        os.path.exists(os.path.join(self.audiopicorigin_dir, audiopicorigin_file))):
                    file_pairs.append({
                        'audiopic': os.path.join(self.audiopic_dir, audiopic_file),
                        'videodisplace': os.path.join(self.videodisplace_dir, videodisplace_file),
                        'audiopicorigin': os.path.join(self.audiopicorigin_dir, audiopicorigin_file),
                        'x': int(x),
                        'y': int(y),
                        'z': int(z)  # 记录 z
                    })
        return file_pairs

    def _compute_video_stats(self):
        if not self.file_pairs: return 0.0, 1.0
        sample_pairs = random.sample(self.file_pairs, min(100, len(self.file_pairs)))
        all_video_data = []
        for pair in sample_pairs:
            video_data = np.load(pair['videodisplace'])
            all_video_data.append(video_data.flatten())
        all_video_data = np.concatenate(all_video_data)
        return np.mean(all_video_data), np.std(all_video_data) + 1e-6

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        pair = self.file_pairs[idx]

        audiopic = cv2.imread(pair['audiopic'], cv2.IMREAD_GRAYSCALE)
        if audiopic is None: return None
        audiopic = audiopic.transpose(1, 0)
        # 修改维度校验为 86
        if audiopic.shape[0] != 86 or audiopic.shape[1] != 1025:
            audiopic = cv2.resize(audiopic, (1025, 86))
        audiopic = audiopic / 255.0
        audiopic = np.expand_dims(audiopic, axis=0)

        audiopicorigin = cv2.imread(pair['audiopicorigin'], cv2.IMREAD_GRAYSCALE)
        if audiopicorigin is None: return None
        audiopicorigin = audiopicorigin.transpose(1, 0)
        # 修改维度校验为 86
        if audiopicorigin.shape[0] != 86 or audiopicorigin.shape[1] != 1025:
            audiopicorigin = cv2.resize(audiopicorigin, (1025, 86))
        audiopicorigin = audiopicorigin / 255.0
        audiopicorigin = np.expand_dims(audiopicorigin, axis=0)

        videodisplace = np.load(pair['videodisplace'])
        videodisplace = (videodisplace - self.video_mean) / self.video_std

        if self.augment:
            brightness = np.random.uniform(0.9, 1.1)
            audiopic = np.clip(audiopic * brightness, 0, 1)
            noise = np.random.normal(0, 0.01, audiopic.shape)
            audiopic = np.clip(audiopic + noise, 0, 1)
            if np.random.random() > 0.5:
                audiopic = np.flip(audiopic, axis=2).copy()
                audiopicorigin = np.flip(audiopicorigin, axis=2).copy()

        return {
            'audiopic': torch.tensor(audiopic, dtype=torch.float32),
            'videodisplace': torch.tensor(videodisplace, dtype=torch.float32),
            'audiopicorigin': torch.tensor(audiopicorigin, dtype=torch.float32),
            'x': pair['x'], 'y': pair['y'], 'z': pair['z']  # 返回 z
        }


# ==========================================
# 轻量化网络模块
# ==========================================

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                                   padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightweightFeatureFusion(nn.Module):
    def __init__(self, audio_channels, video_channels, fusion_channels):
        super(LightweightFeatureFusion, self).__init__()
        self.audio_proj = nn.Conv2d(audio_channels, fusion_channels, kernel_size=1, bias=False)
        self.audio_bn = nn.BatchNorm2d(fusion_channels)

        self.hidden_dim = 64

        self.video_proj = nn.Sequential(
            # 将 150*40 (6000) 修改为 30*40 (1200)
            nn.Linear(30 * 40, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, fusion_channels),
            nn.BatchNorm1d(fusion_channels),
            nn.ReLU(inplace=True)
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fusion_channels, fusion_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels // 4, fusion_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, audio_feat, video_feat):
        a_x = F.relu(self.audio_bn(self.audio_proj(audio_feat)))
        batch_size = video_feat.shape[0]
        v_x = video_feat.view(batch_size, -1)
        v_x = self.video_proj(v_x)
        v_x = v_x.unsqueeze(-1).unsqueeze(-1)
        fused = a_x + v_x
        attn = self.attention(fused)
        return fused * attn


class LightweightAudioDenoiseNetwork(nn.Module):
    def __init__(self, in_channels=1, fusion_channels=48):
        super(LightweightAudioDenoiseNetwork, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.enc1 = DepthwiseSeparableConv(16, 32, stride=2)
        self.enc2 = DepthwiseSeparableConv(32, 64, stride=2)
        self.enc3 = DepthwiseSeparableConv(64, 128, stride=2)

        self.fusion = LightweightFeatureFusion(128, 128, fusion_channels)

        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparableConv(fusion_channels, 64)
        )

        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparableConv(64, 32)
        )

        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparableConv(32, 16)
        )

        self.final_conv = nn.Conv2d(16, in_channels, kernel_size=3, padding=1)

    def forward(self, audio_noisy, video_disp):
        x0 = self.stem(audio_noisy)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        fused = self.fusion(x3, video_disp)

        d1 = self.dec1(fused)
        if d1.shape[2:] != x2.shape[2:]:
            d1 = F.interpolate(d1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(d1 + x2)

        if d2.shape[2:] != x1.shape[2:]:
            d2 = F.interpolate(d2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(d2 + x1)

        out = self.final_conv(d3)

        # 修改为 (86, 1025)
        if out.shape[2:] != (86, 1025):
            out = F.interpolate(out, size=(86, 1025), mode='bilinear', align_corners=True)

        return torch.sigmoid(out)


class EnhancedLoss(nn.Module):
    def __init__(self):
        super(EnhancedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_x.weight.data = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).reshape(1, 1, 3, 3).to(
            device)
        self.sobel_y.weight.data = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).reshape(1, 1, 3, 3).to(
            device)
        for param in self.sobel_x.parameters(): param.requires_grad = False
        for param in self.sobel_y.parameters(): param.requires_grad = False

    def forward(self, pred, target):
        # 修改为 (86, 1025)
        if pred.shape != target.shape:
            pred = F.interpolate(pred, size=(86, 1025), mode='bilinear', align_corners=True)

        mse = self.mse_loss(pred, target)
        l1 = self.l1_loss(pred, target)

        pred_edge = torch.abs(self.sobel_x(pred)) + torch.abs(self.sobel_y(pred))
        target_edge = torch.abs(self.sobel_x(target)) + torch.abs(self.sobel_y(target))
        edge_loss = self.mse_loss(pred_edge, target_edge)

        return 0.6 * mse + 0.3 * l1 + 0.1 * edge_loss


# ==========================================
# 训练与测试流程
# ==========================================
def compute_metrics(pred, target):
    # 修改为 (86, 1025)
    if pred.shape != target.shape:
        pred = F.interpolate(pred, size=(86, 1025), mode='bilinear', align_corners=True)

    pred_np = torch.clamp(pred, 0.0, 1.0).cpu().detach().numpy().flatten()
    target_np = torch.clamp(target, 0.0, 1.0).cpu().detach().numpy().flatten()

    pred_np = np.nan_to_num(pred_np)
    target_np = np.nan_to_num(target_np)

    mse = mean_squared_error(target_np, pred_np)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(target_np, pred_np)

    max_val = 1.0
    psnr = 20 * np.log10(max_val / rmse) if rmse > 0 else float('inf')

    mu_x, mu_y = np.mean(pred_np), np.mean(target_np)
    sigma_x, sigma_y = np.var(pred_np), np.var(target_np)
    sigma_xy = np.cov(pred_np, target_np)[0, 1]

    c1, c2 = (0.01 * max_val) ** 2, (0.03 * max_val) ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
           ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))

    return {
        'mse': mse, 'rmse': rmse, 'mae': mae,
        'psnr': psnr, 'ssim': ssim
    }


def save_corrected_image(img_tensor, save_path):
    img_np = img_tensor.cpu().detach().numpy()[0]
    img = cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.flip(img, 1)
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(save_path, img)


def train_epoch(model, dataloader, criterion, optimizer, scaler, epoch, device):
    model.train()
    total_loss = 0.0
    metrics_sum = {k: 0.0 for k in ['mse', 'rmse', 'mae', 'psnr', 'ssim']}

    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    if device.type == 'mps': device_type = 'mps'

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Train Epoch {epoch}")
    for batch_idx, batch in pbar:
        if batch is None: continue

        audio_noisy = batch['audiopic'].to(device, non_blocking=True)
        video_disp = batch['videodisplace'].to(device, non_blocking=True)
        audio_clean = batch['audiopicorigin'].to(device, non_blocking=True)

        optimizer.zero_grad()

        use_amp = (device.type == 'cuda')
        if use_amp:
            with autocast(device_type=device_type):
                pred_clean = model(audio_noisy, video_disp)
                loss = criterion(pred_clean, audio_clean)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_clean = model(audio_noisy, video_disp)
            loss = criterion(pred_clean, audio_clean)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        metrics = compute_metrics(pred_clean, audio_clean)
        for k in metrics_sum:
            metrics_sum[k] += metrics[k]

        pbar.set_postfix({"Batch Loss": f"{loss.item():.6f}"})

    avg_loss = total_loss / len(dataloader)
    avg_metrics = {k: v / len(dataloader) for k, v in metrics_sum.items()}

    print_log(f"Train Epoch {epoch} | Avg Loss: {avg_loss:.6f}")
    print_log(f"Train Metrics | MSE: {avg_metrics['mse']:.6f}, RMSE: {avg_metrics['rmse']:.6f}, "
              f"MAE: {avg_metrics['mae']:.6f}, PSNR: {avg_metrics['psnr']:.2f}dB, SSIM: {avg_metrics['ssim']:.4f}")

    return avg_loss, avg_metrics


def test_epoch(model, dataloader, criterion, epoch, device):
    model.eval()
    total_loss = 0.0
    metrics_sum = {k: 0.0 for k in ['mse', 'rmse', 'mae', 'psnr', 'ssim']}

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Test Epoch {epoch}")
        for batch_idx, batch in pbar:
            if batch is None: continue

            audio_noisy = batch['audiopic'].to(device)
            video_disp = batch['videodisplace'].to(device)
            audio_clean = batch['audiopicorigin'].to(device)
            # 解析出 z
            x, y, z = batch['x'], batch['y'], batch['z']

            pred_clean = model(audio_noisy, video_disp)

            loss = criterion(pred_clean, audio_clean)

            metrics = compute_metrics(pred_clean, audio_clean)
            for k in metrics_sum:
                metrics_sum[k] += metrics[k]

            total_loss += loss.item()

            if random.random() < 0.02:
                idx = 0
                # 保存文件名增加 z
                save_path = os.path.join(OUTPUT_DIR, 'test', f'pred_{x[idx]}_{y[idx]}_{z[idx]}_epoch{epoch}.png')
                save_corrected_image(pred_clean[idx], save_path)

            pbar.set_postfix({"Batch Loss": f"{loss.item():.6f}"})

    avg_loss = total_loss / len(dataloader)
    avg_metrics = {k: v / len(dataloader) for k, v in metrics_sum.items()}

    print_log(f"Test Epoch {epoch} | Avg Loss: {avg_loss:.6f}")
    print_log(f"Test Metrics | MSE: {avg_metrics['mse']:.6f}, RMSE: {avg_metrics['rmse']:.6f}, "
              f"MAE: {avg_metrics['mae']:.6f}, PSNR: {avg_metrics['psnr']:.2f}dB, SSIM: {avg_metrics['ssim']:.4f}")

    return avg_loss, avg_metrics


def main():
    model = LightweightAudioDenoiseNetwork().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print_log(f"轻量化模型参数量 (瘦身版): {param_count / 1e6:.2f} M")

    criterion = EnhancedLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    scaler = GradScaler(enabled=(device.type == 'cuda'))

    train_dataset = AudioVideoDataset(TRAIN_DIR, augment=True)
    test_dataset = AudioVideoDataset(TEST_DIR, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    epochs = 100
    best_loss = float('inf')

    history = {
        'train_loss': [], 'test_loss': [],
        'train_metrics': [], 'test_metrics': []
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, epoch, device)
        test_loss, test_metrics = test_epoch(model, test_loader, criterion, epoch, device)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_metrics'].append(train_metrics)
        history['test_metrics'].append(test_metrics)

        scheduler.step(test_loss)

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'models', 'best_model_light.pth'))
            print_log(f"最佳模型已保存（测试损失: {best_loss:.6f}）")

    np.save(os.path.join(OUTPUT_DIR, 'training_history.npy'), history)


if __name__ == '__main__':
    main()