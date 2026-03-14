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

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# 设备配置
device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
print(f"使用设备: {device}")

# 路径配置
TRAIN_DIR = '/Volumes/戴熠辰/2025.infocom.dataset/2025.7.12dataset/train'
TEST_DIR = '/Volumes/戴熠辰/2025.infocom.dataset/2025.7.12dataset/test'
OUTPUT_DIR = '/Volumes/戴熠辰/2025.infocom.dataset/video_audiopic_results/19'

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'test'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'visualizations'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'debug'), exist_ok=True)

# 日志文件
LOG_FILE = os.path.join(OUTPUT_DIR, 'training_log.txt')


# 日志打印函数
def print_log(*args, **kwargs):
    print(*args, **kwargs)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        print(*args, **kwargs, file=f)


class AudioVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment=False, is_test=False):
        self.root_dir = root_dir
        self.audiopic_dir = os.path.join(root_dir, 'audiopic')
        self.videodisplace_dir = os.path.join(root_dir, 'videodisplace150')
        self.audiopicorigin_dir = os.path.join(root_dir, 'audiopicorigin')
        self.transform = transform
        self.augment = augment
        self.is_test = is_test
        self.file_pairs = self._find_file_pairs()
        self.video_mean, self.video_std = self._compute_video_stats()
        self.known_speakers = {1, 2, 3, 5, 6, 7, 9, 10, 11}
        self.unknown_speakers = {4, 8, 12}

    def _find_file_pairs(self):
        file_pairs = []
        audiopic_files = [f for f in os.listdir(self.audiopic_dir) if f.endswith('.png')]

        for audiopic_file in audiopic_files:
            match = re.search(r'audiopic_(\d+)_(\d+)\.png', audiopic_file)
            if match:
                x, y = match.groups()
                videodisplace_file = f'videodisplace150_{x}_{y}.npy'
                audiopicorigin_file = f'audiopicorigin_{x}_{y}.png'

                if (os.path.exists(os.path.join(self.videodisplace_dir, videodisplace_file)) and
                        os.path.exists(os.path.join(self.audiopicorigin_dir, audiopicorigin_file))):
                    file_pairs.append({
                        'audiopic': os.path.join(self.audiopic_dir, audiopic_file),
                        'videodisplace': os.path.join(self.videodisplace_dir, videodisplace_file),
                        'audiopicorigin': os.path.join(self.audiopicorigin_dir, audiopicorigin_file),
                        'x': int(x),
                        'y': int(y)
                    })

        return file_pairs

    def _compute_video_stats(self):
        if not self.file_pairs:
            return 0.0, 1.0
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
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pair = self.file_pairs[idx]
        y = pair['y']
        original_x = pair['x']

        # 确定domain属性
        if self.is_test:
            domain = 1
        else:
            if y in self.unknown_speakers:
                domain = 1
            elif y in self.known_speakers:
                domain = 0
            else:
                domain = 1
                print_log(f"警告: 发现未定义的说话人编号 {y}，默认视为未知域")

        # 处理x值
        if not self.is_test and domain == 1:
            x = -1
        else:
            x = original_x

        # 加载带噪音频图像
        audiopic = cv2.imread(pair['audiopic'], cv2.IMREAD_GRAYSCALE)
        if audiopic is None:
            print_log(f"警告: 无法加载图像 {pair['audiopic']}")
            return None

        audiopic = audiopic.transpose(1, 0)
        if audiopic.shape[0] != 431 or audiopic.shape[1] != 1025:
            audiopic = cv2.resize(audiopic, (1025, 431))
        audiopic = audiopic / 255.0
        audiopic = np.expand_dims(audiopic, axis=0)

        # 加载干净音频图像
        audiopicorigin = cv2.imread(pair['audiopicorigin'], cv2.IMREAD_GRAYSCALE)
        if audiopicorigin is None:
            print_log(f"警告: 无法加载图像 {pair['audiopicorigin']}")
            return None

        audiopicorigin = audiopicorigin.transpose(1, 0)
        if audiopicorigin.shape[0] != 431 or audiopicorigin.shape[1] != 1025:
            audiopicorigin = cv2.resize(audiopicorigin, (1025, 431))
        audiopicorigin = audiopicorigin / 255.0
        audiopicorigin = np.expand_dims(audiopicorigin, axis=0)

        # 加载视频位移矩阵
        videodisplace = np.load(pair['videodisplace'])
        videodisplace = (videodisplace - self.video_mean) / self.video_std

        # 数据增强
        if self.augment:
            brightness = np.random.uniform(0.9, 1.1)
            audiopic = np.clip(audiopic * brightness, 0, 1)

            noise = np.random.normal(0, 0.01, audiopic.shape)
            audiopic = np.clip(audiopic + noise, 0, 1)

            if np.random.random() > 0.5:
                audiopic = np.flip(audiopic, axis=2).copy()
                audiopicorigin = np.flip(audiopicorigin, axis=2).copy()

        if idx < 5:
            print_log(f"样本 {idx}: 形状=(1, 431, 1025), x={x}, y={y}, domain={domain}, 原始x={original_x}")

        return {
            'audiopic': torch.tensor(audiopic, dtype=torch.float32),
            'videodisplace': torch.tensor(videodisplace, dtype=torch.float32),
            'audiopicorigin': torch.tensor(audiopicorigin, dtype=torch.float32),
            'x': x,
            'original_x': original_x,
            'y': y,
            'domain': domain
        }


# 特征融合模块
class FeatureFusionModule(nn.Module):
    def __init__(self, audio_channels, video_channels, fusion_channels):
        super(FeatureFusionModule, self).__init__()
        self.audio_proj = nn.Sequential(
            nn.Conv2d(audio_channels, fusion_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_channels, eps=1e-3),
            nn.ReLU()
        )
        self.video_proj = nn.Sequential(
            nn.Linear(150 * 40, video_channels),
            nn.BatchNorm1d(video_channels, eps=1e-3),
            nn.ReLU(),
            nn.Linear(video_channels, fusion_channels),
            nn.BatchNorm1d(fusion_channels, eps=1e-3),
            nn.ReLU()
        )
        self.attention = nn.Sequential(
            nn.Linear(fusion_channels * 2, fusion_channels),
            nn.ReLU(),
            nn.Linear(fusion_channels, 2),
            nn.Softmax(dim=1)
        )
        self.fusion_refine = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_channels, eps=1e-3),
            nn.ReLU()
        )

    def forward(self, audio_feat, video_feat):
        batch_size = audio_feat.shape[0]
        audio_x = self.audio_proj(audio_feat)
        audio_x = torch.clamp(audio_x, -5.0, 5.0)

        video_x = video_feat.reshape(batch_size, -1)
        video_x = self.video_proj(video_x)
        video_x = torch.clamp(video_x, -5.0, 5.0)

        audio_global = F.adaptive_avg_pool2d(audio_x, (1, 1)).reshape(batch_size, -1)
        attn_input = torch.cat([audio_global, video_x], dim=1)
        attn_weights = self.attention(attn_input)
        audio_attn, video_attn = attn_weights[:, 0:1], attn_weights[:, 1:2]

        video_x_expanded = video_x.unsqueeze(-1).unsqueeze(-1).expand_as(audio_x)
        fused_feat = (audio_attn.unsqueeze(-1).unsqueeze(-1) * audio_x) + \
                     (video_attn.unsqueeze(-1).unsqueeze(-1) * video_x_expanded)
        fused_feat = torch.clamp(fused_feat, -5.0, 5.0)
        fused_feat = self.fusion_refine(fused_feat)
        return fused_feat


# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-3)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-3)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        out = torch.clamp(out, -5.0, 5.0)
        return out


# 梯度反转层
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output  # 梯度反转


class GradientReversal(nn.Module):
    def __init__(self):
        super(GradientReversal, self).__init__()

    def forward(self, x):
        return GradientReversalFunction.apply(x)


# 音频降噪主网络（修改：域分类器移到自注意力之后解码器之前）
class AudioDenoiseNetwork(nn.Module):
    def __init__(self, in_channels=1, fusion_channels=32):
        super(AudioDenoiseNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16, eps=1e-3),
            nn.ReLU(),
            ResidualBlock(16, 16),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32, eps=1e-3),
            nn.ReLU(),
            ResidualBlock(32, 32),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64, eps=1e-3),
            nn.ReLU(),
            ResidualBlock(64, 64),
        )

        self.feature_fusion = FeatureFusionModule(
            audio_channels=64,
            video_channels=128,
            fusion_channels=fusion_channels
        )

        self.self_attention = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(fusion_channels // 2, fusion_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 修改1：域分类器移到自注意力之后，使用注意力处理后的特征
        self.domain_classifier = nn.Sequential(
            GradientReversal(),  # 保留梯度反转层
            nn.AdaptiveAvgPool2d((1, 1)),  # 对注意力处理后的特征做全局池化
            nn.Flatten(),
            nn.Linear(fusion_channels, 32),  # 输入维度为fusion_channels（32）
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(fusion_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32, eps=1e-3),
            nn.ReLU(),
            ResidualBlock(32, 32),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16, eps=1e-3),
            nn.ReLU(),
            ResidualBlock(16, 16),

            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8, eps=1e-3),
            nn.ReLU(),
            nn.Conv2d(8, in_channels, kernel_size=3, padding=1),
            nn.Upsample(size=(431, 1025), mode='bilinear', align_corners=True),
            nn.Sigmoid()
        )

    def forward(self, audio_noisy, video_disp):
        audio_encoded = self.encoder(audio_noisy)
        audio_encoded = torch.clamp(audio_encoded, -5.0, 5.0)

        # 进行音视频特征融合
        fused_feat = self.feature_fusion(audio_encoded, video_disp)
        fused_feat = torch.clamp(fused_feat, -5.0, 5.0)

        # 修改2：先应用自注意力机制
        attn_map = self.self_attention(fused_feat)
        attended_feat = fused_feat * attn_map  # 得到经过注意力处理的特征

        # 修改3：基于注意力处理后的特征进行领域预测
        domain_pred = self.domain_classifier(attended_feat)

        # 解码器使用注意力处理后的特征
        audio_clean = self.decoder(attended_feat)
        audio_clean = torch.clamp(audio_clean, 0.0, 1.0)

        return audio_clean, domain_pred


# 损失函数（无需修改，领域损失计算逻辑不变）
class EnhancedLoss(nn.Module):
    def __init__(self, domain_loss_weight=0.1, lambda_domain=1.0):
        super(EnhancedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCELoss()
        self.domain_loss_weight = domain_loss_weight
        self.lambda_domain = lambda_domain

        # 边缘检测算子
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_x.weight.data = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ]).reshape(1, 1, 3, 3).to(device)
        self.sobel_y.weight.data = torch.tensor([
            [-1., -2., -1.],
            [0., 0., 0.],
            [1., 2., 1.]
        ]).reshape(1, 1, 3, 3).to(device)

        for param in self.sobel_x.parameters():
            param.requires_grad = False
        for param in self.sobel_y.parameters():
            param.requires_grad = False

    def forward(self, pred, target, domain_pred, domain_label):
        if pred.shape != target.shape:
            pred = F.interpolate(pred, size=(431, 1025), mode='bilinear', align_corners=True)

        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        # 音频重建损失
        mse = self.mse_loss(pred, target)
        l1 = self.l1_loss(pred, target)
        pred_edge = torch.abs(self.sobel_x(pred)) + torch.abs(self.sobel_y(pred))
        target_edge = torch.abs(self.sobel_x(target)) + torch.abs(self.sobel_y(target))
        edge_loss = self.mse_loss(pred_edge, target_edge)
        reconstruction_loss = 0.6 * mse + 0.3 * l1 + 0.1 * edge_loss

        # 领域对抗损失
        domain_label = domain_label.float().unsqueeze(1)
        domain_loss = self.bce_loss(domain_pred, domain_label)

        total_loss = reconstruction_loss + self.domain_loss_weight * (self.lambda_domain * domain_loss)

        return total_loss, reconstruction_loss, domain_loss


# 评估指标计算
def compute_metrics(pred, target):
    if pred.shape != target.shape:
        pred = F.interpolate(pred, size=(431, 1025), mode='bilinear', align_corners=True)

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


# 可视化结果对比
def plot_comparison(noisy, pred, target, x, original_x, y, domain, epoch, phase='train'):
    noisy_img = noisy[0]
    pred_img = pred[0]
    target_img = target[0]

    noisy_img = process_image(noisy_img)
    pred_img = process_image(pred_img)
    target_img = process_image(target_img)

    plt.figure(figsize=(20, 6))

    plt.subplot(131)
    plt.imshow(noisy_img, cmap='magma')
    plt.title(f'带噪音频频谱 (domain={domain})')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(pred_img, cmap='magma')
    plt.title('预测干净音频频谱')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(target_img, cmap='magma')
    plt.title('真实干净音频频谱')
    plt.axis('off')

    save_path = os.path.join(OUTPUT_DIR, 'visualizations',
                             f'{phase}_epoch{epoch}_x{x}_origx{original_x}_y{y}_domain{domain}_processed.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# 保存图像函数
def save_corrected_image(img_tensor, save_path):
    img_np = img_tensor.cpu().detach().numpy()[0]
    img_np = process_image(img_np)
    img_np = (img_np * 255).astype(np.uint8)
    cv2.imwrite(save_path, img_np)


# 图像处理函数
def process_image(img):
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.flip(img, 1)
    return img


# 梯度裁剪
def clip_gradients(model, max_norm=0.5):
    total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    if total_norm > max_norm:
        print_log(f"梯度裁剪: 从 {total_norm:.4f} 到 {max_norm}")


# 训练函数
def train_epoch(model, dataloader, criterion, optimizer, epoch, device):
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_domain_loss = 0.0
    metrics_sum = {k: 0.0 for k in ['mse', 'rmse', 'mae', 'psnr', 'ssim']}

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Train Epoch {epoch}")
    for batch_idx, batch in pbar:
        if batch is None:
            continue

        audio_noisy = batch['audiopic'].to(device)
        video_disp = batch['videodisplace'].to(device)
        audio_clean = batch['audiopicorigin'].to(device)
        domain = batch['domain'].to(device)
        x, original_x = batch['x'], batch['original_x']
        y = batch['y']

        optimizer.zero_grad()
        pred_clean, domain_pred = model(audio_noisy, video_disp)

        if torch.isnan(pred_clean).any():
            print_log(f"警告: 批次 {batch_idx} 输出含NaN，强制修正")
            pred_clean = torch.nan_to_num(pred_clean)
            pred_clean = torch.clamp(pred_clean, 0.0, 1.0)

        loss, recon_loss, dom_loss = criterion(pred_clean, audio_clean, domain_pred, domain)
        loss.backward()
        clip_gradients(model, max_norm=0.5)
        optimizer.step()

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_domain_loss += dom_loss.item()

        metrics = compute_metrics(pred_clean, audio_clean)
        for k in metrics_sum:
            metrics_sum[k] += metrics[k]

        pbar.set_postfix({
            "总损失": f"{loss.item():.6f}",
            "重建损失": f"{recon_loss.item():.6f}",
            "领域损失": f"{dom_loss.item():.6f}"
        })

        if epoch % 5 == 0 and batch_idx == 0:
            for i in range(min(3, len(audio_noisy))):
                plot_comparison(
                    audio_noisy[i].cpu().detach().numpy(),
                    pred_clean[i].cpu().detach().numpy(),
                    audio_clean[i].cpu().detach().numpy(),
                    x[i].item(), original_x[i].item(), y[i].item(), domain[i].item(),
                    epoch, phase='train'
                )

    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_domain_loss = total_domain_loss / len(dataloader)
    avg_metrics = {k: v / len(dataloader) for k, v in metrics_sum.items()}

    print_log(f"Train Epoch {epoch} | "
              f"总损失: {avg_loss:.6f}, "
              f"重建损失: {avg_recon_loss:.6f}, "
              f"领域损失: {avg_domain_loss:.6f}")
    print_log(f"Train Metrics | MSE: {avg_metrics['mse']:.6f}, RMSE: {avg_metrics['rmse']:.6f}, "
              f"MAE: {avg_metrics['mae']:.6f}, PSNR: {avg_metrics['psnr']:.2f}dB, SSIM: {avg_metrics['ssim']:.4f}")

    return avg_loss, avg_metrics


# 测试函数
def test_epoch(model, dataloader, criterion, epoch, device, save_results=True):
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_domain_loss = 0.0
    metrics_sum = {k: 0.0 for k in ['mse', 'rmse', 'mae', 'psnr', 'ssim']}

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Test Epoch {epoch}")
        for batch_idx, batch in pbar:
            if batch is None:
                continue

            audio_noisy = batch['audiopic'].to(device)
            video_disp = batch['videodisplace'].to(device)
            audio_clean = batch['audiopicorigin'].to(device)
            domain = batch['domain'].to(device)
            x, original_x = batch['x'], batch['original_x']
            y = batch['y']

            pred_clean, domain_pred = model(audio_noisy, video_disp)
            pred_clean = torch.clamp(pred_clean, 0.0, 1.0)

            loss, recon_loss, dom_loss = criterion(pred_clean, audio_clean, domain_pred, domain)

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_domain_loss += dom_loss.item()

            metrics = compute_metrics(pred_clean, audio_clean)
            for k in metrics_sum:
                metrics_sum[k] += metrics[k]

            if save_results:
                for i in range(len(pred_clean)):
                    if 'test' in dataloader.dataset.root_dir:
                        save_path = os.path.join(OUTPUT_DIR, 'test', f'audiopic_{original_x[i]}_{y[i]}.png')
                    else:
                        save_path = os.path.join(OUTPUT_DIR, 'train', f'pred_x{original_x[i]}_{y[i]}.png')
                    save_corrected_image(pred_clean[i], save_path)

            if epoch % 5 == 0 and batch_idx == 0:
                for i in range(min(3, len(audio_noisy))):
                    plot_comparison(
                        audio_noisy[i].cpu().detach().numpy(),
                        pred_clean[i].cpu().detach().numpy(),
                        audio_clean[i].cpu().detach().numpy(),
                        x[i].item(), original_x[i].item(), y[i].item(), domain[i].item(),
                        epoch, phase='test'
                    )

            pbar.set_postfix({
                "总损失": f"{loss.item():.6f}",
                "重建损失": f"{recon_loss.item():.6f}",
                "领域损失": f"{dom_loss.item():.6f}"
            })

    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_domain_loss = total_domain_loss / len(dataloader)
    avg_metrics = {k: v / len(dataloader) for k, v in metrics_sum.items()}

    print_log(f"Test Epoch {epoch} | "
              f"总损失: {avg_loss:.6f}, "
              f"重建损失: {avg_recon_loss:.6f}, "
              f"领域损失: {avg_domain_loss:.6f}")
    print_log(f"Test Metrics | MSE: {avg_metrics['mse']:.6f}, RMSE: {avg_metrics['rmse']:.6f}, "
              f"MAE: {avg_metrics['mae']:.6f}, PSNR: {avg_metrics['psnr']:.2f}dB, SSIM: {avg_metrics['ssim']:.4f}")

    return avg_loss, avg_metrics


# 主函数
def main():
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("训练日志开始: {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))

    train_dataset = AudioVideoDataset(TRAIN_DIR, augment=True, is_test=False)
    test_dataset = AudioVideoDataset(TEST_DIR, augment=False, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                              num_workers=4, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                             num_workers=4, pin_memory=False)

    model = AudioDenoiseNetwork().to(device)
    criterion = EnhancedLoss(domain_loss_weight=0.1, lambda_domain=1.0).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                  verbose=True, min_lr=1e-7)

    epochs = 200
    best_test_loss = float('inf')
    history = {
        'train_loss': [], 'test_loss': [],
        'train_metrics': [], 'test_metrics': []
    }

    print_log(f"开始训练，共 {epochs} 个epoch...")
    print_log(f"已知说话人 (domain=0): {sorted(train_dataset.known_speakers)}")
    print_log(f"未知说话人 (domain=1): {sorted(train_dataset.unknown_speakers)}")
    print_log(f"测试集所有样本均视为未知说话人 (domain=1)，且保留原始x值")

    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, epoch, device
        )
        test_loss, test_metrics = test_epoch(
            model, test_loader, criterion, epoch, device
        )

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_metrics'].append(train_metrics)
        history['test_metrics'].append(test_metrics)

        scheduler.step(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            model_path = os.path.join(OUTPUT_DIR, 'models', 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print_log(f"最佳模型已保存（测试损失: {best_test_loss:.6f}）")

    # 绘制训练曲线
    plt.figure(figsize=(12, 10))

    plt.subplot(221)
    plt.plot(history['train_loss'], label='训练总损失')
    plt.plot(history['test_loss'], label='测试总损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练与测试总损失曲线')
    plt.legend()

    plt.subplot(222)
    plt.plot([m['psnr'] for m in history['train_metrics']], label='训练PSNR')
    plt.plot([m['psnr'] for m in history['test_metrics']], label='测试PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR曲线')
    plt.legend()

    plt.subplot(223)
    plt.plot([m['ssim'] for m in history['train_metrics']], label='训练SSIM')
    plt.plot([m['ssim'] for m in history['test_metrics']], label='测试SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM曲线')
    plt.legend()

    plt.subplot(224)
    plt.plot([m['mse'] for m in history['train_metrics']], label='训练MSE')
    plt.plot([m['mse'] for m in history['test_metrics']], label='测试MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('MSE曲线')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=200)
    plt.close()

    np.save(os.path.join(OUTPUT_DIR, 'training_history.npy'), history)
    print_log(f"训练完成！结果保存在: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()