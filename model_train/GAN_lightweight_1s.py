import os
import re
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from torchmetrics import StructuralSimilarityIndexMeasure

# 设置设备
device = torch.device(
    'mps' if torch.backends.mps.is_available() else
    'cuda' if torch.cuda.is_available() else
    'cpu'
)
print(f"使用设备: {device}")

# 初始化SSIM评估器
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

# ==========================================
# 修改点 1：调整输入输出张量的 Shape
# ==========================================
IMU_SHAPE = (30, 2)
VIDEO_SHAPE = (30, 40)
BATCH_SIZE = 32
EPOCHS = 100
LR_G = 0.00012
LR_D = 0.00006
BETA1 = 0.5
LAMBDA_SSIM = 80.0
LAMBDA_MSE = 20.0
LAMBDA_PERCEPTUAL = 20.0
LAMBDA_EDGE = 8.0
LAMBDA_GAN = 0.8
LAMBDA_GP = 10.0
EMA_DECAY = 0.999

# ==========================================
# 修改点 2：更新数据路径为 30 的文件夹
# ==========================================
train_imu_path = '/Volumes/train/accgyropic30*2'
train_video_path = '/Volumes/train/videodisplace30pic'
test_imu_path = '/Volumes/test/accgyropic30*2'
test_video_path = '/Volumes/test/videodisplace30pic'

# 创建保存结果的目录
os.makedirs('results', exist_ok=True)
os.makedirs('results/train', exist_ok=True)
os.makedirs('results/test', exist_ok=True)
os.makedirs('results/final_test', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('metrics', exist_ok=True)


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(
            0).to(device)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(
            0).to(device)

    def forward(self, x, y):
        x_edge_x = F.conv2d(x, self.sobel_x, padding=1)
        x_edge_y = F.conv2d(x, self.sobel_y, padding=1)
        x_edge = torch.sqrt(x_edge_x ** 2 + x_edge_y ** 2 + 1e-8)
        y_edge_x = F.conv2d(y, self.sobel_x, padding=1)
        y_edge_y = F.conv2d(y, self.sobel_y, padding=1)
        y_edge = torch.sqrt(y_edge_x ** 2 + y_edge_y ** 2 + 1e-8)
        return F.l1_loss(x_edge, y_edge)

# ==========================================
# 修改点 3：增加对 z 的正则解析
# ==========================================
def parse_filename(filename):
    pattern = r'([\w*]+)_(\d+)_(\d+)_(\d+)\.png'
    match = re.match(pattern, filename)
    if match:
        prefix, x, y, z = match.groups()
        return prefix, int(x), int(y), int(z)
    return None, None, None, None


class IMUToVideoDataset(Dataset):
    def __init__(self, imu_dir, video_dir, transform=None, is_train=True):
        self.imu_dir = imu_dir
        self.video_dir = video_dir
        self.transform = transform
        self.is_train = is_train

        imu_files = [f for f in os.listdir(imu_dir) if f.endswith('.png')]
        self.imu_dict = {}
        for f in imu_files:
            _, x, y, z = parse_filename(f)
            if x is not None and y is not None and z is not None:
                self.imu_dict[(x, y, z)] = f

        video_files = [f for f in os.listdir(video_dir) if f.endswith('.png')]
        self.video_dict = {}
        self.video_filenames = {}
        for f in video_files:
            prefix, x, y, z = parse_filename(f)
            if x is not None and y is not None and z is not None:
                self.video_dict[(x, y, z)] = f
                self.video_filenames[(x, y, z)] = f

        # 配对逻辑增加对 z 的匹配
        self.matching_pairs = list(set(self.imu_dict.keys()) & set(self.video_dict.keys()))
        print(f"找到 {len(self.matching_pairs)} 对匹配的数据")

    def __len__(self):
        return len(self.matching_pairs)

    def __getitem__(self, idx):
        x, y, z = self.matching_pairs[idx]
        imu_img_path = os.path.join(self.imu_dir, self.imu_dict[(x, y, z)])
        imu_image = Image.open(imu_img_path).convert('L')
        video_img_path = os.path.join(self.video_dir, self.video_dict[(x, y, z)])
        video_image = Image.open(video_img_path).convert('L')
        video_filename = self.video_filenames[(x, y, z)]

        # ==========================================
        # 修改点 4：精准修复旋转逻辑
        # PIL.Image 的 size 为 (宽, 高)。我们需要张量是 (30, 2) 和 (30, 40)
        # 也就是经过 ToTensor 转换前，PIL 的宽高应当分别是 (2, 30) 和 (40, 30)
        # ==========================================
        if imu_image.size[0] != 2:
            imu_image = imu_image.rotate(90, expand=True)
        if video_image.size[0] != 40:
            video_image = video_image.rotate(90, expand=True)

        if self.is_train:
            if np.random.random() < 0.5:
                imu_array = np.array(imu_image)
                noise = np.random.normal(0, 5, imu_array.shape).astype(np.int8)
                imu_array = np.clip(imu_array + noise, 0, 255).astype(np.uint8)
                imu_image = Image.fromarray(imu_array)
                video_array = np.array(video_image)
                noise = np.random.normal(0, 5, video_array.shape).astype(np.int8)
                video_array = np.clip(video_array + noise, 0, 255).astype(np.uint8)
                video_image = Image.fromarray(video_array)

        if self.transform:
            imu_image = self.transform(imu_image)
            video_image = self.transform(video_image)

        if imu_image.shape[1:] != IMU_SHAPE:
            imu_image = F.interpolate(
                imu_image.unsqueeze(0), size=IMU_SHAPE, mode='bilinear', align_corners=True
            ).squeeze(0)
        if video_image.shape[1:] != VIDEO_SHAPE:
            video_image = F.interpolate(
                video_image.unsqueeze(0), size=VIDEO_SHAPE, mode='bilinear', align_corners=True
            ).squeeze(0)

        imu_original = imu_image.clone()
        video_original = video_image.clone()
        imu_normalized = (imu_image * 2) - 1
        video_normalized = (video_image * 2) - 1

        # 返回值增加了一个 z
        return imu_normalized, video_normalized, imu_original, video_original, x, y, z, video_filename


transform = transforms.Compose([transforms.ToTensor()])
train_dataset = IMUToVideoDataset(train_imu_path, train_video_path, transform, is_train=True)
test_dataset = IMUToVideoDataset(test_imu_path, test_video_path, transform, is_train=False)

num_workers = 0 if device.type == 'mps' else 4
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features.to(device)
        self.features1 = nn.Sequential(*list(vgg.children())[:4])
        self.features2 = nn.Sequential(*list(vgg.children())[:9])
        self.features3 = nn.Sequential(*list(vgg.children())[:16])
        for param in self.features1.parameters(): param.requires_grad = False
        for param in self.features2.parameters(): param.requires_grad = False
        for param in self.features3.parameters(): param.requires_grad = False
        self.to_3ch = nn.Conv2d(1, 3, kernel_size=1).to(device)
        self.l1_loss = nn.L1Loss()

    def forward(self, x, y):
        x_3ch = self.to_3ch(x)
        y_3ch = self.to_3ch(y)
        x1, y1 = self.features1(x_3ch), self.features1(y_3ch)
        x2, y2 = self.features2(x_3ch), self.features2(y_3ch)
        x3, y3 = self.features3(x_3ch), self.features3(y_3ch)
        return 0.3 * self.l1_loss(x1, y1) + 0.5 * self.l1_loss(x2, y2) + 0.2 * self.l1_loss(x3, y3)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels,
                                   bias=False).to(device)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False).to(device)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SpatialAttentionResidualBlock(nn.Module):
    def __init__(self, in_channels, device):
        super(SpatialAttentionResidualBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, in_channels).to(device)
        self.bn1 = nn.BatchNorm2d(in_channels).to(device)
        self.conv2 = DepthwiseSeparableConv(in_channels, in_channels).to(device)
        self.bn2 = nn.BatchNorm2d(in_channels).to(device)

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=5, padding=2).to(device),
            nn.Sigmoid()
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1).to(device),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1).to(device),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        spatial_att = self.spatial_attention(out)
        channel_att = self.channel_attention(out)
        out = out * spatial_att * channel_att

        out += residual
        return F.relu(out)


class Generator(nn.Module):
    def __init__(self, device):
        super(Generator, self).__init__()
        self.device = device
        base_ch = 32

        self.init_conv = nn.Sequential(
            nn.Conv2d(1, base_ch, kernel_size=5, stride=1, padding=2).to(device),
            nn.BatchNorm2d(base_ch).to(device),
            nn.ReLU(inplace=True)
        )

        self.down1 = nn.Sequential(
            DepthwiseSeparableConv(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1).to(device),
            nn.BatchNorm2d(base_ch * 2).to(device),
            nn.ReLU(inplace=True)
        )

        self.down2 = nn.Sequential(
            DepthwiseSeparableConv(base_ch * 2, base_ch * 4, kernel_size=3, stride=2, padding=1).to(device),
            nn.BatchNorm2d(base_ch * 4).to(device),
            nn.ReLU(inplace=True)
        )

        self.shared_res_block = SpatialAttentionResidualBlock(base_ch * 4, device)
        self.num_res_blocks = 5

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=4, stride=2, padding=1).to(device),
            nn.BatchNorm2d(base_ch * 2).to(device),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1).to(device),
            nn.BatchNorm2d(base_ch).to(device),
            nn.ReLU(inplace=True)
        )

        self.width_expansion = nn.Sequential(
            DepthwiseSeparableConv(base_ch, base_ch * 2, kernel_size=3, stride=1, padding=1).to(device),
            nn.BatchNorm2d(base_ch * 2).to(device),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch * 2, base_ch * 2, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)).to(device),
            nn.BatchNorm2d(base_ch * 2).to(device),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=(1, 5), stride=(1, 3), padding=(0, 1)).to(device),
            nn.BatchNorm2d(base_ch).to(device),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Sequential(
            DepthwiseSeparableConv(base_ch + base_ch, base_ch, kernel_size=3, stride=1, padding=1).to(device),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(base_ch, base_ch // 2, kernel_size=3, stride=1, padding=1).to(device),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, 1, kernel_size=5, stride=1, padding=2).to(device),
            nn.Tanh()
        )

    def forward(self, x):
        x0 = self.init_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)

        x_res = x2
        for _ in range(self.num_res_blocks):
            x_res = self.shared_res_block(x_res)

        x_up1 = self.up1(x_res)
        x_up2 = self.up2(x_up1)

        x_expanded = self.width_expansion(x_up2)
        x0_up = F.interpolate(x0, size=x_expanded.shape[2:], mode='bilinear', align_corners=True)
        fused = torch.cat([x_expanded, x0_up], dim=1)

        return self.final_conv(fused)


class Discriminator(nn.Module):
    def __init__(self, device):
        super(Discriminator, self).__init__()
        self.device = device

        def conv_block(in_channels, out_channels, stride=1, kernel_size=3):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                          stride=stride, padding=kernel_size // 2).to(device),
                nn.BatchNorm2d(out_channels).to(device),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3)
            )

        self.imu_features = nn.Sequential(
            conv_block(1, 32, stride=2),
            conv_block(32, 64, stride=2),
            conv_block(64, 128, stride=2)
        )

        self.video_features = nn.Sequential(
            conv_block(1, 32, stride=2),
            conv_block(32, 64, stride=2),
            conv_block(64, 128, stride=2)
        )

        self.fusion = nn.Sequential(
            conv_block(256, 256, stride=2),
            conv_block(256, 256, stride=1)
        )

        self._initialize_classifier()

    def _initialize_classifier(self):
        with torch.no_grad():
            dummy = torch.randn(1, 1, *VIDEO_SHAPE).to(self.device)
            imu_feat = self.imu_features(dummy)
            video_feat = self.video_features(dummy)
            combined = torch.cat([imu_feat, video_feat], dim=1)
            feat = self.fusion(combined)
            self.classifier = nn.Linear(int(np.prod(feat.shape[1:])), 1).to(self.device)

    def forward(self, imu, video):
        imu_resized = F.interpolate(imu, size=VIDEO_SHAPE, mode='bilinear', align_corners=True)
        imu_feat = self.imu_features(imu_resized)
        video_feat = self.video_features(video)
        combined = torch.cat([imu_feat, video_feat], dim=1)
        combined = self.fusion(combined)
        return self.classifier(combined.view(combined.size(0), -1))


def compute_gradient_penalty(D, real_samples, fake_samples, imu_input):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(imu_input, interpolates)
    fake = torch.ones(real_samples.size(0), 1, device=device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def save_dataset_results(generator, dataloader, ema, save_dir, dataset_name):
    generator.eval()
    print(f"正在保存{dataset_name}集上的结果到 {save_dir}...")
    ema.apply_shadow()
    all_metrics = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"保存{dataset_name}集输出"):
            # 修改点 5：解包适配新的 batch 返回值
            imu_norm, _, imu_orig, video_orig, x_coords, y_coords, z_coords, video_filenames = batch
            imu_norm = imu_norm.to(device)
            video_orig = video_orig.to(device)
            generated_video_norm = generator(imu_norm)
            generated_video_orig = (generated_video_norm + 1) / 2
            if generated_video_orig.shape[2:] != VIDEO_SHAPE:
                generated_video_orig = F.interpolate(
                    generated_video_orig, size=VIDEO_SHAPE, mode='bilinear', align_corners=True
                )
            for i in range(imu_orig.size(0)):
                video_filename = video_filenames[i]
                sample_mse = F.mse_loss(generated_video_orig[i], video_orig[i]).item()
                sample_ssim = ssim(generated_video_orig[i].unsqueeze(0), video_orig[i].unsqueeze(0)).item()
                all_metrics.append({'filename': video_filename, 'mse': sample_mse, 'ssim': sample_ssim})
                generated_np = generated_video_orig[i].cpu().squeeze().numpy()
                generated_img = Image.fromarray((generated_np * 255).astype(np.uint8))
                save_path = os.path.join(save_dir, video_filename)
                generated_img.save(save_path)

                if i == 0:
                    imu_np = imu_orig[i].cpu().squeeze().numpy()
                    video_np = video_orig[i].cpu().squeeze().numpy()
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(imu_np, cmap='gray');
                    axes[0].axis('off')
                    axes[1].imshow(video_np, cmap='gray');
                    axes[1].axis('off')
                    axes[2].imshow(generated_np, cmap='gray');
                    axes[2].axis('off')
                    plt.savefig(os.path.join(save_dir, f"compare_{video_filename}"))
                    plt.close()

        mse_values = [m['mse'] for m in all_metrics]
        ssim_values = [m['ssim'] for m in all_metrics]
        with open(os.path.join('metrics', f'{dataset_name}_statistics.txt'), 'w') as f:
            f.write(f"平均MSE: {np.mean(mse_values):.6f}\n")
            f.write(f"平均SSIM: {np.mean(ssim_values):.6f}\n")
    ema.restore()
    generator.train()


def test(epoch, generator, discriminator, ema):
    generator.eval()
    discriminator.eval()
    ema.apply_shadow()
    mse_loss = nn.MSELoss()
    total_mse = 0.0
    total_ssim = 0.0
    count = 0
    with torch.no_grad():
        # 修改点 6：解包适配新的 batch 返回值
        for imu_norm, video_norm, imu_orig, video_orig, _, _, _, _ in test_loader:
            imu_norm, video_norm = imu_norm.to(device), video_norm.to(device)
            imu_orig, video_orig = imu_orig.to(device), video_orig.to(device)
            generated_video_norm = generator(imu_norm)
            generated_video_orig = (generated_video_norm + 1) / 2
            if generated_video_orig.shape != video_orig.shape:
                generated_video_orig = F.interpolate(
                    generated_video_orig, size=video_orig.shape[2:], mode='bilinear', align_corners=True
                )
            total_mse += mse_loss(generated_video_orig, video_orig).item()
            total_ssim += ssim(generated_video_orig, video_orig).item()
            count += 1
    avg_mse = total_mse / count if count > 0 else 0.0
    avg_ssim = total_ssim / count if count > 0 else 0.0
    print(f"Epoch {epoch} 测试集指标: 平均MSE = {avg_mse:.6f}, 平均SSIM = {avg_ssim:.6f}")
    with open('metrics/evaluation_metrics.txt', 'a') as f:
        f.write(f"Epoch {epoch}: MSE = {avg_mse:.6f}, SSIM = {avg_ssim:.6f}\n")
    ema.restore()
    generator.train()
    discriminator.train()


def train():
    generator = Generator(device).to(device)
    discriminator = Discriminator(device).to(device)

    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"=====================================")
    print(f"Generator 版本4 (权重共享) FP32 理论参数量: {g_params / 1e6:.3f}M")
    print(f"Discriminator 参数量: {d_params / 1e6:.3f}M")
    print(f"=====================================")

    ema = EMA(generator, EMA_DECAY)
    perceptual_criterion = PerceptualLoss().to(device)
    edge_criterion = EdgeLoss().to(device)
    mse_criterion = nn.MSELoss()

    optimizer_G = optim.AdamW(generator.parameters(), lr=LR_G, betas=(BETA1, 0.999), weight_decay=1e-4)
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=LR_D, betas=(BETA1, 0.999), weight_decay=1e-4)

    scheduler_G = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_G, T_0=30, T_mult=2, eta_min=1e-6)
    scheduler_D = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_D, T_0=30, T_mult=2, eta_min=1e-6)

    with open('metrics/evaluation_metrics.txt', 'w') as f:
        f.write("训练过程中的测试集评估指标\n============================\n")

    generator.train()
    discriminator.train()

    for epoch in range(EPOCHS):
        running_loss_G = 0.0
        running_loss_D = 0.0

        if epoch < 5:
            lr_scale = (epoch + 1) / 5
            for param_group in optimizer_G.param_groups: param_group['lr'] = LR_G * lr_scale
            for param_group in optimizer_D.param_groups: param_group['lr'] = LR_D * lr_scale

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")):
            # 修改点 7：解包适配新的 batch 返回值
            imu_norm, video_norm, _, video_orig, _, _, _, _ = batch
            imu_norm, video_norm = imu_norm.to(device), video_norm.to(device)
            video_orig = video_orig.to(device)

            if video_orig.shape[2:] != VIDEO_SHAPE:
                video_orig = F.interpolate(video_orig, size=VIDEO_SHAPE, mode='bilinear', align_corners=True)

            # 1. 训练判别器
            optimizer_D.zero_grad()
            fake_video = generator(imu_norm)
            if fake_video.shape[2:] != video_norm.shape[2:]:
                fake_video = F.interpolate(fake_video, size=video_norm.shape[2:], mode='bilinear', align_corners=True)

            real_validity = discriminator(imu_norm, video_norm)
            fake_validity = discriminator(imu_norm, fake_video.detach())
            loss_real = -torch.mean(real_validity)
            loss_fake = torch.mean(fake_validity)
            gradient_penalty = compute_gradient_penalty(discriminator, video_norm.data, fake_video.data, imu_norm)
            loss_D = loss_real + loss_fake + LAMBDA_GP * gradient_penalty
            loss_D.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_D.step()
            running_loss_D += loss_D.item()

            # 2. 训练生成器
            optimizer_G.zero_grad()
            fake_validity = discriminator(imu_norm, fake_video)
            loss_G_gan = -torch.mean(fake_validity)

            fake_video_orig = (fake_video + 1) / 2
            if fake_video_orig.shape[2:] != video_orig.shape[2:]:
                fake_video_orig = F.interpolate(fake_video_orig, size=video_orig.shape[2:], mode='bilinear',
                                                align_corners=True)

            ssim_score = ssim(fake_video_orig, video_orig)
            loss_G_ssim = 1 - ssim_score
            loss_G_mse = mse_criterion(fake_video_orig, video_orig)
            loss_G_perceptual = perceptual_criterion(fake_video_orig, video_orig)
            loss_G_edge = edge_criterion(fake_video_orig, video_orig)

            loss_G = (LAMBDA_GAN * loss_G_gan +
                      LAMBDA_SSIM * loss_G_ssim +
                      LAMBDA_MSE * loss_G_mse +
                      LAMBDA_PERCEPTUAL * loss_G_perceptual +
                      LAMBDA_EDGE * loss_G_edge)
            loss_G.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_G.step()
            ema.update()

            running_loss_G += loss_G.item()

        scheduler_G.step()
        scheduler_D.step()

        avg_loss_G = running_loss_G / len(train_loader) if len(train_loader) > 0 else 0
        avg_loss_D = running_loss_D / len(train_loader) if len(train_loader) > 0 else 0
        print(f"Epoch {epoch + 1}, Loss_G: {avg_loss_G:.4f}, Loss_D: {avg_loss_D:.4f}")
        test(epoch + 1, generator, discriminator, ema)

        if (epoch + 1) % 10 == 0:
            ema.apply_shadow()
            torch.save(generator.state_dict(), f"models/generator_ema_epoch_{epoch + 1}.pth")

            gen_fp16 = copy.deepcopy(generator).half()
            torch.save(gen_fp16.state_dict(), f"models/generator_ema_epoch_{epoch + 1}_fp16.pth")

            ema.restore()
            torch.save(generator.state_dict(), f"models/generator_epoch_{epoch + 1}.pth")
            torch.save(discriminator.state_dict(), f"models/discriminator_epoch_{epoch + 1}.pth")

    print("训练完成! 正在生成最终的超轻量级模型...")

    ema.apply_shadow()
    torch.save(generator.state_dict(), "models/generator_ema_final_fp32.pth")
    gen_final_fp16 = copy.deepcopy(generator).half()
    torch.save(gen_final_fp16.state_dict(), "models/generator_ema_final_fp16.pth")
    print(">>> 极度压缩模型已保存至: models/generator_ema_final_fp16.pth <<<")
    ema.restore()

    save_dataset_results(generator, train_loader, ema, 'results/train', 'train')
    save_dataset_results(generator, test_loader, ema, 'results/test', 'test')


if __name__ == "__main__":
    print("开始训练...")
    train()
    print("所有任务完成!")