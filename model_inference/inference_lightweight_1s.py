import os
import glob
import re
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import time

# 忽略警告
warnings.filterwarnings("ignore")


# ==========================================
# 1. 网络模型定义 (适配 1s 尺寸)
# ==========================================

# ----------------- GAN 网络 (模型 4) 组件 -----------------
class GAN_DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(GAN_DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels,
                                   bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class GAN_SpatialAttentionResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(GAN_SpatialAttentionResidualBlock, self).__init__()
        self.conv1 = GAN_DepthwiseSeparableConv(in_channels, in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = GAN_DepthwiseSeparableConv(in_channels, in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
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
    def __init__(self):
        super(Generator, self).__init__()
        base_ch = 32

        self.init_conv = nn.Sequential(
            nn.Conv2d(1, base_ch, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )

        self.down1 = nn.Sequential(
            GAN_DepthwiseSeparableConv(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True)
        )

        self.down2 = nn.Sequential(
            GAN_DepthwiseSeparableConv(base_ch * 2, base_ch * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True)
        )

        self.shared_res_block = GAN_SpatialAttentionResidualBlock(base_ch * 4)
        self.num_res_blocks = 5

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )

        self.width_expansion = nn.Sequential(
            GAN_DepthwiseSeparableConv(base_ch, base_ch * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch * 2, base_ch * 2, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=(1, 5), stride=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Sequential(
            GAN_DepthwiseSeparableConv(base_ch + base_ch, base_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            GAN_DepthwiseSeparableConv(base_ch, base_ch // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, 1, kernel_size=5, stride=1, padding=2),
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


# ----------------- 去噪网络 (模型 2) 组件 -----------------
class Denoise_DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Denoise_DepthwiseSeparableConv, self).__init__()
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
        v_x = video_feat.reshape(batch_size, -1)
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

        self.enc1 = Denoise_DepthwiseSeparableConv(16, 32, stride=2)
        self.enc2 = Denoise_DepthwiseSeparableConv(32, 64, stride=2)
        self.enc3 = Denoise_DepthwiseSeparableConv(64, 128, stride=2)

        self.fusion = LightweightFeatureFusion(128, 128, fusion_channels)

        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Denoise_DepthwiseSeparableConv(fusion_channels, 64)
        )

        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Denoise_DepthwiseSeparableConv(64, 32)
        )

        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Denoise_DepthwiseSeparableConv(32, 16)
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

        if out.shape[2:] != (86, 1025):
            out = F.interpolate(out, size=(86, 1025), mode='bilinear', align_corners=True)

        return torch.sigmoid(out)


# ==========================================
# 2. 完整处理管线类
# ==========================================

class EnhancerPipeline:
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self._load_models()

    def _load_models(self):
        print(f"正在加载模型 (使用设备: {self.device})...")
        self.generator = Generator().to(self.device)
        try:
            self.generator.load_state_dict(
                torch.load(self.config['generator_model_path'], map_location=self.device, weights_only=True))
            self.generator.eval()
            print("GAN Generator 加载成功")
        except Exception as e:
            print(f"GAN Generator 加载失败 (请检查路径): {e}")

        self.denoise_net = LightweightAudioDenoiseNetwork().to(self.device)
        try:
            self.denoise_net.load_state_dict(
                torch.load(self.config['denoise_model_path'], map_location=self.device, weights_only=True))
            self.denoise_net.eval()
            print("Denoise Network 加载成功")
        except Exception as e:
            print(f"Denoise Network 加载失败 (请检查路径): {e}")

    # -------------------------------------------------------------------------
    # 步骤 1: 预处理 IMU 数据 (1s = 25截断 -> 30插值)
    # -------------------------------------------------------------------------
    def process_imu_data(self, acc_path, gyro_path):
        def load_and_fix(path):
            df = pd.read_csv(path, sep=' ', header=None)
            data = df.iloc[:, :3].values if df.shape[1] == 3 else df.iloc[:, 1:4].values
            if len(data) >= 25:
                data = data[:25]
            else:
                data = np.vstack([data, np.zeros((25 - len(data), 3))])
            return data

        def interpolate(data):
            orig = np.linspace(0, 1.0, 25, endpoint=False)
            target = np.linspace(0, 1.0, 30, endpoint=False)
            res = np.empty((30, 3))
            for i in range(3):
                res[:, i] = np.interp(target, orig, data[:, i])
            return res

        acc = load_and_fix(acc_path)
        gyro = load_and_fix(gyro_path)

        acc_30 = interpolate(acc)
        gyro_30 = interpolate(gyro)

        acc_norm = np.sqrt(np.sum(acc_30 ** 2, axis=1, keepdims=True))
        gyro_norm = np.sqrt(np.sum(gyro_30 ** 2, axis=1, keepdims=True))
        return np.hstack((acc_norm, gyro_norm))

    # -------------------------------------------------------------------------
    # 步骤 2: GAN 生成视频位移 (150 -> 30)
    # -------------------------------------------------------------------------
    def generate_video_displacement(self, imu_data):
        min_v, max_v = imu_data.min(), imu_data.max()
        if max_v == min_v:
            norm = np.zeros_like(imu_data, dtype=np.uint8)
        else:
            norm = ((imu_data - min_v) / (max_v - min_v) * 255).astype(np.uint8)

        img_pil = Image.fromarray(norm.T, mode='L')
        if img_pil.size[0] > img_pil.size[1]:
            img_pil = img_pil.rotate(90, expand=True)

        imu_np = np.array(img_pil, dtype=np.float32)
        imu_np = (imu_np / 127.5) - 1.0
        imu_tensor = torch.from_numpy(imu_np).unsqueeze(0).unsqueeze(0).to(self.device)

        if imu_tensor.shape[2:] != (30, 6):
            imu_tensor = F.interpolate(imu_tensor, size=(30, 6), mode='bilinear', align_corners=True)

        with torch.no_grad():
            gen_tensor = self.generator(imu_tensor)

        gen_tensor = (gen_tensor + 1) / 2
        gen_tensor = F.interpolate(gen_tensor, size=(30, 40), mode='bilinear', align_corners=True)
        gen_np = gen_tensor.squeeze().cpu().numpy()
        gen_img = Image.fromarray((gen_np * 255).astype(np.uint8))

        rotated_img = gen_img.rotate(-90, expand=True)
        gray_image = np.array(rotated_img)
        data_transposed = gray_image.T.astype(np.float32)

        FIXED_MIN, FIXED_MAX = -50.0, 50.0
        RANGE = FIXED_MAX - FIXED_MIN
        recovered_data = (data_transposed / 255.0) * RANGE + FIXED_MIN

        return recovered_data

    # -------------------------------------------------------------------------
    # 步骤 3: 音频预处理 (1s => 86 帧)
    # -------------------------------------------------------------------------
    def process_audio_input(self, wav_path):
        y, sr = librosa.load(wav_path, sr=None)
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-9)

        S_norm = S_norm[::-1, :].copy()
        phase = np.angle(D)

        if S_norm.shape != (1025, 86):
            S_norm = cv2.resize(S_norm, (86, 1025))

        return S_norm, phase, sr

    # -------------------------------------------------------------------------
    # 步骤 4: 联合去噪
    # -------------------------------------------------------------------------
    def run_denoise_model(self, audio_spec_norm, video_data_npy):
        audio_in = audio_spec_norm.T.copy()
        audio_tensor = torch.tensor(audio_in, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        video_norm = (video_data_npy - self.config['video_mean']) / self.config['video_std']
        video_tensor = torch.tensor(video_norm, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            clean_spec = self.denoise_net(audio_tensor, video_tensor)
            clean_spec = torch.clamp(clean_spec, 0.0, 1.0)

        return clean_spec.squeeze().cpu().numpy()

    # -------------------------------------------------------------------------
    # 步骤 5: Pic -> NPY 转换
    # -------------------------------------------------------------------------
    def convert_spec_image_to_npy(self, clean_spec_np, output_img_path):
        img_data = clean_spec_np.T
        img_uint8 = (img_data * 255).astype(np.uint8)
        Image.fromarray(img_uint8, mode='L').save(output_img_path)

        img = Image.open(output_img_path).convert('L')
        img_array = np.array(img, dtype=np.float32)
        img_normalized = img_array / 255.0

        img_flipped = img_normalized[::-1, :].copy()
        return img_flipped

    # -------------------------------------------------------------------------
    # 步骤 6: ISTFT 合成语音
    # -------------------------------------------------------------------------
    def reconstruct_audio_istft(self, norm_mag, phase, output_wav_path, db_range=90):
        norm_mag = norm_mag.astype(np.float32)
        if norm_mag.shape != phase.shape:
            target_width = phase.shape[1]
            target_height = phase.shape[0]
            norm_mag = cv2.resize(norm_mag, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        S_db = norm_mag * db_range - db_range
        magnitude = librosa.db_to_amplitude(S_db)

        if magnitude.shape != phase.shape:
            magnitude = magnitude[:phase.shape[0], :phase.shape[1]]

        D = magnitude * np.exp(1j * phase)
        y_recon = librosa.istft(D, hop_length=512, win_length=2048, window='hann', center=True)

        max_amp = np.max(np.abs(y_recon))
        if max_amp > 0:
            scale_factor = 0.95 / max_amp
            y_recon = y_recon * scale_factor

        sf.write(output_wav_path, y_recon, 44100)

        plt.figure(figsize=(10, 4))
        plt.plot(y_recon)
        plt.title("Reconstructed Audio Waveform")
        plt.tight_layout()
        plt.savefig(output_wav_path + ".png")
        plt.close()

    def process_single_sample(self, wav_path, acc_path, gyro_path, output_dir, file_id):
        sample_start_time = time.time()
        imu_data = self.process_imu_data(acc_path, gyro_path)
        video_disp = self.generate_video_displacement(imu_data)
        audio_spec_norm, audio_phase, sr = self.process_audio_input(wav_path)
        clean_spec_raw = self.run_denoise_model(audio_spec_norm, video_disp)

        spec_img_path = os.path.join(output_dir, f"audiopic_{file_id}.png")
        cleaned_npy = self.convert_spec_image_to_npy(clean_spec_raw, spec_img_path)
        np.save(os.path.join(output_dir, f"audionpy_{file_id}.npy"), cleaned_npy)

        out_wav_path = os.path.join(output_dir, f"enhanced_{file_id}.wav")
        self.reconstruct_audio_istft(cleaned_npy, audio_phase, out_wav_path, db_range=90)
        return time.time() - sample_start_time


# ==========================================
# 批量处理逻辑 (适配 segment_x_z 命名)
# ==========================================
def process_batch_directory(pipeline, audio_dir, acc_dir, gyro_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"正在扫描音频文件: {audio_dir} ...")

    # 扫描包含 .wav 的所有文件
    audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))

    if not audio_files:
        print("❌ 未找到任何 .wav 文件")
        return

    processing_times = []
    print(f"找到 {len(audio_files)} 个音频文件，开始批量处理...")
    print("-" * 50)

    count = 0
    for audio_path in audio_files:
        filename = os.path.basename(audio_path)

        # 【核心修改】直接提取纯粹的文件名（去除 .wav）作为 ID
        # 这确保了无论是 segment_1_1 还是什么格式，都能精准匹配对应的 txt 和生成对应名称的输出文件
        file_id = filename.replace('.wav', '')

        # 也可以加个正则做双重保险，这里只针对 segment_x_z 的格式：
        match = re.search(r'segment_(\d+)_(\d+)\.wav', filename)
        if match:
            x, z = match.groups()
            file_id = f"segment_{x}_{z}"

        # 构造 txt 文件路径，直接利用完整的替换
        acc_path = os.path.join(acc_dir, filename.replace('.wav', '.txt'))
        gyro_path = os.path.join(gyro_dir, filename.replace('.wav', '.txt'))

        if os.path.exists(acc_path) and os.path.exists(gyro_path):
            count += 1
            print(f"[{count}/{len(audio_files)}] 处理 {file_id} ...", end="", flush=True)
            try:
                t = pipeline.process_single_sample(audio_path, acc_path, gyro_path, output_dir, file_id)
                processing_times.append(t)
                print(f" 完成 ({t:.3f}s)")
            except Exception as e:
                print(f" 失败: {e}")
        else:
            print(f"跳过 {file_id}: 缺少对应的 Acc 或 Gyro txt 文件")

    print("-" * 50)
    print("✅ 批量处理完成！")
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        print(f"成功处理文件数: {len(processing_times)}")
        print(f"总耗时: {sum(processing_times):.4f} 秒")
        print(f"平均每文件耗时: {avg_time:.4f} 秒")
    else:
        print("未成功处理任何文件。")


# ==========================================
# 3. 运行配置
# ==========================================
if __name__ == "__main__":
    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"

    CONFIG = {
        "device": torch.device(device_name),
        # 【修改】将权重路径更新为你最新的 1s 模型权重文件位置
        "generator_model_path": "/Volumes//model_1s/results_GAN_轻量化1s/models/generator_ema_final_fp16.pth",
        "denoise_model_path": "/Volumes//model_1s/results_video2audio_轻量化test1_1s/models/best_model_light.pth",
        "video_mean": 0.0,
        "video_std": 1.0
    }

    print(f"🚀 初始化完成，使用设备: {device_name.upper()}")

    # --- 输入文件夹路径 (请修改为你本地 1s 数据的实际存放路径) ---
    input_audio_dir = "/Volumes/批量wavtxt测试数据_1s/audio"
    input_acc_dir = "/Volumes/批量wavtxt测试数据_1s/acc"
    input_gyro_dir = "/Volumes/批量wavtxt测试数据_1s/gyro"

    output_folder = "./batch_output_IMUtest3_videotest1_1s"

    pipeline = EnhancerPipeline(CONFIG)

    if os.path.exists(input_audio_dir):
        process_batch_directory(pipeline, input_audio_dir, input_acc_dir, input_gyro_dir, output_folder)
    else:
        print(f"错误: 音频文件夹不存在 -> {input_audio_dir}")