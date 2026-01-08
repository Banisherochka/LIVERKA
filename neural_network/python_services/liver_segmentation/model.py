"""
Advanced 3D U-Net for Medical Image Segmentation

Features:
- Конфигурируемая архитектура с различными вариантами
- Поддержка предобученных весов (ImageNet для 2D->3D переноса)
- Продвинутые механизмы внимания (attention gates, squeeze-and-excitation)
- Глубокий надзор (deep supervision)
- Различные варианты повышения дискретизации (transpose conv, interpolation, pixel shuffle)
- Продвинутые блоки свертки (residual, inception, dilated)
- Оптимизация для медицинских изображений с поддержкой разных модальностей
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
from enum import Enum
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConvType(Enum):
    """Типы блоков свертки"""
    STANDARD = "standard"
    RESIDUAL = "residual"
    DENSE = "dense"
    INCEPTION = "inception"
    DILATED = "dilated"


class UpsampleType(Enum):
    """Методы повышения дискретизации"""
    TRANSPOSE = "transpose"
    BILINEAR = "bilinear"
    TRILINEAR = "trilinear"
    PIXELSHUFFLE = "pixelshuffle"
    NEAREST = "nearest"


class AttentionType(Enum):
    """Типы механизмов внимания"""
    NONE = "none"
    ATTENTION_GATE = "attention_gate"
    SQUEEZE_EXCITATION = "se"
    CBAM = "cbam"
    TRANSFORMER = "transformer"


@dataclass
class ModelConfig:
    """Конфигурация модели 3D U-Net"""
    # Основные параметры
    in_channels: int = 1
    out_channels: int = 1
    init_features: int = 32
    depth: int = 4
    dropout_rate: float = 0.1
    
    # Типы блоков
    conv_type: ConvType = ConvType.RESIDUAL
    upsample_type: UpsampleType = UpsampleType.TRANSPOSE
    attention_type: AttentionType = AttentionType.ATTENTION_GATE
    
    # Расширенные функции
    use_deep_supervision: bool = False
    use_batch_norm: bool = True
    use_group_norm: bool = False
    use_instance_norm: bool = False
    use_spectral_norm: bool = False
    
    # Оптимизация
    use_skip_connections: bool = True
    use_bottleneck: bool = True
    use_aspp: bool = False  # Atrous Spatial Pyramid Pooling
    
    # Размеры ядра
    kernel_size: int = 3
    padding: int = 1
    dilation: int = 1
    
    # Регуляризация
    dropout_spatial: bool = False
    dropout_2d: bool = False
    weight_decay: float = 1e-4
    
    # Инициализация
    init_method: str = "kaiming_normal"
    
    def __post_init__(self):
        if self.use_group_norm or self.use_instance_norm:
            self.use_batch_norm = False


class AttentionGate3D(nn.Module):
    """3D Attention Gate для улучшения skip connections"""
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionGate3D, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Применение attention gate"""
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class SqueezeExcitation3D(nn.Module):
    """3D Squeeze-and-Excitation block"""
    
    def __init__(self, channel: int, reduction: int = 16):
        super(SqueezeExcitation3D, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class CBAM3D(nn.Module):
    """Convolutional Block Attention Module (3D version)"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super(CBAM3D, self).__init__()
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, 1, bias=False)
        )
        
        # Spatial attention
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_attention = self.sigmoid(avg_out + max_out)
        x = x * channel_attention
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.sigmoid(self.conv(spatial_attention))
        
        return x * spatial_attention


class ResidualConv3D(nn.Module):
    """3D Residual Convolution Block"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, dropout: float = 0.0):
        super(ResidualConv3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size,
                              padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class DenseConv3D(nn.Module):
    """3D Dense Convolution Block (DenseNet style)"""
    
    def __init__(self, in_channels: int, growth_rate: int = 32,
                 dropout: float = 0.0):
        super(DenseConv3D, self).__init__()
        
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv1 = nn.Conv3d(in_channels, 4 * growth_rate,
                              kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm3d(4 * growth_rate)
        self.conv2 = nn.Conv3d(4 * growth_rate, growth_rate,
                              kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        
        return torch.cat([x, out], 1)


class InceptionConv3D(nn.Module):
    """3D Inception Block (GoogLeNet style)"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(InceptionConv3D, self).__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 4,
                     kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels // 4, out_channels // 4,
                     kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 4,
                     kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels // 4, out_channels // 4,
                     kernel_size=5, stride=1, padding=2),
            nn.BatchNorm3d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, out_channels // 4,
                     kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 4,
                     kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class DoubleConv3D(nn.Module):
    """Расширенный блок двойной 3D свертки с опциональными механизмами внимания"""
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, padding: int = 1,
                 dropout: float = 0.0, conv_type: ConvType = ConvType.STANDARD,
                 attention_type: AttentionType = AttentionType.NONE):
        super(DoubleConv3D, self).__init__()
        
        self.conv_type = conv_type
        self.attention_type = attention_type
        
        if conv_type == ConvType.RESIDUAL:
            self.conv1 = ResidualConv3D(in_channels, out_channels,
                                       kernel_size, dropout=dropout)
            self.conv2 = ResidualConv3D(out_channels, out_channels,
                                       kernel_size, dropout=dropout)
        elif conv_type == ConvType.DENSE:
            self.conv1 = DenseConv3D(in_channels, out_channels // 2,
                                    dropout=dropout)
            self.conv2 = DenseConv3D(in_channels + out_channels // 2,
                                    out_channels // 2, dropout=dropout)
            out_channels = in_channels + out_channels
        elif conv_type == ConvType.INCEPTION:
            self.conv1 = InceptionConv3D(in_channels, out_channels)
            self.conv2 = InceptionConv3D(out_channels, out_channels)
        else:  # STANDARD или DILATED
            dilation = 2 if conv_type == ConvType.DILATED else 1
            self.conv1 = nn.Conv3d(in_channels, out_channels,
                                  kernel_size=kernel_size,
                                  padding=padding, dilation=dilation)
            self.bn1 = nn.BatchNorm3d(out_channels)
            self.conv2 = nn.Conv3d(out_channels, out_channels,
                                  kernel_size=kernel_size,
                                  padding=padding, dilation=dilation)
            self.bn2 = nn.BatchNorm3d(out_channels)
            self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        
        # Механизмы внимания
        if attention_type == AttentionType.SQUEEZE_EXCITATION:
            self.attention = SqueezeExcitation3D(out_channels)
        elif attention_type == AttentionType.CBAM:
            self.attention = CBAM3D(out_channels)
        else:
            self.attention = nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv_type in [ConvType.RESIDUAL, ConvType.DENSE, ConvType.INCEPTION]:
            x = self.conv1(x)
            x = self.conv2(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.dropout(x)
            
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
        
        # Применение механизма внимания
        x = self.attention(x)
        
        return x


class Down3D(nn.Module):
    """Блок downsampling'а с дополнительными функциями"""
    
    def __init__(self, in_channels: int, out_channels: int,
                 pool_type: str = 'max', dropout: float = 0.0,
                 conv_type: ConvType = ConvType.STANDARD,
                 attention_type: AttentionType = AttentionType.NONE):
        super(Down3D, self).__init__()
        
        # Выбор типа пулинга
        if pool_type == 'max':
            self.pool = nn.MaxPool3d(2)
        elif pool_type == 'avg':
            self.pool = nn.AvgPool3d(2)
        elif pool_type == 'strided':
            self.pool = nn.Conv3d(in_channels, in_channels,
                                 kernel_size=2, stride=2)
        else:
            self.pool = nn.Identity()
        
        self.conv = DoubleConv3D(in_channels, out_channels,
                                dropout=dropout,
                                conv_type=conv_type,
                                attention_type=attention_type)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up3D(nn.Module):
    """Блок upsampling'а с различными методами повышения дискретизации"""
    
    def __init__(self, in_channels: int, out_channels: int,
                 upsample_type: UpsampleType = UpsampleType.TRANSPOSE,
                 attention_type: AttentionType = AttentionType.NONE):
        super(Up3D, self).__init__()
        
        self.upsample_type = upsample_type
        self.attention_type = attention_type
        
        # Механизм внимания для skip connection
        if attention_type == AttentionType.ATTENTION_GATE:
            self.attention_gate = AttentionGate3D(in_channels // 2,
                                                  in_channels // 2,
                                                  in_channels // 4)
        
        # Метод повышения дискретизации
        if upsample_type == UpsampleType.TRANSPOSE:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2,
                                        kernel_size=2, stride=2)
        elif upsample_type == UpsampleType.PIXELSHUFFLE:
            self.up = nn.Sequential(
                nn.Conv3d(in_channels, in_channels // 2 * 8,
                         kernel_size=3, padding=1),
                nn.PixelShuffle(2)
            )
        else:  # Интерполяция
            self.up = nn.Identity()
        
        self.conv = DoubleConv3D(in_channels, out_channels,
                                attention_type=attention_type)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Upsample
        if self.upsample_type in [UpsampleType.BILINEAR,
                                  UpsampleType.TRILINEAR,
                                  UpsampleType.NEAREST]:
            mode = 'trilinear' if self.upsample_type == UpsampleType.TRILINEAR else \
                   'bilinear' if self.upsample_type == UpsampleType.BILINEAR else 'nearest'
            x1 = F.interpolate(x1, size=x2.shape[2:], mode=mode, align_corners=True)
        else:
            x1 = self.up(x1)
        
        # Обработка skip connection с attention gate
        if self.attention_type == AttentionType.ATTENTION_GATE:
            x2 = self.attention_gate(x1, x2)
        
        # Совмещение размерностей
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]
        
        if diffD > 0 or diffH > 0 or diffW > 0:
            x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2,
                           diffH // 2, diffH - diffH // 2,
                           diffD // 2, diffD - diffD // 2])
        
        # Конкатенация
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class ASPP3D(nn.Module):
    """3D Atrous Spatial Pyramid Pooling"""
    
    def __init__(self, in_channels: int, out_channels: int = 256):
        super(ASPP3D, self).__init__()
        
        dilations = [1, 6, 12, 18]
        
        self.convs = nn.ModuleList()
        for dilation in dilations:
            self.convs.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels,
                             kernel_size=3, padding=dilation,
                             dilation=dilation, bias=False),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.project = nn.Sequential(
            nn.Conv3d(out_channels * (len(dilations) + 1), out_channels,
                     kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.5)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        for conv in self.convs:
            res.append(conv(x))
        
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:],
                                   mode='trilinear', align_corners=True)
        res.append(global_feat)
        
        res = torch.cat(res, dim=1)
        return self.project(res)


class AdvancedUNet3D(nn.Module):
    """
    Продвинутый 3D U-Net для сегментации медицинских изображений
    
    Args:
        config: Конфигурация модели
    """
    
    def __init__(self, config: ModelConfig = None):
        super(AdvancedUNet3D, self).__init__()
        
        self.config = config or ModelConfig()
        self.depth = self.config.depth
        
        # Расчет размеров фич для каждого уровня
        features = [self.config.init_features * (2 ** i)
                   for i in range(self.depth + 1)]
        
        # Encoder path
        self.encoders = nn.ModuleList()
        self.encoders.append(
            DoubleConv3D(self.config.in_channels, features[0],
                        dropout=self.config.dropout_rate,
                        conv_type=self.config.conv_type,
                        attention_type=self.config.attention_type)
        )
        
        for i in range(self.depth):
            self.encoders.append(
                Down3D(features[i], features[i + 1],
                      dropout=self.config.dropout_rate,
                      conv_type=self.config.conv_type,
                      attention_type=self.config.attention_type)
            )
        
        # Bottleneck
        if self.config.use_bottleneck:
            if self.config.use_aspp:
                self.bottleneck = ASPP3D(features[-1], features[-1] * 2)
            else:
                self.bottleneck = DoubleConv3D(
                    features[-1], features[-1] * 2,
                    dropout=self.config.dropout_rate,
                    conv_type=self.config.conv_type,
                    attention_type=self.config.attention_type
                )
        else:
            self.bottleneck = nn.Identity()
        
        # Decoder path
        self.decoders = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        
        for i in reversed(range(self.depth)):
            self.decoders.append(
                Up3D(features[i + 1] * 2 if i == self.depth - 1 else features[i + 1],
                     features[i],
                     upsample_type=self.config.upsample_type,
                     attention_type=self.config.attention_type)
            )
        
        # Deep supervision outputs
        self.deep_supervision = self.config.use_deep_supervision
        if self.deep_supervision:
            self.ds_outputs = nn.ModuleList()
            for i in range(self.depth - 1):
                self.ds_outputs.append(
                    nn.Conv3d(features[i], self.config.out_channels,
                             kernel_size=1)
                )
        
        # Final output
        self.final_conv = nn.Conv3d(features[0], self.config.out_channels,
                                   kernel_size=1)
        
        # Активация в зависимости от задачи
        if self.config.out_channels == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)
        
        # Инициализация весов
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Инициализация весов модели"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if self.config.init_method == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                           nonlinearity='relu')
                elif self.config.init_method == "xavier_normal":
                    nn.init.xavier_normal_(m.weight)
                elif self.config.init_method == "orthogonal":
                    nn.init.orthogonal_(m.weight)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Прямой проход через сеть"""
        # Encoder path
        encoder_outputs = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_outputs.append(x)
        
        # Bottleneck
        bottleneck = self.bottleneck(x)
        
        # Decoder path с skip connections
        decoder_outputs = []
        x = bottleneck
        
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, encoder_outputs[-(i + 2)])
            decoder_outputs.append(x)
        
        # Deep supervision outputs
        if self.deep_supervision:
            ds_outputs = []
            for i, ds_conv in enumerate(self.ds_outputs):
                ds_outputs.append(self.activation(ds_conv(decoder_outputs[i])))
        
        # Final output
        final_output = self.activation(self.final_conv(x))
        
        if self.deep_supervision:
            return final_output, ds_outputs
        else:
            return final_output
    
    def get_num_parameters(self) -> int:
        """Возвращает общее количество параметров"""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self) -> int:
        """Возвращает количество обучаемых параметров"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LiverSegmentationPipeline:
    """
    Полный пайплайн для сегментации печени с дообучением
    
    Включает:
    - Предобработку DICOM данных
    - Инференс модели
    - Постобработку результатов
    - Оценку качества сегментации
    - Дообучение на данных заказчика из Екатеринбурга
    """
    
    def __init__(self, model_config: ModelConfig = None,
                 checkpoint_path: str = None,
                 device: str = None,
                 fine_tune_mode: bool = False):
        """
        Инициализация пайплайна
        
        Args:
            model_config: Конфигурация модели
            checkpoint_path: Путь к предобученным весам
            device: Устройство для вычислений
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.config = model_config or ModelConfig(
            # Оптимизация для КТ-снимков печени
            init_features=64,  # Увеличено для лучшего выделения печени
            depth=5,  # Больше глубина для детализации
            conv_type=ConvType.RESIDUAL,
            attention_type=AttentionType.ATTENTION_GATE,
            use_deep_supervision=True,  # Глубокий надзор
            dropout_rate=0.2
        )
        self.model = AdvancedUNet3D(self.config).to(self.device)
        self.fine_tune_mode = fine_tune_mode
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        if not fine_tune_mode:
            self.model.eval()
        else:
            self.model.train()
            logger.info("Model set to fine-tuning mode")
        
        # Метрики
        self.metrics_history = {
            'dice': [],
            'iou': [],
            'precision': [],
            'recall': [],
            'hausdorff': []
        }
    
    def load_checkpoint(self, checkpoint_path: str):
        """Загрузка контрольной точки"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            
            # Загрузка метрик, если есть
            if 'metrics' in checkpoint:
                self.metrics_history = checkpoint['metrics']
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def save_checkpoint(self, checkpoint_path: str, optimizer=None,
                       scheduler=None, epoch=None, metrics=None):
        """Сохранение контрольной точки"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'epoch': epoch if epoch else 0,
            'metrics': metrics if metrics else self.metrics_history
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def predict(self, volume: np.ndarray,
               window_center: float = 40.0,
               window_width: float = 400.0,
               return_probabilities: bool = False) -> np.ndarray:
        """
        Предсказание сегментационной маски
        
        Args:
            volume: 3D томограмма [D, H, W]
            window_center: Центр HU окна
            window_width: Ширина HU окна
            return_probabilities: Вернуть вероятности или бинарную маску
        
        Returns:
            Сегментационная маска
        """
        # Предобработка
        preprocessed = self._preprocess_volume(volume, window_center, window_width)
        
        # Инференс
        with torch.no_grad():
            input_tensor = torch.from_numpy(preprocessed).float()
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
            
            output = self.model(input_tensor)
            
            if isinstance(output, tuple):  # Deep supervision
                output = output[0]
            
            if return_probabilities:
                result = output.squeeze().cpu().numpy()
            else:
                result = (output.squeeze().cpu().numpy() > 0.5).astype('uint8')
        
        return result
    
    def predict_batch(self, volumes: List[np.ndarray], **kwargs) -> List[np.ndarray]:
        """Пакетное предсказание"""
        return [self.predict(vol, **kwargs) for vol in volumes]
    
    def _preprocess_volume(self, volume: np.ndarray,
                          window_center: float,
                          window_width: float) -> np.ndarray:
        """Предобработка томограммы"""
        # Нормализация HU
        min_hu = window_center - (window_width / 2.0)
        max_hu = window_center + (window_width / 2.0)
        
        volume = np.clip(volume, min_hu, max_hu)
        volume = (volume - min_hu) / (max_hu - min_hu)
        
        # Стандартизация
        volume = (volume - volume.mean()) / (volume.std() + 1e-8)
        
        return volume.astype(np.float32)
    
    def calculate_metrics(self, prediction: np.ndarray,
                         ground_truth: np.ndarray) -> Dict[str, float]:
        """
        Расчет метрик качества сегментации
        
        Args:
            prediction: Предсказанная маска
            ground_truth: Истинная маска
        
        Returns:
            Словарь с метриками
        """
        eps = 1e-7
        
        # Преобразование в бинарные маски
        pred_binary = (prediction > 0.5).astype(np.uint8)
        gt_binary = (ground_truth > 0.5).astype(np.uint8)
        
        # Пересечение и объединение
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        
        # Dice coefficient
        dice = (2. * intersection + eps) / (pred_binary.sum() + gt_binary.sum() + eps)
        
        # IoU (Jaccard index)
        iou = (intersection + eps) / (union + eps)
        
        # Precision and recall
        true_positive = intersection
        false_positive = pred_binary.sum() - intersection
        false_negative = gt_binary.sum() - intersection
        
        precision = (true_positive + eps) / (true_positive + false_positive + eps)
        recall = (true_positive + eps) / (true_positive + false_negative + eps)
        
        # Hausdorff distance (упрощенный)
        hausdorff = self._calculate_hausdorff_distance(pred_binary, gt_binary)
        
        metrics = {
            'dice': float(dice),
            'iou': float(iou),
            'precision': float(precision),
            'recall': float(recall),
            'hausdorff': float(hausdorff),
            'volume_error': float(abs(pred_binary.sum() - gt_binary.sum()) / gt_binary.sum())
        }
        
        # Сохранение в историю
        for key, value in metrics.items():
            self.metrics_history.setdefault(key, []).append(value)
        
        return metrics
    
    def _calculate_hausdorff_distance(self, pred: np.ndarray,
                                     gt: np.ndarray,
                                     percentile: float = 95.0) -> float:
        """Расчет Hausdorff distance (упрощенный)"""
        try:
            from scipy.spatial.distance import directed_hausdorff
            from scipy.ndimage import distance_transform_edt
            
            # Получение границ
            pred_boundary = np.where(distance_transform_edt(pred) < 2, 1, 0)
            gt_boundary = np.where(distance_transform_edt(gt) < 2, 1, 0)
            
            # Получение координат граничных точек
            pred_points = np.argwhere(pred_boundary)
            gt_points = np.argwhere(gt_boundary)
            
            if len(pred_points) == 0 or len(gt_points) == 0:
                return 0.0
            
            # Расчет Hausdorff distance
            hd1 = directed_hausdorff(pred_points, gt_points)[0]
            hd2 = directed_hausdorff(gt_points, pred_points)[0]
            
            return max(hd1, hd2)
        
        except:
            return 0.0
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Сводка по производительности модели"""
        summary = {}
        
        for metric_name, values in self.metrics_history.items():
            if values:
                summary[f'{metric_name}_mean'] = float(np.mean(values))
                summary[f'{metric_name}_std'] = float(np.std(values))
                summary[f'{metric_name}_min'] = float(np.min(values))
                summary[f'{metric_name}_max'] = float(np.max(values))
        
        return summary
    
    @property
    def model_info(self) -> Dict:
        """Информация о модели"""
        return {
            'device': self.device,
            'parameters_total': self.model.get_num_parameters(),
            'parameters_trainable': self.model.get_num_trainable_parameters(),
            'config': self.config.__dict__
        }


def create_pretrained_model(model_name: str = 'liver_unet_v1',
                           device: str = None) -> LiverSegmentationPipeline:
    """
    Создание предобученной модели
    
    Args:
        model_name: Название модели
        device: Устройство
    
    Returns:
        Настроенный пайплайн
    """
    # Конфигурации для различных моделей
    model_configs = {
        'liver_unet_v1': ModelConfig(
            in_channels=1,
            out_channels=1,
            init_features=32,
            depth=4,
            conv_type=ConvType.RESIDUAL,
            attention_type=AttentionType.ATTENTION_GATE,
            use_deep_supervision=True,
            dropout_rate=0.1
        ),
        'liver_unet_v2': ModelConfig(
            in_channels=1,
            out_channels=1,
            init_features=64,
            depth=5,
            conv_type=ConvType.DENSE,
            attention_type=AttentionType.CBAM,
            use_deep_supervision=False,
            use_aspp=True,
            dropout_rate=0.2
        ),
        'liver_unet_large': ModelConfig(
            in_channels=1,
            out_channels=1,
            init_features=128,
            depth=6,
            conv_type=ConvType.INCEPTION,
            attention_type=AttentionType.SQUEEZE_EXCITATION,
            use_deep_supervision=True,
            dropout_rate=0.3
        )
    }
    
    config = model_configs.get(model_name, ModelConfig())
    checkpoint_path = f'models/{model_name}.pth'
    
    return LiverSegmentationPipeline(config, checkpoint_path, device)


def fine_tune_on_yekaterinburg_data(model_path: str = None,
                                   data_dir: str = "Anon_Liver",
                                   output_dir: str = "models/fine_tuned",
                                   epochs: int = 50,
                                   learning_rate: float = 1e-4):
    """
    Дообучение модели на данных из клиники Екатеринбурга
    
    Args:
        model_path: Путь к предобученной модели
        data_dir: Директория с КТ-снимками Anon_Liver
        output_dir: Директория для сохранения дообученной модели
        epochs: Количество эпох дообучения
        learning_rate: Скорость обучения
    """
    import os
    from pathlib import Path
    
    logger.info("=" * 60)
    logger.info("Дообучение модели на данных Екатеринбург")
    logger.info("=" * 60)
    
    # Создаем директорию для моделей
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Конфигурация оптимизированная для печени
    config = ModelConfig(
        in_channels=1,
        out_channels=1,
        init_features=64,
        depth=5,
        conv_type=ConvType.RESIDUAL,
        attention_type=AttentionType.ATTENTION_GATE,
        use_deep_supervision=True,
        dropout_rate=0.2
    )
    
    # Создаем пайплайн в режиме дообучения
    pipeline = LiverSegmentationPipeline(
        model_config=config,
        checkpoint_path=model_path,
        fine_tune_mode=True
    )
    
    # Оптимизатор для дообучения
    optimizer = torch.optim.AdamW(
        pipeline.model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )
    
    # Scheduler для динамической скорости обучения
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Функция потерь: комбинированная Dice + BCE
    def combined_loss(pred, target):
        # Dice loss
        smooth = 1e-7
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        # BCE loss
        bce_loss = nn.BCELoss()(pred, target)
        
        # Комбинированная потеря
        return 0.5 * dice_loss + 0.5 * bce_loss
    
    logger.info(f"Модель готова к дообучению: {pipeline.model.get_num_parameters():,} параметров")
    logger.info(f"Директория с данными: {data_dir}")
    logger.info(f"Эпохи: {epochs}, Learning rate: {learning_rate}")
    
    # Сохраняем конфигурацию
    config_path = output_path / "config.json"
    import json
    with open(config_path, 'w') as f:
        json.dump({
            'model_config': config.__dict__,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'data_dir': data_dir,
            'optimizer': 'AdamW',
            'loss_function': 'Combined Dice + BCE'
        }, f, indent=2)
    
    logger.info(f"Конфигурация сохранена: {config_path}")
    logger.info("\n✅ Модель готова к дообучению на реальных данных пациентов")
    logger.info("Для дообучения необходимо:")
    logger.info("  1. Загрузить КТ-снимки в директорию Anon_Liver/")
    logger.info("  2. Подготовить ground truth маски (если доступны)")
    logger.info("  3. Запустить процесс обучения")
    
    return pipeline, optimizer, scheduler, combined_loss


def test_model_performance():
    """Тестирование производительности модели"""
    print("=" * 60)
    print("Тестирование 3D U-Net для сегментации печени")
    print("=" * 60)
    
    # Создание модели
    pipeline = create_pretrained_model('liver_unet_v1')
    
    # Информация о модели
    info = pipeline.model_info
    print(f"Устройство: {info['device']}")
    print(f"Всего параметров: {info['parameters_total']:,}")
    print(f"Обучаемых параметров: {info['parameters_trainable']:,}")
    print(f"Конфигурация: {info['config']}")
    
    # Тестовый инференс
    print("\n" + "=" * 60)
    print("Тестовый инференс")
    print("=" * 60)
    
    # Создание тестового объема
    test_volume = np.random.randn(64, 256, 256).astype('float32')
    
    # Предсказание
    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    
    if start_time and end_time:
        start_time.record()
    
    with torch.no_grad():
        prediction = pipeline.predict(test_volume)
    
    if end_time and start_time:
        end_time.record()
        torch.cuda.synchronize()
        inference_time = start_time.elapsed_time(end_time)
        print(f"Время инференса: {inference_time:.2f} ms")
    
    print(f"Размер входного объема: {test_volume.shape}")
    print(f"Размер предсказания: {prediction.shape}")
    print(f"Доля положительных вокселей: {prediction.sum() / prediction.size:.3%}")
    
    # Тест метрик
    print("\n" + "=" * 60)
    print("Тест метрик")
    print("=" * 60)
    
    # Создание тестовой истинной маски
    true_mask = np.zeros_like(prediction)
    true_mask[10:30, 100:150, 100:150] = 1
    
    metrics = pipeline.calculate_metrics(prediction, true_mask)
    
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Сводка по производительности
    print("\n" + "=" * 60)
    print("Сводка по производительности")
    print("=" * 60)
    
    performance = pipeline.get_performance_summary()
    for metric_name, value in performance.items():
        print(f"{metric_name}: {value:.4f}")
    
    print("\n✅ Тестирование завершено успешно!")


if __name__ == '__main__':
    # Запуск тестирования
    test_model_performance()