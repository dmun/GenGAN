import torchvision.models as models
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F
import torch.utils.data
from torch.nn.utils.parametrizations import weight_norm
import math



def double_conv(channels_in, channels_out, kernel_size):
    return nn.Sequential(
        weight_norm(nn.Conv2d(channels_in, channels_out, kernel_size, padding=1)),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(),
        weight_norm(nn.Conv2d(channels_out, channels_out, kernel_size, padding=1)),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(),
    )


class UNetFilter(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_out,
        chs=[8, 16, 32, 64, 128],
        kernel_size=3,
        image_width=64,
        image_height=64,
        noise_dim=65,
        activation="sigmoid",
        nb_classes=2,
        embedding_dim=16,
        use_cond=True,
    ):
        super().__init__()
        self.use_cond = use_cond
        self.width = image_width
        self.height = image_height
        self.activation = activation
        self.embed_condition = nn.Embedding(nb_classes, embedding_dim)  # (not used)
        # noise projection layer
        self.project_noise = nn.Linear(noise_dim, noise_dim)
        # self.project_noise = nn.Linear(noise_dim, 5 * image_width // 16) 
        # condition projection layer (not used)
        self.project_cond = nn.Linear(
            embedding_dim, image_width // 16 * image_height // 16
        )

        self.dconv_down1 = double_conv(channels_in, chs[0], kernel_size)
        self.pool_down1 = nn.MaxPool2d(2, stride=2)

        self.dconv_down2 = double_conv(chs[0], chs[1], kernel_size)
        self.pool_down2 = nn.MaxPool2d(2, stride=2)

        self.dconv_down3 = double_conv(chs[1], chs[2], kernel_size)
        self.pool_down3 = nn.MaxPool2d(2, stride=2)

        self.dconv_down4 = double_conv(chs[2], chs[3], kernel_size)
        self.pool_down4 = nn.MaxPool2d(2, stride=2)

        self.dconv_down5 = double_conv(chs[3], chs[4], kernel_size)
        self.dconv_up5 = double_conv(chs[4] + chs[3] + 1, chs[3], kernel_size)

        self.dconv_up4 = double_conv(chs[3] + chs[2], chs[2], kernel_size)

        self.dconv_up3 = double_conv(chs[2] + chs[1], chs[1], kernel_size)

        self.dconv_up2 = double_conv(chs[1] + chs[0], chs[0], kernel_size)

        self.dconv_up1 = nn.Conv2d(chs[0], channels_out, kernel_size=1)

        self.pad = nn.ConstantPad2d((1, 0, 0, 0), 0)

    def forward(self, x, z, __):
        conv1_down = self.dconv_down1(x)
        pool1 = self.pool_down1(conv1_down)

        conv2_down = self.dconv_down2(pool1)
        pool2 = self.pool_down2(conv2_down)

        conv3_down = self.dconv_down3(pool2)
        pool3 = self.pool_down3(conv3_down)

        conv4_down = self.dconv_down4(pool3)
        pool4 = self.pool_down4(conv4_down)

        conv5_down = self.dconv_down5(pool4)

        # print("conv5_down.shape:", conv5_down.shape)
        # print("noise", z.shape)
        z = z.reshape(x.shape[0], 1, 5, conv5_down.shape[-1])
        noise = self.project_noise(z)

        conv5_down = torch.cat((conv5_down, noise), dim=1)
        conv5_up = F.interpolate(conv5_down, scale_factor=2, mode="nearest")
        conv5_up = TF.crop(conv5_up, 0, 0, conv4_down.shape[2], conv4_down.shape[3])
        conv5_up = torch.cat((conv4_down, conv5_up), dim=1)

        conv5_up = self.dconv_up5(conv5_up)
        conv4_up = F.interpolate(conv5_up, scale_factor=2, mode="nearest")
        conv4_up = TF.crop(conv4_up, 0, 0, conv3_down.shape[2], conv3_down.shape[3])
        conv4_up = torch.cat((conv3_down, conv4_up), dim=1)

        conv4_up = self.dconv_up4(conv4_up)
        conv3_up = F.interpolate(conv4_up, scale_factor=2, mode="nearest")
        conv3_up = TF.crop(conv3_up, 0, 0, conv2_down.shape[2], conv2_down.shape[3])
        conv3_up = torch.cat((conv2_down, conv3_up), dim=1)

        conv3_up = self.dconv_up3(conv3_up)
        conv2_up = F.interpolate(conv3_up, scale_factor=2, mode="nearest")
        conv2_up = TF.crop(conv2_up, 0, 0, conv1_down.shape[2], conv1_down.shape[3])
        conv2_up = torch.cat((conv1_down, conv2_up), dim=1)

        conv2_up = self.dconv_up2(conv2_up)

        conv1_up = self.dconv_up1(conv2_up)

        out = torch.tanh(conv1_up)

        return out


def AlexNet_Discriminator(num_classes):
    model = models.AlexNet(num_classes=num_classes)

    # Make single input channel
    model.features = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(64, 192, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(192, 384, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
    )
    # Change number of output classes to num_classes
    model.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(256 * 6 * 6, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(1024, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, num_classes),
    )

    return model

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Calculate positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (won't be updated during backprop but saved in state_dict)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to input
        return x + self.pe[:, :x.size(1)]

class TransformerDiscriminator(nn.Module):
    def __init__(self, num_classes, d_model=128, nhead=4, num_layers=3, dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        # Initial convolutional layer to process spectrograms
        # Input shape: [batch_size, 1, height, width]
        self.conv_embed = nn.Sequential(
            nn.Conv2d(1, d_model//2, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(d_model//2),
            nn.ReLU(),
            nn.Conv2d(d_model//2, d_model, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(d_model),
            nn.ReLU()
        )
        
        # Positional encoding layer
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        
        # Stack of transformer encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        # self.classifier = nn.Sequential(
        #     nn.Linear(d_model, dim_feedforward),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(dim_feedforward, dim_feedforward // 2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(dim_feedforward // 2, num_classes)
        # )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
        
    def forward(self, x):
        # Pass through CNN layers
        x = self.conv_embed(x)  # [batch_size, d_model, h', w']
        
        # Reshape for transformer
        batch_size, d_model, h, w = x.shape
        x = x.permute(0, 2, 3, 1)  # [batch_size, h', w', d_model]
        x = x.reshape(batch_size, h * w, d_model)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer
        x = self.transformer_encoder(x)
        
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)
        
        x = self.classifier(x)
        
        return x
