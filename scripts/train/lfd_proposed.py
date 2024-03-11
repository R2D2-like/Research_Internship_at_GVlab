import torch
from torch import nn
from torch.nn import functional as F
from types import *
from typing import List
from torch import Tensor
from vae import VAE
from tcn import TCN

class MotionDecoder(nn.Module):
    def __init__(self, vae_encoder_path, tcn_input_size=9, tcn_output_size=9, tcn_num_channels=[32, 64, 128, 256, 256, 516, 516], kernel_size=4, dropout=0.1, mlp_output_size=3):
        super(MotionDecoder, self).__init__()
        self.vae_encoder = VAE()  # VAEの初期化
        self.vae_encoder.load_state_dict(torch.load(vae_encoder_path), strict=False)  # 重みの読み込み
        for param in self.vae_encoder.parameters():  # エンコーダーの重みをフリーズ
            param.requires_grad = False
        
        self.tcn = TCN(input_size=tcn_input_size, output_size=tcn_output_size, num_channels=tcn_num_channels, kernel_size=kernel_size, dropout=dropout)  # TCNの初期化
        
        # VAEの潜在変数zの次元数とTCNの出力次元数の合計を入力とするMLP
        self.mlp = nn.Sequential(
            nn.Linear(5 + tcn_output_size, 128),  # 5 (VAEのzの次元) + 9 (TCNの出力次元)
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, mlp_output_size)  # 最終的な出力サイズ(B, 3)
        )
    
    def forward(self, vae_input, tcn_input):
        # VAEのエンコーダーを通して潜在変数zを取得
        mu, log_var = self.vae_encoder.encode(vae_input)
        z = self.vae_encoder.reparameterize(mu, log_var)  # (B, 5)
        
        # TCNを通して出力を取得
        tcn_output = self.tcn(tcn_input)  # (B, 9)
        
        # VAEの潜在変数zとTCNの出力を結合
        combined = torch.cat((z, tcn_output), dim=1)  # (B, 5+9)
        
        # 結合したベクトルをMLPに通す
        output = self.mlp(combined)  # (B, 3)
        
        return output

