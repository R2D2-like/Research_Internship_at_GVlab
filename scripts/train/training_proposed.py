import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from train.lfd_proposed import MotionDecoder
import numpy as np
from config.values import *
import os

def load_data(vae_data_path, tcn_data_path):
    exp_action_data = np.load(vae_data_path) #npz file
    demo_data = np.load(tcn_data_path) #npz file

    vae_data, tcn_data = None, None
    for sponge in TRAIN_SPONGES_LIST:
        if vae_data is None:
            vae_data = np.expand_dims(exp_action_data[sponge], axis=0) #(1, 400, 6)
            tcn_data = np.expand_dims(demo_data[sponge], axis=0) #(1, 9, 2000)
        else:
            vae_data = np.concatenate([vae_data, np.expand_dims(exp_action_data[sponge], axis=0)], axis=0)
            tcn_data = np.concatenate([tcn_data, np.expand_dims(demo_data[sponge], axis=0)], axis=0)

    return vae_data, tcn_data

def data_loader(vae_data, tcn_data, batch_size=32):
    '''
    vae_data は (N, 400, 6), tcn_data は (N, 9, 2000) の形状のテンソル
    (N=len(TRAIN_SPONGES_LIST)*DATA_PER_SPONGE)
    
    ・vae_encoderの入力データは(B, 400, 6)
    ・tcnの入力データは(B, 9, T_length=20) 
    ・target（正解データ）は(B, 3)(tcn_inputの9の最初の3要素)

    return: vae_inputs, tcn_inputs, targets
    '''
    fixed_T_length = 100  # TCNの入力データの固定長
    
    # TCNデータ用のインデックスを選択し、固定長を適用
    tcn_indices = torch.randint(0, tcn_data.shape[0], (batch_size,))
    # tcn_indicesの値が0~DATA_PER_SPONGE-1の範囲なら0, DATA_PER_SPONGE~2*DATA_PER_SPONGE-1の範囲なら1, ...となるように変換
    vae_indices = tcn_indices // DATA_PER_SPONGE
    # 対応するvae_dataを取得
    vae_inputs = vae_data[vae_indices]

    tcn_inputs = torch.zeros(batch_size, 9, fixed_T_length)
    end_indices = torch.randint(fixed_T_length, tcn_data.shape[2], (batch_size,))
    targets = torch.zeros(batch_size, 3)
    for i, idx in enumerate(tcn_indices):
        # データの選択範囲をランダムに設定（終点は最後から固定長を引いた位置）
        end_index = end_indices[i].item()
        start_index = end_index - fixed_T_length
        tcn_inputs[i, :, :] = tcn_data[idx, :, start_index:end_index]
        # ターゲットをTCN入力データの次のタイムステップから生成
        targets[i] = tcn_data[idx, :3, end_index]

    return vae_inputs, tcn_inputs, targets

# VAEの入力: (B, 400, 6), TCNの入力: (B, 9, 2000), 正解データ: (B, 3)
batch_size = 32

'''
# ダミーデータセットの作成
dummy_vae_input = torch.randn(batch_size, 400, 6)
dummy_tcn_input = torch.randn(batch_size, 9, 2000)
dummy_targets = torch.randn(batch_size, 3)
'''

# load path
vae_encoder_path = '/root/Research_Internship_at_GVlab/sim/model/vae_encoder.pth' # VAEのエンコーダーの重みへのパス
vae_data_path = '/root/Research_Internship_at_GVlab/real/step1/data/exploratory_action_preprocessed.npz'
tcn_data_path = '/root/Research_Internship_at_GVlab/real/step2/data/demo_preprocessed.npz'

# save path
dir = '/root/Research_Internship_at_GVlab/real/model/proposed/'
if not os.path.exists(dir):
    os.makedirs(dir)
model_path = dir + 'proposed_model.pth'
decoder_path = dir + 'proposed_decoder.pth'

# load data
vae_data, tcn_data = load_data(vae_data_path, tcn_data_path)

vae_data = torch.tensor(vae_data) # (N, 400, 6)
tcn_data = torch.tensor(tcn_data) # (N, 9, 2000)

# モデル、損失関数、オプティマイザの設定
model = MotionDecoder(vae_encoder_path=vae_encoder_path)
criterion = torch.nn.MSELoss()  # 平均二乗誤差損失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# デバイスの設定（GPUが利用可能な場合はGPUを使用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 学習ループ
num_epochs = 10000  # エポック数
for epoch in range(num_epochs):
    model.train()  # モデルを訓練モードに設定
    for vae_inputs, tcn_inputs, targets in data_loader(vae_data, tcn_data, batch_size):
        vae_inputs, tcn_inputs, targets = vae_inputs.to(device), tcn_inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()  # オプティマイザの勾配をゼロに設定
        
        # フォワードパス
        outputs = model(vae_inputs, tcn_inputs)
        
        # 損失の計算
        loss = criterion(outputs, targets)
        
        # バックプロパゲーション
        loss.backward()
        
        # オプティマイザの更新
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Loss {loss.item()}")

# モデルの保存
torch.save(model.state_dict(), model_path)
decoder_weights = {k: v for k, v in model.state_dict().items() if 'decoder' in k}
torch.save(decoder_weights, decoder_path)
print('Training complete.')
print('Model saved at', model_path)
