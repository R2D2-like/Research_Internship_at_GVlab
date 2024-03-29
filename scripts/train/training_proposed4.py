import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from lfd_proposed import LfDProposed
import numpy as np
import sys
sys.path.append('/root/Research_Internship_at_GVlab/scripts/config')
from values import *
import os

def load_data(vae_data_path, ft_data_path, z_diff_data_path):
    exp_action_data = np.load(vae_data_path) #npz file
    ft_data = np.load(ft_data_path) #npz file
    z_diff_data = np.load(z_diff_data_path) #npz file

    vae_data, tcn_data, target_data = None, None, None
    for sponge in TRAIN_SPONGES_LIST:
        if vae_data is None:
            vae_data = exp_action_data[sponge] #(DATA_PER_SPONGE, 400, 6)
            #(DATA_PER_SPONGE, 9, 100) -> (DATA_PER_SPONGE, 6, 100)に変更 (idx 0~2のデータを取り除く)
            tcn_data = ft_data[sponge] # (DATA_PER_SPONGE, 6, 100)
            target_data = z_diff_data[sponge] #(DATA_PER_SPONGE, 99)

        else:
            vae_data = np.concatenate([vae_data, exp_action_data[sponge]], axis=0)
            tcn_data = np.concatenate([tcn_data, ft_data[sponge]], axis=0) # (N, 6, 100)
            target_data = np.concatenate([target_data, z_diff_data[sponge]], axis=0) #(N, 99)

    return vae_data, tcn_data, target_data

def data_loader(vae_data, tcn_data, target_data, batch_size=32):
    '''
    vae_data は (N, 400, 6), tcn_data は (N, 9, 100) の形状のテンソル
    (N=len(TRAIN_SPONGES_LIST)*DATA_PER_SPONGE)
    
    ・vae_encoderの入力データは(B, 400, 6)
    ・tcnの入力データは(B, 6, T_length=5) 
    ・target（正解データ）は(B, 1)

    return: vae_inputs, tcn_inputs, targets
    '''
    fixed_T_length = 5  # TCNの入力データの固定長
    
    # TCNデータ用のインデックスを選択し、固定長を適用
    tcn_indices = torch.randint(0, tcn_data.shape[0], (batch_size,))
    # tcn_indicesの値が0~DATA_PER_SPONGE-1の範囲なら0, DATA_PER_SPONGE~2*DATA_PER_SPONGE-1の範囲なら1, ...となるように変換
    vae_indices = tcn_indices // DATA_PER_SPONGE
    # 対応するvae_dataを取得
    vae_inputs = vae_data[vae_indices]

    tcn_inputs = torch.zeros(batch_size, 6, fixed_T_length)
    end_indices = torch.randint(fixed_T_length, tcn_data.shape[2]-1, (batch_size,))
    targets = torch.zeros(batch_size, 1)
    for i, idx in enumerate(tcn_indices):
        # データの選択範囲をランダムに設定（終点は最後から固定長を引いた位置）
        end_index = end_indices[i].item()
        start_index = end_index - fixed_T_length
        tcn_inputs[i, :, :] = tcn_data[idx, :, start_index:end_index]
        targets[i] = target_data[idx, end_index]
        # print(targets[i])

    return vae_inputs, tcn_inputs, targets

# VAEの入力: (B, 400, 6), TCNの入力: (B, 9, 100), 正解データ: (B, 3)
batch_size = 32

'''
# ダミーデータセットの作成
dummy_vae_input = torch.randn(batch_size, 400, 6)
dummy_tcn_input = torch.randn(batch_size, 9, 100)
dummy_targets = torch.randn(batch_size, 3)
'''

# load path
vae_encoder_path = '/root/Research_Internship_at_GVlab/sim/model/vae_encoder.pth' # VAEのエンコーダーの重みへのパス
vae_data_path = '/root/Research_Internship_at_GVlab/real/step1/data/exploratory_action_preprocessed.npz'
ft_data_path = '/root/Research_Internship_at_GVlab/real/step2/data/demo_preprocessed_ft.npz'
z_diff_data_path = '/root/Research_Internship_at_GVlab/real/step2/data/demo_preprocessed_z_diff.npz'

# save path
dir = '/root/Research_Internship_at_GVlab/real/model/proposed/'
if not os.path.exists(dir):
    os.makedirs(dir)
model_path = dir + 'proposed_model4.pth'
decoder_path = dir + 'proposed_decoder4.pth'

# load data
vae_data, tcn_data, target_data = load_data(vae_data_path, ft_data_path, z_diff_data_path)

vae_data = torch.tensor(vae_data, dtype=torch.float32) # (N, 400, 6)
tcn_data = torch.tensor(tcn_data, dtype=torch.float32) # (N, 8, 100)
target_data = torch.tensor(target_data, dtype=torch.float32) # (N, 9, 100)
print(tcn_data.size())

# モデル、損失関数、オプティマイザの設定
model = LfDProposed(vae_encoder_path=vae_encoder_path, tcn_input_size=6, tcn_output_size=7, mlp_output_size=1)
criterion = torch.nn.MSELoss()  # 平均二乗誤差損失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# デバイスの設定（GPUが利用可能な場合はGPUを使用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()  # モデルを訓練モードに設定

# 学習ループ
num_epochs = 2000  # エポック数
for epoch in range(num_epochs):
    vae_inputs, tcn_inputs, targets = data_loader(vae_data, tcn_data, target_data, batch_size)
    vae_inputs, tcn_inputs, targets = vae_inputs.to(device), tcn_inputs.to(device), targets.to(device)
    
    optimizer.zero_grad()  # オプティマイザの勾配をゼロに設定
    
    # フォワードパス
    outputs = model(vae_inputs, tcn_inputs)
    print(outputs)
    
    # 損失の計算
    loss = criterion(outputs, targets)
    
    # バックプロパゲーション
    loss.backward()
    
    # オプティマイザの更新
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss {loss.item()}")

# モデルの保存
torch.save(model.state_dict(), model_path)
decoder_weights = {k: v for k, v in model.state_dict().items() if 'decoder' in k}
torch.save(decoder_weights, decoder_path)
print('Training complete.')
print('Model saved at', model_path)