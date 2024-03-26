import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from lfd_proposed import LfDProposed
import numpy as np
import sys
sys.path.append('/root/Research_Internship_at_GVlab/scripts/config')
from values import *
import os

def load_data(vae_data_path, tcn_data_path):
    exp_action_data = np.load(vae_data_path) #npz file
    demo_data = np.load(tcn_data_path) #npz file

    vae_data, tcn_data, target_data = None, None, None
    for sponge in TRAIN_SPONGES_LIST:
        if vae_data is None:
            vae_data = exp_action_data[sponge] #(DATA_PER_SPONGE, 400, 6)
            target_data = demo_data[sponge] #(DATA_PER_SPONGE, 9, 2000)
            #(DATA_PER_SPONGE, 9, 2000) -> (DATA_PER_SPONGE, 8, 2000)に変更 (idx 2のデータを取り除く)
            tcn_data = np.delete(target_data, 2, axis=1) # (DATA_PER_SPONGE, 8, 2000)

        else:
            vae_data = np.concatenate([vae_data, exp_action_data[sponge]], axis=0)
            target_data = np.concatenate([target_data, demo_data[sponge]], axis=0)
            tcn_data = np.concatenate([tcn_data, np.delete(target_data, 2, axis=1)], axis=0)

    return vae_data, tcn_data, target_data

def data_loader(vae_data, tcn_data, target_data, batch_size=32):
    '''
    vae_data は (N, 400, 6), tcn_data は (N, 9, 2000) の形状のテンソル
    (N=len(TRAIN_SPONGES_LIST)*DATA_PER_SPONGE)
    
    ・vae_encoderの入力データは(B, 400, 6)
    ・tcnの入力データは(B, 9, T_length=20) 
    ・target（正解データ）は(B, 3)(tcn_inputの9の最初の3要素)

    return: vae_inputs, tcn_inputs, targets
    '''
    fixed_T_length = 2000  # TCNの入力データの固定長
    
    # TCNデータ用のインデックスを選択し、固定長を適用
    tcn_indices = torch.randint(0, tcn_data.shape[0], (batch_size,))
    # tcn_indicesの値が0~DATA_PER_SPONGE-1の範囲なら0, DATA_PER_SPONGE~2*DATA_PER_SPONGE-1の範囲なら1, ...となるように変換
    vae_indices = tcn_indices // DATA_PER_SPONGE
    # 対応するvae_dataを取得
    vae_inputs = vae_data[vae_indices]

    tcn_inputs = torch.zeros(batch_size, 8, fixed_T_length)
    end_indices = torch.randint(0, tcn_data.shape[2], (batch_size,))
    targets = torch.zeros(batch_size, 3)
    for i, idx in enumerate(tcn_indices):
        # データの選択範囲をランダムに設定（終点は最後から固定長を引いた位置）
        end_index = end_indices[i].item()
        start_index = end_index - fixed_T_length
        if start_index < 0:
            start_index = 0
        # tcn_inputs[i, :, :] = tcn_data[idx, :, start_index:end_index]
        # 足りない部分は0で埋める
        tcn_inputs[i, :, :end_index-start_index] = tcn_data[idx, :, start_index:end_index]
        # ターゲットをTCN入力データの次のタイムステップから生成
        # x,y(idx=0,1)は絶対値を取る
        targets[i][:2] = target_data[idx, :2, end_index]
        # z(idx=2)は次のタイムステップの値-現在のタイムステップの値の差を取る
        targets[i][2] = target_data[idx, 2, end_index] - target_data[idx, 2, end_index-1]
        

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
tcn_data_path = '/root/Research_Internship_at_GVlab/real/step2/data/demo_preprocessed2.npz'

# save path
dir = '/root/Research_Internship_at_GVlab/real/model/proposed/'
if not os.path.exists(dir):
    os.makedirs(dir)
model_path = dir + 'proposed_model2.pth'
decoder_path = dir + 'proposed_decoder2.pth'

# load data
vae_data, tcn_data, target_data = load_data(vae_data_path, tcn_data_path)

vae_data = torch.tensor(vae_data, dtype=torch.float32) # (N, 400, 6)
tcn_data = torch.tensor(tcn_data, dtype=torch.float32) # (N, 8, 2000)
target_data = torch.tensor(target_data, dtype=torch.float32) # (N, 9, 2000)
print(tcn_data.size())

# モデル、損失関数、オプティマイザの設定
model = LfDProposed(vae_encoder_path=vae_encoder_path, tcn_input_size=8)
criterion = torch.nn.MSELoss()  # 平均二乗誤差損失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# デバイスの設定（GPUが利用可能な場合はGPUを使用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()  # モデルを訓練モードに設定

# 学習ループ
num_epochs = 1000  # エポック数
for epoch in range(num_epochs):
    vae_inputs, tcn_inputs, targets = data_loader(vae_data, tcn_data, target_data, batch_size)
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
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss {loss.item()}")

# モデルの保存
torch.save(model.state_dict(), model_path)
decoder_weights = {k: v for k, v in model.state_dict().items() if 'decoder' in k}
torch.save(decoder_weights, decoder_path)
print('Training complete.')
print('Model saved at', model_path)
