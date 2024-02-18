import h5py

# HDF5ファイルを開く
file_path = "/root/demo2/1708228689_5592735/demo.hdf5"
# file_path = "/root/external/robosuite/robosuite/models/assets/demonstrations/wipe/panda/demo.hdf5"
with h5py.File(file_path, 'r') as file:
    # ファイル内の全キーを表示
    print(list(file.keys()))
    # 'data' や 'mask' オブジェクトにアクセスする例
    if 'data' in file:
        data = file['data']
        print(data)
    if 'mask' in file:
        mask = file['mask']
        print(mask)
