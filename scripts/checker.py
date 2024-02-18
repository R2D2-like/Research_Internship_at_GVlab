import h5py
file_path = "/root/Research_Internship_at_GVlab/demo.hdf5" # HDF5ファイルへのパスを指定
with h5py.File(file_path, 'r') as f:
    print(list(f.keys()))  # ファイル内のトップレベルオブジェクトをリストアップ
