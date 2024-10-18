import numpy as np
from os.path import join
import pickle
from tensorflow.keras.utils import to_categorical
from pythagore_modreco.data import read_augmod, read_RML2016, read_RML2018
from msilib.schema import File
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
# 读取原始文件
def read_raw_data(file_path,downsample:int=4)->None:
    '''
    读取单个文件，降采样，保存为pkl文件，输出格式为(batch,1024,2)     
    预处理原始数据.raw格式文件，转换为字典类的pkl文件，降采样比例为downsample：int
    file_path: str, 原始文件路径
    downsample: int, 降采样比例
    
    '''
    raw_data = np.fromfile(file_path, dtype=np.float32)
    # 将I和Q分量组合成复数形式
    i_data = raw_data[::2]
    q_data = raw_data[1::2]
    # 降采样
    i_data = i_data[::downsample]
    q_data = q_data[::downsample]
    
    iq = np.stack([i_data,q_data],axis=1)
    t = []
    print(iq.shape)#(33074688,2)

    for i in range(i_data.shape[0]//1024):
        trunk = iq[i*1024:(i+1)*1024,:]
        t.append(trunk)
    print(len(t))
    print(t[0].shape)
    data = np.array(t)
    print(data.shape)
    labels = np.zeros(data.shape[0])

    dict = {}
    dict['data'] = data
    dict['label'] = labels
    dict['label_names'] = ['fm_90.5']

    with open('./data.pkl','wb') as f:
        pickle.dump(dict,f)

def load_data(file):
    with open(file,'rb') as f:
        data = pickle.load(f)
    return data
def plot_data(data):
    '''
    逐个绘制显示时域波形，用来检查数据是否正确
    data: dict, pkl文件读出的data数据
    '''
    data_i = data["data"][:,:,1]
    data_q = data["data"][:,:,0]
    labels = data["label"]
    figure = plt.figure()
    plt.ion()
    plt.show()
    for i in range(data_i.shape[0]):
        figure.clear()
        plt.title(f"sample:{labels[i]}")
        plt.plot(data_i[i],label='I')
        plt.plot(data_q[i],label='Q')
        figure.canvas.draw()
        plt.pause(0.01)

def read_batch(file_path,downsample,file_name,verbose=False):
    '''
    从文件夹内读取各类的raw文件存成pkl类数据集
    file_path: str, 原始文件路径
    downsample: int, 降采样比例
    file_name: str, 保存的pkl文件名
    verbose: bool, 是否打印信息
    ！注意将文件夹按照如下方式排列：
    file_path
    ├── class1
    │   ├── file1.raw
    │   ├── file2.raw
    │   └── ...
    ├── class2
    │   ├── file1.raw
    │   ├── file2.raw
    │   └── ...
    └── ...
    '''
    t = []
    class_idx = 0
    labels = []
    label_names = []
    for folder in os.listdir(file_path):
        for file in os.listdir(os.path.join(file_path,folder)):
            if file.endswith('.raw'):
                raw_data = np.fromfile(os.path.join(file_path,folder,file), dtype=np.float32)
                # 将I和Q分量组合成复数形式
                i_data = raw_data[::2]
                q_data = raw_data[1::2]
                # 降采样
                i_data = i_data[::downsample]
                q_data = q_data[::downsample]
                # 拼接组合
                iq = np.stack([i_data,q_data],axis=1) #(33074688,2)
        
                for i in range(i_data.shape[0]//1024):
                    trunk = iq[i*1024:(i+1)*1024,:]
                    t.append(trunk)
                    labels.append(class_idx)
            
                if verbose:
                    print(f"file:{file} has been processed. with shape of {len(t)} - {len(labels)} of {class_idx}")
            else:
                print(f"file:{file} is not a .raw file,skipped.")
                continue
        label_names.append(folder)
        class_idx += 1
        
        
    
    data = np.array(t)
    dict = {}
    dict['data'] = data
    dict['label'] = labels
    dict['sample_rate'] = 20000000//downsample
    dict['label_names'] = label_names
    
    if verbose:
        bincount = np.bincount(labels)
        print("各类别样本数量：")
        print(bincount)
        
        print(f"all files have been processed of {class_idx} classes. with shape of {data.shape} - {len(labels)}")
    with open(file_name,'wb') as f:
        pickle.dump(dict,f)
        print(f"file:{file} has been saved as data.pkl")

def read_my(fname):
    '''
    读取自制数据集：按照pickle存储的字典类型数据集
    输出signals格式是（batch_size, 1024,2）
    '''
    with open(fname, "rb") as f:
        data = pickle.load(f)
    signals = data["data"]
    lables = data["label"]
    sample_rate = data["sample_rate"]
    return signals, lables,sample_rate

def read_dataset(dataset_name, data_path):
    '''
    Read and process a dataset based on the dataset name.

    Parameters:
    - dataset_name (str): The name of the dataset to be read.
    - data_path (str): The path to the dataset.

    Returns:
    - signals (ndarray): The signals from the dataset.
    - class_onehot (ndarray): The one-hot encoded class labels.
    - snrs (ndarray): The signal-to-noise ratios.
    - class_list (list): The list of class labels.

    Raises:
    - None

    '''
    if dataset_name == "AugMod":

        fName = join(data_path, "augmod.hdf5")
        data_dict = read_augmod(fName)

        signals = data_dict["signals"]
        class_idx = data_dict["modulations"]
        print(class_idx)
        snrs = data_dict["snr"]

        class_onehot = to_categorical(class_idx)
        class_list = data_dict["classes"]
        
        return signals, class_onehot, snrs, class_list

    elif dataset_name == "RadioML2016.04c":

        fName = join(data_path, "2016.04C.multisnr.pkl")

        signals, class_idx, snrs, class_list = read_RML2016(fName)
        class_onehot = to_categorical(class_idx)

        return signals, class_onehot, snrs, class_list


    elif dataset_name == "RadioML2016.10a":

        fName = join(data_path, "RML2016.10a_dict.pkl")

        signals, class_idx, snrs, class_list = read_RML2016(fName)
        class_onehot = to_categorical(class_idx)
        
        return signals, class_onehot, snrs, class_list
    
    elif dataset_name == "RadioML2016.10b":

        fName = join(data_path, "RML2016.10b.dat")

        signals, class_idx, snrs, class_list = read_RML2016(fName)
        class_onehot = to_categorical(class_idx)
        
        return signals, class_onehot, snrs, class_list
    elif dataset_name == "RadioML2018.01a":

        fName = join(data_path, "GOLD_XYZ_OSC.0001_1024.hdf5")

        signals, class_onehot, snrs, class_list = read_RML2018(fName)
        # snrs = snrs.reshape(-1)
        class_idx = np.argmax(class_onehot, axis=-1)
        
        return signals, class_onehot, snrs, class_list
    elif dataset_name == "my_dataset":
        fname  = join(data_path, "data.pkl")
        signals,class_idx,sample_rate = read_my(fname)
        class_onehot = to_categorical(class_idx)
        snrs = np.zeros(len(class_idx))
        class_list = []
        return signals, class_onehot, snrs, class_list
    else:
        print("Data not found")
        

def trunk_normalize(signals, signal_duration,dataset_name,snr_cut=None):
    '''
    Truncate the signals to the specified duration and normalize them.

    Parameters:
    - signals (ndarray): The signals to be truncated and normalized.
    - signal_duration (int): The duration to which the signals should be truncated.

    Returns:
    - signals (ndarray): The truncated and normalized signals.

    Raises:
    - None

    '''
    if not signal_duration is None:
        print(f"Initial data shape: {signals.shape}")
        signals = signals[:, :signal_duration, :]
        print(f"Truncated data shape: {signals.shape}")
        
    if not snr_cut is None:
        print(f"Initial data shape: {signals.shape}")
        w = snrs >= snr_cut
        signals = signals[w]
        class_idx = class_idx[w]
        snrs = snrs[w]
        class_onehot = class_onehot[w]
        print(f"New data shape: {signals.shape}")

    if dataset_name != "my_dataset":
        # Normalize the signals
        print("transposing signals")
        print(f"Initial data shape: {signals.shape}")
        signals = signals.transpose((0, 2, 1))
        print(f"Transposed data shape: {signals.shape}")
    
    norm = np.sqrt(np.mean(signals**2, axis=(1, 2), keepdims=True))
    signals /= norm
    return signals

def save_to_pickle(signals, class_onehot, snrs, class_list):
    data = dict()
    data["signals"]=signals
    data["label"]=class_onehot
    data["snr"]=snrs
    data["classes"]=class_list
    with open("data.pkl","wb") as f:
        pickle.dump(data,f)
    print("Data saved to data.pkl")
    return

def read_pickle(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data["signals"], data["label"], data["snr"], data["classes"]
    
if __name__ == "__main__":
    '''数据集转换可以参考这个例子'''
    # PC端保存
    # signals, class_onehot, snrs, class_list = read_dataset("AugMod", "./dataset")
    # #signals = trunk_normalize(signals, 1024, "my_dataset")
    # save_to_pickle(signals, class_onehot, snrs, class_list)
    # # Jetson端读取
    # signals, class_onehot, snrs, class_list = read_pickle("data.pkl")
    # print(signals.shape)
    # print(class_onehot.shape)
    # print(snrs.shape)
    # print(class_list)
    # '''自制数据集可以参考下面'''
    # file_path = "E:\\2024_spring\jetson\codes\pythagore-mod-reco\make_data\\signal"
    # read_batch(file_path,4,'./data.pkl',verbose=True)
    # data = load_data('data.pkl')
    # print(data['data'].shape)
    # print(len(data['label']))
    # print(data['label'])
    # print(data['label_names'])
    # plot_data(data)
    '''
    tf的.h5模型转换为saved_model格式，再从saved_model格式转换为onnx格式可以参考下面
    需要提前安装tf2onnx库 pip install tf2onnx
    
    '''
    from convert import tf_model_to_saved_model
    model_path = "./res.h5"
    saved_model_path = "saved_model_res"
    import subprocess
    tf_model_to_saved_model(model_path, saved_model_path)
    onnx_model_path = "res.onnx"
    subprocess.run(f"python -m tf2onnx.convert --saved-model {saved_model_path} --output {onnx_model_path} --opset 11")
    
    '''
    2024-05-12 16:16:51,225 - INFO - Using tensorflow=2.16.1, onnx=1.16.0, tf2onnx=1.16.1/15c810
    2024-05-12 16:16:51,225 - INFO - Using opset <onnx, 15>
    2024-05-12 16:16:51,337 - INFO - Computed 0 values for constant folding
    2024-05-12 16:16:51,439 - INFO - Optimizing ONNX model
    2024-05-12 16:16:51,782 - INFO - After optimization: Add -2 (4->2), Cast -3 (3->0), Concat -1 (1->0), Const -13 (23->10), Identity -2 (2->0), Shape -1 (1->0), Slice -1 (1->0), Squeeze -1 (1->0), Transpose -3 (4->1), Unsqueeze -4 (4->0)
    2024-05-12 16:16:51,803 - INFO - 
    2024-05-12 16:16:51,803 - INFO - Successfully converted TensorFlow model saved_model_cnn to ONNX
    2024-05-12 16:16:51,804 - INFO - Model inputs: ['inputs']
    2024-05-12 16:16:51,804 - INFO - Model outputs: ['output_0']
    2024-05-12 16:16:51,804 - INFO - ONNX model is saved at cnn.onnx
    '''
    
    
    
    