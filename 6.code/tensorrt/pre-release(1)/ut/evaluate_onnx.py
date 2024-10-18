
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from h5py import File
from numpy import array
from tqdm import tqdm
def trunk_normalize(signals, signal_duration=128,dataset_name='augmod',snr_cut=None):
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
        
    print(f"Initial data shape: {signals.shape}")
    signals = signals[:, :signal_duration, :]
    print(f"Truncated data shape: {signals.shape}")
    norm = np.sqrt(np.mean(signals**2, axis=(1, 2), keepdims=True))
    signals /= norm
    return signals


def read_augmod(path):
    data = dict()
    with File(path, "r") as f:
        data["classes"] = [c.decode() for c in f["classes"]]
        data["signals"] = array(f["signals"])
        data["modulations"] = array(f["modulations"])
        data["snr"] = array(f["snr"])
        data["frequency_offsets"] = array(f["frequency_offsets"])
    return data["signals"],data["modulations"] 




# model_path = "logs/model-AugMod-LModCNNResNetRelu-trained128.h5"
# # 从文件中加载模型
# model = tf.keras.models.load_model(model_path)
# # 检查模型
# model.summary()

# 获取测试集
# signals, class_onehot, snrs, class_list = read_dataset("AugMod", "./dataset")
signals,classes = read_augmod("dataset/augmod.hdf5")
signals = trunk_normalize(signals)
# 随机取出一部分数据
num_samples = 1000
random_indices = np.random.choice(len(signals), num_samples, replace=False)
random_signals = signals[random_indices]
random_class = classes[random_indices]


import onnxruntime as rt
sess = rt.InferenceSession("onnx128.onnx")
print("Model has been loaded.")

preds = []
r = 0
w = 0
for signal,gt in tqdm(zip(random_signals,random_class)):
    signal = signal.reshape(1,128,2)
    signal = signal.astype(np.float32)
    c
    pred = np.argmax(output[0])
    preds.append(pred)
    if pred == gt:
       r += 1
    else:
       w += 1
    #print(f"Predicted: {pred} Ground Truth: {gt} answer: {pred == gt}")
print(f"Correct: {r} Wrong: {w}")
print(f"Predictions: {np.bincount(preds==random_class)}")   
    
true = random_class

# 计算准确率
acc = np.mean(preds == true)
print(f"Random Accuracy: {acc}")

# 绘制混淆矩阵
cm = confusion_matrix(y_true=true, y_pred=preds, normalize='pred')
# 绘制混淆矩阵
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()

# 添加标签
tick_marks = np.arange(len(cm))
plt.xticks(tick_marks, rotation=45)
plt.yticks(tick_marks)

# 添加数值标签
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.show()


