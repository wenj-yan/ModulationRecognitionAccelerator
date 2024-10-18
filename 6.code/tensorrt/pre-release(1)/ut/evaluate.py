import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as rt
from utils import read_dataset,trunk_normalize
model_path = "logs/model-AugMod-LModCNNResNetRelu-trained128.h5"
#从文件中加载模型
model = tf.keras.models.load_model(model_path)
# 检查模型
model.summary()
# 加载模型
# sess = rt.InferenceSession("augmod128.onnx")

# # 获取模型的输入和输出名称
# input_name = sess.get_inputs()[0].name
# output_name = sess.get_outputs()[0].name

# 假设你有一个名为input_data的numpy数组作为输入
# input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

signals, class_onehot, snrs, class_list = read_dataset("AugMod", "./dataset")

signals = trunk_normalize(signals,128,"Augod")
# 随机取出一部分数据


num_samples = 100
random_indices = np.random.choice(len(signals), num_samples, replace=False)
random_signals = signals[random_indices]
random_class_onehot = class_onehot[random_indices]
print(random_signals.shape)
# 预测随机数据
# 运行模型
# random_pred = sess.run([output_name], {input_name: signals})

random_pred = model.predict(random_signals)
pred = np.argmax(random_pred, axis=1)
true = np.argmax(random_class_onehot, axis=1)

# 计算准确率
acc = np.mean(pred == true)
print(f"Random Accuracy: {acc}")

# 绘制混淆矩阵
confusion_matrix = tf.math.confusion_matrix(true, pred).numpy()

# 绘制混淆矩阵
plt.figure(figsize=(10, 8))
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()

# 添加标签
tick_marks = np.arange(len(confusion_matrix))
plt.xticks(tick_marks, rotation=45)
plt.yticks(tick_marks)

plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.show()

