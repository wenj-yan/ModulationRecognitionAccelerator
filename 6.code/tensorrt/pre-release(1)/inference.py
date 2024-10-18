# from tensorflow.keras.models import load_model
import subprocess
import pdb
import keyboard # for hotkey handling
import numpy as np
import os
import matplotlib.pyplot as plt
from multiprocessing import Queue,Process
import threading

import time
from os import mkdir
from os.path import exists as path_exists
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
# using different libraries for spectrogram drawing
#import librosa 
from scipy.signal import spectrogram
import socket

# -----------------------------------change the following parameters-----------------------------------#
TEST = False #using udp testcase
CNN = False   #'AM','ASK','DRM','FM'
CLASS_LABEL = ['FM','AM','DRM','ASK']#['BPSK', 'PSK8', 'QAM16', 'QAM32', 'QAM64', 'QAM8', 'QPSK'] 
SAMPLE_RATE = 2e6 #sample rate
N_FFT = 128 #fft size
MODEL_ADDROOT = "./model"
MODEL_NAME = "model-AugMod-LModCNNResNetRelu-trained128.h5"
DESTINATION_PORT = 2024
IP_ADDRESS = "localhost"
ROOT = "./model"
ONNX_FNAME = "./onnx128.onnx"
# -----------------------------------import runtime for trt/onnx/tf2-----------------------------------#

TENSORFLOW_ENABLED = True 
ONNX_ENABLED = True
try:
    print("Trying to import tensorflow model")
    from pythagore_modreco import neural_nets_kerasss    
except ImportError:
    TENSORFLOW_ENABLED = False
    print("Tensorflow not found, trying to import ONNX model")
    try:
        import onnxruntime
    except ImportError:
        ONNX_ENABLED = False
        print("Using TensorRT : not thread-safe")
        
        #import pycuda
        import pycuda.driver as cuda
        import pycuda.autoinit
        #from pycuda import autoinit
        import tensorrt as trt
        print(f"loading runtime using: tf{TENSORFLOW_ENABLED} onnx{ONNX_ENABLED}")
        print(f"tensorRT version: {trt.__version__}")
print(f"loading runtime using: tf{TENSORFLOW_ENABLED} onnx{ONNX_ENABLED}")


# -----------------------------------models definations---------------------------------------------#
class build_engine():
    def __init__(self, onnx_path):
        super(build_engine, self).__init__()
        self.onnx = onnx_path
        self.engine = self.onnx2engine() # 调用 onnx2engine 函数生成 engine
        
    def onnx2engine(self):
        # 创建日志记录器
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # 显式batch_size，batch_size有显式和隐式之分
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        
        # 创建builder，用于创建network
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(EXPLICIT_BATCH) # 创建network（初始为空）
        
        # 创建config
        config = builder.create_builder_config()
        profile = builder.create_optimization_profile() # 创建profile
        profile.set_shape("inputs", (1,128,2), (1,128,2), (1,128,2))  # 设置动态输入,分别对应:最小尺寸、最佳尺寸、最大尺寸
        config.add_optimization_profile(profile)
	
        config.set_flag(trt.BuilderFlag.FP16)
        print('FP16:',config.get_flag(trt.BuilderFlag.FP16))
        #config.set_flag(trt.BuilderFlag.TF32)
        print('TF32:',config.get_flag(trt.BuilderFlag.TF32))
        #config.max_workspace_size = 1<<30 # 允许TensorRT使用1GB的GPU内存，<<表示左移，左移30位即扩大2^30倍，使用2^30 bytes即 1 GB
        
        # 创建parser用于解析模型
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # 读取并解析模型
        onnx_model_file = self.onnx # Onnx模型的地址
        model = open(onnx_model_file, 'rb')
        if not parser.parse(model.read()): # 解析模型
            for error in range(parser.num_errors):
                print(parser.get_error(error)) # 打印错误（如果解析失败，根据打印的错误进行Debug）
 
        # 创建序列化engine
        engine = builder.build_serialized_network(network, config)
        return engine
        
    def get_engine(self):
        return self.engine # 返回 engine

    
# 分配内存缓冲区
def Allocate_memory(engine, context,input_data):
    bindings = []
    for binding in engine:
        binding_idx = engine.get_binding_index(binding) # 遍历获取对应的索引
        
        size = trt.volume(context.get_binding_shape(binding_idx))
        # context.get_binding_shape(binding_idx): 获取对应索引的Shape，例如input的Shape为(1, 3, H, W)
        # trt.volume(shape): 根据shape计算分配内存 
        
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # engine.get_binding_dtype(binding): 获取对应index或name的类型
        # trt.nptype(): 映射到numpy类型
        
        if engine.binding_is_input(binding): # 当前index为网络的输入input
            input_buffer = np.ascontiguousarray(input_data) # 将内存不连续存储的数组转换为内存连续存储的数组，运行速度更快
            input_memory = cuda.mem_alloc(input_data.nbytes) # cuda.mem_alloc()申请内存
            bindings.append(int(input_memory))
        else:
            output_buffer = cuda.pagelocked_empty(size, dtype)
            output_memory = cuda.mem_alloc(output_buffer.nbytes)
            bindings.append(int(output_memory))
            
    return input_buffer, input_memory, output_buffer, output_memory, bindings


class TensorRTModel_v:
    def __init__(self, onnx_model_file):
        self.onnx_model_file = onnx_model_file
        
        self.cfx = cuda.Device(1).make_context()
        stream = cuda.Stream()

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        runtime = trt.Runtime(TRT_LOGGER)

        # # deserialize engine
        # with open(trt_engine_path, 'rb') as f:
        #     buf = f.read()
        #     engine = runtime.deserialize_cuda_engine(buf)
        engine_build = build_engine(onnx_model_file)
        engine = engine_build.get_engine()
        context = engine.create_execution_context()

        # prepare buffer
        host_inputs  = []
        cuda_inputs  = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(cuda_mem))
            if engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # store
        self.stream  = stream
        self.context = context
        self.engine  = engine

        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings


    def predict(self, input):
        threading.Thread.__init__(self)
        self.cfx.push()

        # restore
        stream  = self.stream
        context = self.context
        engine  = self.engine

        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        # read image
        #image = 1 - (np.asarray(Image.open(input_img_path), dtype=np.float)/255)
        np.copyto(host_inputs[0],input)

        # inference
        start_time = time.time()
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()
        print("execute times "+str(time.time()-start_time))

        # # parse output
        # output = np.array([math.exp(o) for o in host_outputs[0]])
        # output /= sum(output)
        # for i in range(len(output)): print("%d: %.2f"%(i,output[i]))

        self.cfx.pop()
        return host_outputs[0]

    def destory(self):
        self.cfx.pop()


def Allocate_memory_v2(engine,context,input_data):
    bindings = []
    i = 0
    #device = cuda.Device(0)  # 选择设备0
    
    for binding in engine:
        i = i + 1
        #binding_idx = engine.get_binding_index(binding) # 遍历获取对应的索引
        #print(i,":",binding)
        size = trt.volume(context.get_tensor_shape(binding))
        # context.get_binding_shape(binding_idx): 获取对应索引的Shape，例如input的Shape为(1, 3, H, W)
        # trt.volume(shape): 根据shape计算分配内存 
        
        dtype = trt.nptype(engine.get_tensor_dtype(binding))
        # engine.get_binding_dtype(binding): 获取对应index或name的类型
        # trt.nptype(): 映射到numpy类型
        #print('this tensor mode is:',engine.get_tensor_mode(binding))
        if (engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT): # 当前index为网络的输入input
            input_buffer = np.ascontiguousarray(input_data) # 将内存不连续存储的数组转换为内存连续存储的数组，运行速度更快
            input_memory = cuda.mem_alloc(input_data.nbytes) # cuda.mem_alloc()申请内存
            bindings.append(int(input_memory))
            context.set_tensor_address(engine.get_tensor_name(0), int(input_memory))
        else:
            output_buffer = cuda.pagelocked_empty(size, dtype)
            output_memory = cuda.mem_alloc(output_buffer.nbytes)
            bindings.append(int(output_memory))
            context.set_tensor_address(engine.get_tensor_name(1), int(output_memory))
    return input_buffer, input_memory, output_buffer, output_memory, bindings


class TensorRTModel_v2:
    '''onnx转换成的TensorRT模型，使用TensorRT的API进行推理，'''
    def __init__(self,onnx_model_file):
        self.onnx_model_file = onnx_model_file
        engine_build = build_engine(onnx_model_file)
        engine = engine_build.get_engine()
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(TRT_LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(engine)
        self.context = self.engine.create_execution_context()
        self.context.set_input_shape("inputs", (1,128,2))
        self.cfx = cuda.Device(0).make_context()
        self.stream = cuda.Stream()# 创建一个新流
        
    def predict(self, data):
        self.cfx.push()\
        
        stream  = self.stream
        context = self.context
        engine  = self.engine



        # 这里不需要再次创建执行上下文，直接使用初始化时创建的 self.context
        # context = self.engine.create_execution_context()
        #self.context.set_input_shape("inputs", (1,128,2)
        
        input_buffer, input_memory, output_buffer, output_memory, bindings = Allocate_memory_v2(engine,context,data)
        # 异步内存传输
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        self.context.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        # 等待流中的所有操作完成
        stream.synchronize()
        self.cfx.pop()
        # 返回推理结果
        return output_buffer

class TensoRTModel:
    '''if CNN are used 1,128,2 as 1,2,128'''
    def __init__(self,onnx_model_file):
        self.onnx_model_file = onnx_model_file
        engine_build = build_engine(onnx_model_file)
        self.engine = engine_build.get_engine()
        # 生成context
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(TRT_LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(self.engine)
        self.context = self.engine.create_execution_context()
        # 绑定上下文
        self.context.set_input_shape("inputs", (1,128,2))
        
    def predict(self,data):
            # 在当前线程中创建执行上下文
        with self.engine.create_execution_context() as context:
        # 设置绑定形状
            print(f"Current thread: {threading.current_thread().name}, context id: {id(context)}")
            context.set_binding_shape(self.engine.get_binding_index("inputs"), (1,128,2))
        # 分配内存
            input_buffer, input_memory, output_buffer, output_memory, bindings = Allocate_memory(self.engine, context, data)
        # 创建CUDA流
            stream = cuda.Stream()
        # 异步拷贝数据到设备
            cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        # 执行推理
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # 异步拷贝数据到主机
            cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        # 同步CUDA流
            stream.synchronize()

        return output_buffer

    def predict11(self,data):
        #print("Reading input image from file {}".format(input_file))
        data_nbytes = trt.volume(self.engine.get_binding_shape("input")) * trt.float32.itemsize
        output_buffer_nbytes = trt.volume(self.engine.get_binding_shape("output")) * trt.float32.itemsize
        engine = self.engine
        with engine.create_execution_context() as context:
            # Set input shape based on image dimensions for inference
            context.set_binding_shape(self.engine.get_binding_index("input"), (1,128,2))
            # Allocate host and device buffers
            bindings = []
            for binding in engine:
                binding_idx = engine.get_binding_index(binding)
                size = trt.volume(context.get_binding_shape(binding_idx))
                dtype = trt.nptype(engine.get_binding_dtype(binding))
            if engine.binding_is_input(binding):
                input_buffer = np.ascontiguousarray(data)
                input_memory = cuda.mem_alloc(data_nbytes)
                bindings.append(int(input_memory))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer_nbytes)
                bindings.akppend(int(output_memory))

            stream = cuda.Stream()
            # Transfer input data to the GPU.
            cuda.memcpy_htod_async(input_memory, input_buffer, stream)
            # Run inference
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            # Transfer prediction output from the GPU.
            cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
            # Synchronize the stream
            stream.synchronize()
            # Return the host output.
            return output_buffer



class ONNXModel:
    def __init__(self,onnx_fname) -> None:
        self.sess = onnxruntime.InferenceSession(onnx_fname)
    def predict(self,trunk):
        trunk = trunk.astype(np.float32)
        return self.sess.run(None, {"inputs": trunk})






def get_model(model_fname,model_addroot,name_network="LModCNNResNetRelu",dynamic_input_shp=[128,2],output_shp = 7,
              verbose = False):
    """Load the model from the path"""
    model_path = os.path.join(model_addroot, model_fname)

    model = getattr(neural_nets_keras, "get_{}".format(name_network))(
        dynamic_input_shp, output_shp
    )
    # model = get_LModCNNResNetRelu(input_shp=[128, 2], output_shp=14)
    model.load_weights(model_path)
    assert model is not None, f"Model not found at {model_path}"
    
    if verbose:
        model.summary()
    return model


def preprocess(data, queue, trunk_size=128) -> list:
    """
    data : np.Array shape of (len,2) where len is the length of the signal, 2 is the I and Q channels
    """
    signal_len = data.shape[0]
    # print(signal_len)
    trunk_num = signal_len // trunk_size
    trunk_list = []
    for i in range(trunk_num):
        trunk = data[i * trunk_size:(i + 1) * trunk_size, :]
        trunk = np.expand_dims(trunk, axis=0)
        norm = np.sqrt(np.mean(trunk**2, axis=(1, 2), keepdims=True))
        trunk /= norm
        queue.put(trunk)
        trunk_list.append(trunk)
    return trunk_list, trunk_num

def handle_interrupt():
    print("Exiting...")
    mod_rec.stop()
    exit(0)
class Spectrogram:
    '''
    时频图绘制类
    保留一部分信号的序列，用来绘制时频图，通过STFT得到
    需要一个队列来存储数据，每次会增加一部分数据，再去掉一部分数据，用来绘制稳定长度的序列，不希望造成内存泄漏
    length : 保留信号的长度
    nfft : STFT的窗口大小
    update : 更新信号序列
    spectro_draw : 绘制时频图
    
    '''
    def __init__(self,length=128,nfft=128):
        self.nfft = nfft
        self.length = length
        self.spectrum = np.zeros([length,2])
      
    def set_spec_len(self,length):
        self.length = length
        self.spectrum = np.zeros([length,2])
    def set_nfft(self,nfft):
        self.nfft = nfft
        
    def update(self,data):
        '''
        update the signal sequence
        data : np.array shape of (len,2) where len is the length of the signal, 2 is the I and Q channels
        '''
        data_len = data.shape[0]
        self.spectrum = np.concatenate([self.spectrum,data],axis=0)
        self.spectrum = self.spectrum[-self.length:,:]
        
    def spectro_draw(self,axs):
        '''
        draw the spectrogram
        axs : matplotlib.axes.Axes
        '''
        D = librosa.stft(self.spectrum[:, 0] + self.spectrum[:, 1],n_fft=self.nfft, hop_length=self.nfft//2)
        D_amp = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=self.nfft)
        axs.clear()
        axs.imshow(D_amp, aspect='auto', origin='lower', cmap='turbo',extent=[0, self.length /SAMPLE_RATE, freqs.min(), freqs.max()])            
        
        
        
    def spectro_draw_v2(self,axs):
        
        frequencies, times, Sxx = spectrogram(abs(self.spectrum[:, 0] + 1j*self.spectrum[:, 1]), fs=SAMPLE_RATE, nperseg=self.nfft, noverlap=self.nfft//2)
        # 绘制spectrogram
        Sxx = Sxx.transpose()
        axs.clear()
        # You can tune vmax to make signal more visible
        cmap = plt.colormaps["turbo"]
        cmap = cmap.with_extremes(bad=cmap(0))
        #axs.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud', cmap='jet')
        axs.pcolormesh(frequencies,times, 10 * np.log10(Sxx), shading='gouraud', cmap=cmap)
        axs.set_ylabel('Time [sec]')
        axs.set_xlabel('Frequency [Hz]')
        
        
        

def get_udp_data(queue,ip_address='localhost',destination_port=2023,trunk_size=128):
    '''从UDP协议网络中接收数据'''
    # 创建UDP套接字
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 绑定源端口
    sock.bind((ip_address,destination_port))
    print(f'Lisening on {ip_address}:{destination_port}')
    I_datas = np.zeros(128)
    Q_datas = np.zeros(128)
    # try:
    while True:
        # 接收数据
        data, addr = sock.recvfrom(1024)  # float32占4个字节
        # 将接收到的字节转换为浮点数
        print("Receiving data...")
        numbers = np.frombuffer(data, dtype=np.complex64)
        # 将浮点数添加到列表中
        I_datas = np.concatenate((I_datas,numbers.real))[-128:]
        Q_datas = np.concatenate((Q_datas,numbers.imag))[-128:]
        final_data = np.stack((I_datas,Q_datas),axis=1)
        #print(final_data.shape)
        #self.spec.update(final_data)
        signal_len = final_data.shape[0]
        trunk_num = signal_len // trunk_size
        for i in range(trunk_num):
            trunk = final_data[i * trunk_size:(i + 1) * trunk_size, :]
            trunk = np.expand_dims(trunk, axis=0)
            norm = np.sqrt(np.mean(trunk**2, axis=(1, 2), keepdims=True))
            trunk /= norm
            queue.put(trunk)

    # except Exception as e:
    #     sock.close()
    #     print(e)
    #     raise e
    # finally:
    #     sock.close()     





class Modulation_Recognizer:
    '''
    onnx/tf Modulation Recognizer
    '''
    def __init__(self,model_fname,model_addroot,
                 name_network="LModCNNResNetRelu",
                 dynamic_input_shp=[128,2],output_shp = 7,
                 spec_len=1024,ip_address='localhost',
                 destination_port=5421,n_fft=N_FFT,gui=False,onnx_fname="./onnx128.onnx"):
        self.RUN_FLAG = False
        # TODO :ONNX/TENSORRT model implementation

        if TENSORFLOW_ENABLED:
            print("Loading Tensorflow Model...")
            self.model = get_model(name_network=name_network,model_addroot=model_addroot,
                                model_fname=model_fname,dynamic_input_shp=dynamic_input_shp,
                                output_shp=output_shp,verbose=True)
        elif ONNX_ENABLED:
            print("Loading ONNX Model...")
            self.model = ONNXModel(onnx_fname)
        else:
            print("Loading TensorRT Model...")
            self.model = TensorRTModel_v2(onnx_fname)
        self.queue = Queue()
        #self.data_thread = threading.Thread(target=get_udp_data,args=(self.queue,ip_address,destination_port))
        self.data_thread = threading.Thread(target=self.get_udp_data_flag,args=(128,))
        self.stat = "idle..."
        self.spec_len = spec_len
        self.n_fft = n_fft
        self.ip_address = ip_address
        self.destination_port = destination_port
        self.start_time = time.time()
        self.spec = Spectrogram(self.spec_len,self.n_fft)
        self.figs, self.axs = plt.subplots(3)
        self.time = []
        self.average_time = []
        plt.ion()
        if not gui:
            plt.show()
        if TEST:
            print("USING RANDOM signlas for testing!")
            self.subprocess = subprocess.Popen(['python', './udp_tx.py'])#启动硬件接口

    def predict(self,trunk,threshold=0.5): 
        
        start_time = time.time()
        # TODO: TensorRT / Onnx Runtime based Implementation 
        prediction = self.model.predict(trunk)
        prediction = np.array(prediction).reshape(1,4)
        predictions = np.argmax(prediction) #取最大值之后就只有一个索引了
        print(prediction)
        print(predictions)
        class_labels = CLASS_LABEL
        if prediction[0][predictions] < threshold:
            predicted_class = 'Unknown'
        else:
            predicted_class = class_labels[int(predictions)]
        print('此信号最终识别结果为：'+predicted_class)
        proportions = prediction[0]
        print(f'具体各种类信号比例如下,threshold = {threshold}:')
        formatted_proportions = [f"{proportion * 100:.2f}%" for proportion in proportions]

        for i, proportion in enumerate(formatted_proportions):
            print(f"Class {class_labels[i]}: {proportion}")
        end_time = time.time()
        self.time.append(end_time - start_time)
        self.average_time.append(sum(self.time) / len(self.time))
        return predicted_class 
    
    
    def get_udp_data_flag(self,trunk_size=128):
        '''从UDP协议网络中接收数据'''
        # 创建UDP套接字
        self.stat = "initalizing..."
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 绑定源端口
        ip_address = self.ip_address
        destination_port = self.destination_port
        
        sock.bind((ip_address,destination_port))
        print(f'Lisening on {ip_address}:{destination_port}')
        I_datas = np.zeros(128)
        Q_datas = np.zeros(128)
        # try:
        print('self.RUN_FLAG:',self.RUN_FLAG)
        while True:
            if self.RUN_FLAG:
                # 接收数据
                self.stat = "Receiving data..."
                
                data, addr = sock.recvfrom(1024)  # float32占4个字节
                #print("Receiving data...")
                # 将接收到的字节转换为浮点数
                numbers = np.frombuffer(data, dtype=np.complex64)
                # 将浮点数添加到列表中
                I_datas = np.concatenate((I_datas,numbers.real))[-128:]
                Q_datas = np.concatenate((Q_datas,numbers.imag))[-128:]
                final_data = np.stack((I_datas,Q_datas),axis=1)
                #print(final_data.shape)
                #self.spec.update(final_data)
                signal_len = final_data.shape[0]
                trunk_num = signal_len // trunk_size
                for i in range(trunk_num):
                    trunk = final_data[i * trunk_size:(i + 1) * trunk_size, :]
                    trunk = np.expand_dims(trunk, axis=0)
                    norm = np.sqrt(np.mean(trunk**2, axis=(1, 2), keepdims=True))
                    trunk /= norm
                    self.queue.put(trunk)
            else:
                self.stat = "idle..."
                time.sleep(0.1)

    
    
                      
    def stop(self):
            self.RUN_FLAG = False
            plt.close('all')
            self.subprocess.terminate()  
            print('subprocess terminated')       
            exit(0)
    def plot_draw(self,axs,trunk,prediction,type = "constellation"):
        '''
        绘制时域波形，频谱图，IQ图
        
        
        '''
        axs[0].clear()
        axs[0].plot(trunk[0, :, 0])
        axs[0].plot(trunk[0, :, 1])
        axs[0].set_title(f'{prediction}')
        
        
        self.spec.spectro_draw_v2(axs[1])
        
        
        axs[2].clear()
        if type == "constellation":
            I = np.array(trunk[0, :, 0]).flatten()
            Q = np.array(trunk[0, :, 1]).flatten()
            axs[2].hist2d(I, Q, bins=150, range=[[-1.5, 1.5], [-1.5, 1.5]], cmap=plt.cm.binary)
            axs[2].set_aspect('equal', 'box')
        elif type == "fft":
            I = np.array(trunk[0, :, 0]).flatten()
            Q = np.array(trunk[0, :, 1]).flatten()
            fft = np.fft.fft(trunk[0, :, 0] + 1j * trunk[0, :, 1])
            fft_freq = np.fft.fftfreq(len(fft), 1 / SAMPLE_RATE)
            axs[2].plot(fft_freq, np.abs(fft))
            
        plt.draw()
        plt.pause(0.1)
        
        
        
    def plot_predict(self):
        # try:
            while self.RUN_FLAG:
                if not self.queue.empty():
                    trunk = self.queue.get()
                    print(trunk.shape)

                    self.spec.update(trunk[0,:,:])
                    # prediction = self.predict(trunk)
                    if CNN:
                        #1,128,2 
                        p_trunk = trunk.transpose(0,2,1)
                        prediction = self.predict(p_trunk
                                                    )
                    else:
                        prediction = self.predict(trunk)
                    self.plot_draw(self.axs,trunk,prediction,type="fft")
                    
                    
        # except Exception as e:
        #     plt.close()
        #     self.stop()
        #     print(e)
        #     raise e
            

    def start(self):
        self.RUN_FLAG = True
        self.data_thread.start()
        self.plot_predict()

        
     
if __name__ == "__main__":
    


    fname = os.path.join(ROOT,MODEL_NAME)
    mod_rec = Modulation_Recognizer(model_fname=MODEL_NAME,
                                    model_addroot=MODEL_ADDROOT,
                                    name_network="LModCNNResNetRelu",
                                    onnx_fname=fname,
                                    destination_port=DESTINATION_PORT,
                                    ip_address=IP_ADDRESS,
                                    gui=False )
    keyboard.add_hotkey('x', handle_interrupt)
    mod_rec.start()

    #mod_rec.stop()
    
    # signal,classid = get_original_data()
    # my_signal, my_class = extract_random_signal(signal,classid)
    # np.savetxt('one_signal.txt',my_signal)
