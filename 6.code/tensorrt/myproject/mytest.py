import pycuda
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
from h5py import File
import time
 
# 前处理
def preprocess(data):
    data = np.asarray(data)
    return data
    
# 后处理
def postprocess(data):
    data = np.reshape(data, (B, H, W))
    return data
    
# 创建build_engine类
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
        print('EXPLICIT_BATCH:',EXPLICIT_BATCH)
        # 创建builder，用于创建network
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(EXPLICIT_BATCH) # 创建network（初始为空）
        # 创建config
        config = builder.create_builder_config()
        profile = builder.create_optimization_profile() # 创建profile
        profile.set_shape("inputs", (32,128,2), (32,128,2), (32,128,2))  # 设置动态输入,分别对应:最小尺寸、最佳尺寸、最大尺寸
        config.add_optimization_profile(profile)
        #允许使用精度
        config.set_flag(trt.BuilderFlag.FP16)
        print('FP16:',config.get_flag(trt.BuilderFlag.FP16))
        #config.set_flag(trt.BuilderFlag.INT8)
        print('INT8:',config.get_flag(trt.BuilderFlag.INT8))
        #config.set_flag(trt.BuilderFlag.BF16)
        print('BF16:',config.get_flag(trt.BuilderFlag.BF16))
        #config.set_flag(trt.BuilderFlag.FP8)
        print('FP8:',config.get_flag(trt.BuilderFlag.FP8))
        config.clear_flag(trt.BuilderFlag.TF32)
        print('TF32:',config.get_flag(trt.BuilderFlag.TF32))


        #config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
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
def Allocate_memory(engine, context):
    bindings = []
    i = 0
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
def read_augmod(path):
    data = dict()
    with File(path, "r") as f:
        data["classes"] = [c.decode() for c in f["classes"]]
        data["signals"] = np.array(f["signals"])
        data["modulations"] = np.array(f["modulations"])
        data["snr"] = np.array(f["snr"])
        data["frequency_offsets"] = np.array(f["frequency_offsets"])
    return data["signals"],data["modulations"] 
        
if __name__ == "__main__":
    #FP16_ENABLE = ture
    # 设置输入参数，生成输入数据     
    Batch_size = 32
    Channel = 3
    Height = 128
    Width = 2
    start_time = time.time()
    #input_data = np.random.rand(Batch_size, Height, Width)
    #数据输入
    signals,classes =read_augmod("./augmod.hdf5")
    norm = np.sqrt(np.mean(signals**2, axis=(1, 2), keepdims=True))
    signals /= norm
    num_samples = 10000
    accurary = 0 
    random_indices = np.random.choice(len(signals), num_samples, replace=False)
    random_signals = signals[random_indices]
    random_class = classes[random_indices]
    random_signals = np.array(random_signals).transpose(0,2,1)
    start_time_1 = time.time() - start_time
    start_time = time.time()
    print("input data successful!  time:",start_time_1)
    #print(input_data.shape)
# 生成engine
    onnx_model_file = "./onnx128.onnx"
    engine_build = build_engine(onnx_model_file)
    engine = engine_build.get_engine()
    start_time_1 = time.time() - start_time
    start_time = time.time()
    print("build engine successful!  time:",start_time_1)

    # 生成context
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine)
    context = engine.create_execution_context()
    start_time_1 = time.time() - start_time
    start_time = time.time()
    print("build context successful!  time:",start_time_1)

    # 绑定上下文
    context.set_input_shape("inputs", (32,128,2))
    print("binding successful!")

    start_time = time.time()

    num_full_batches = len(random_signals) // Batch_size
    print(random_signals.shape)
   # 为方便处理，这里我们只使用完整批次的数据
    random_signals = random_signals[:num_full_batches * Batch_size,:,:]
    random_class = random_class[:num_full_batches * Batch_size]
    print(random_signals.shape)

#xun huan
    for i in range(num_full_batches):
        start_index = i * Batch_size
        end_index = start_index + Batch_size

    # 批量提取信号和类别
        batch_signals = random_signals[start_index:end_index]
        #print(batch_signals.shape)
        batch_class = random_class[start_index:end_index]

    # 前处理
        input_data = preprocess(batch_signals)
    # 拷贝数据到GPU （host -> device)
 # 分配内存缓冲区
        input_buffer, input_memory, output_buffer, output_memory, bindings = Allocate_memory(engine, context)
    # 创建Cuda流
        stream = cuda.Stream()
        cuda.memcpy_htod_async(input_memory, input_buffer, stream) # 异步拷贝数据
    # 推理
        #print(type(context),type(stream.handle))
        context.execute_async_v3(stream.handle)
    
    # 将GPU得到的推理结果 拷贝到主机（device -> host）
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
    
    # 同步Cuda流
        stream.synchronize()
        output_buffer = output_buffer.reshape(32,7)
        i = np.argmax(output_buffer, axis=1, keepdims=True)
        #print(output_buffer.shape)
        #print(f"pred:{i},true:{batch_class}")
        if i[0] == batch_class[0]:
            accurary += 1
        else:
            pass

    
    # 后处理
    #output_data = postprocess(output_buffer)
    accurary /= num_samples
    accurary *= Batch_size
    print("acc:",accurary,"need time:",time.time()-start_time)
