import socket
import numpy as np
import time
#from .inference import DESTINATION_PORT
def udp_send(source_port, destination_port):
    # 创建UDP套接字
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 绑定源端口
    sock.bind(('localhost', source_port))

    while True:
        # 生成随机数据
        # rand_freq = np.random.rand() * 1000
        # sin_real = np.sin(np.linspace(0, 0.5*np.pi, 128))
        # sin_imag = np.sin(np.linspace(0, 0.05*np.pi, 128))
        data_real = np.random.randn(128)
        data_imag = np.random.randn(128)
        data = data_real + 1j * data_imag
        # data = sin_real + 1j * sin_imag
        # 将numpy数组转换为字节
        data_bytes = data.astype(np.complex64).tobytes()

        # 发送数据
        sock.sendto(data_bytes, ('localhost', destination_port))
        #time.sleep(0.1)

    # 关闭套接字
    sock.close()

if __name__ == "__main__":
    # 设置源端口和目标端口
    source_port = 11122
    destination_port = 2024
    # 发送数据
    udp_send(source_port, destination_port)