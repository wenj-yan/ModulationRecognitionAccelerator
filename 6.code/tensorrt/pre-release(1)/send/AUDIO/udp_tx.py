import socket
import numpy as np
import time

def udp_send(source_port, destination_port):
    # 创建UDP套接字
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 绑定源端口
    sock.bind(('localhost', source_port))

    # 从文件中读取数据
    raw_data = np.fromfile('./AUDIO/gnuradio_2024429_2000000_2ASK_432.96M.raw', dtype=np.float32)
#E:\embedded system\6.code\pre-release(1)\send\AUDIO\AUDIO\gnuradio_2024512_1619_500000_89.9M_FM.raw
#E:\embedded system\6.code\pre-release(1)\send\AUDIO\AUDIO\gnuradio_2024512_1223_500000_15.71M_AM(2).raw
#E:\embedded system\6.code\pre-release(1)\send\AUDIO\AUDIO\gnuradio_2024510_1610_500000_13.825M_DRM(2).raw
#E:\embedded system\6.code\pre-release(1)\send\AUDIO\AUDIO\gnuradio_2024429_2000000_2ASK_432.96M.raw
    start_index = 0
    while start_index + 256 <= len(raw_data):
        # 从raw_data中取出256个点，奇数点作为I路信号，偶数点作为Q路信号
        idata = raw_data[start_index:start_index + 256:2]
        qdata = raw_data[start_index + 1:start_index + 257:2]

        # 将I路和Q路信号合并成复数形式
        iq_data = idata + 1j * qdata

        # 将numpy数组转换为字节
        data_bytes = iq_data.astype(np.complex64).tobytes()

        # 发送数据
        sock.sendto(data_bytes, ('localhost', destination_port))


        # 更新起始索引
        start_index += 256

    # 关闭套接字
    sock.close()

if __name__ == "__main__":
    # 设置源端口和目标端口
    source_port = 12534
    destination_port = 2024

    # 发送数据
    udp_send(source_port, destination_port)
