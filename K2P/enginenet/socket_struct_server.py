import socket
import struct
import numpy as np
from PIL import Image
import cv2

# def pack_cvimg():
#     # Load the image
#     img = cv2.imread('K2P.jpg')
#     # Convert the image to bytes using cv2.imencode
#     img_bytes = cv2.imencode('.png', img)[1].tobytes()
#     # Pack the image bytes using struct.pack
#     packed_img = struct.pack('Q{}s'.format(len(img_bytes)), len(img_bytes), img_bytes)
#     print(len(img_bytes), 'sss')

#     return packed_img

# def send_array(array, sock):
#     # 将 numpy array 转换为字节数组
#     data = array.tobytes()
#     # 获取字节数组长度
#     size = len(data)
#     # 使用 struct 库将长度打包为 4 字节的字节数组
#     data_bytes = struct.pack('!i{}s'.format(size), size, data)
#     # 将长度字节数组和实际数据字节数组一起发送
#     sock.sendall(data_bytes)

def pack_img_array(img, array):
    img_bytes = cv2.imencode('.png', img)[1].tobytes()
    array_bytes = array.tobytes()
    # print(array_bytes, len(array_bytes), len(img_bytes))
    len_img_bytes, len_array_bytes = len(img_bytes), len(array_bytes)
    packed_img_array = struct.pack("!Q{}si{}s".format(len_img_bytes, len_array_bytes), 
                                   len_img_bytes, img_bytes,
                                   len_array_bytes, array_bytes)
    return packed_img_array

def unpacke_img_array(packed_img_array):
    """
    一次只支持一张图和对应的参数
    """
    int_size, uul_size = struct.calcsize("i"), struct.calcsize("Q")
    img_bytes_len = struct.unpack("Q", packed_img_array[:uul_size])[0]
    img_bytes = packed_img_array[uul_size:uul_size+img_bytes_len][0]
    # array_bytes_len = struct.unpack("i", packed_img_array[uul_size+img_bytes_len:
    #                                                       uul_size+img_bytes_len+int_size])[0]
    # array_bytes = packed_img_array[-array_bytes_len:]
    array_bytes = packed_img_array[uul_size+img_bytes_len+int_size:]
    return img_bytes, array_bytes


# def send_img(sock, packed_img):
#     sock.sendall(packed_img)

def send_img_array(sock, packed_img_array):
    sock.sendall(packed_img_array)

# # 服务端代码 多线程
def handle_client(conn, addr, data):
    try:
        with conn:
            # # 发送numpy array
            # send_array(np.array([1.2, 3, 9.9], dtype=np.float32), conn)
            # # 发送图像
            # packed_img = pack_cvimg()
            # send_img(conn, packed_img)
            send_img_array(conn, data)
            print(f"Connection from {addr} has been handled.")
    except Exception as e:
        print(e)

def main(data):
    import threading
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 12345))
        s.listen()
        print("Server started, waiting for connection...")
        while True:
            conn, addr = s.accept()
            print(f"Accepted connection from {addr}")
            # 开启一个新线程处理客户端请求
            t = threading.Thread(target=handle_client, args=(conn, addr, data))
            t.start()


if __name__ == "__main__":
    img = cv2.imread('K2P.jpg')
    array = np.array([1.2, 3, 9.9], dtype=np.float32)
    data = pack_img_array(img, array)
    main(data)
