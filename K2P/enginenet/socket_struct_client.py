import socket
import struct
import numpy as np
from PIL import Image
import cv2

# def receive_array(sock):
#     # 从 socket 中接收 4 字节的字节数组，表示数据的长度
#     size_bytes_format = sock.recv(4)
#     # 使用 struct 库将字节数组解包为整数
#     size_array = struct.unpack('!i', size_bytes_format)[0]
#     # 从 socket 中接收数据字节数组
#     array_binary_data = sock.recv(size_array)
#     # 使用 numpy 将字节数组转换为 numpy array
#     array = np.frombuffer(array_binary_data, dtype=np.float32)
#     return array

# def receive_image(sock):
#     # 接收并解包图像的大  
#     # 8 = struct.calcsize("Q")
#     size_bytes_format = sock.recv(8) 
#     size_img = struct.unpack("Q", size_bytes_format)[0]
#     # 接收并解包图像
#     print(size_img)
#     img_bytes = b""
#     n_iters = size_img // 4096 
#     while  n_iters > 0 :
#         img_bytes += sock.recv(4096)
#         n_iters -= 1
#     if n_iters * 4096 < size_img:
#         img_bytes += sock.recv(4096)
#     # img_binary_data = sock.recv(size_img)
#     # 将二进制字符串转换为 PIL 图像
#     image = Image.frombytes(mode="RGB", size=(960, 414), data=img_bytes)
#     return image

# def receive_image(sock):
#     # Unpack the image bytes using struct.unpack
#     header_size = struct.calcsize('Q')
#     size_bytes_format = sock.recv(header_size)
#     size_img = struct.unpack("Q", size_bytes_format)[0]
#     img_byte = sock.recv(size_img)
#     # img_byte = packed_data[header_size:]  # img_byte
#     img = cv2.imdecode(np.frombuffer(img_byte, np.uint8), cv2.IMREAD_COLOR)

#     return img


# def unpack_cvimg(packed_data):
#     # Unpack the image bytes using struct.unpack
#     header_size = struct.calcsize('Q')
#     # header = packed_data[:header_size]
#     # data = packed_data[header_size:]
#     # img_size = struct.unpack('Q', header)[0]
#     # img_bytes = data[:img_size]
#     img_bytes = packed_data[header_size:]  # if only contains image 
#     # Convert the bytes back to an image using cv2.imdecode
#     img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

#     return img

def receive_img_array_bytes(sock):
    img_len_bytes = sock.recv(8)
    size_img = struct.unpack("!Q", img_len_bytes)[0]
    img_bytes = sock.recv(size_img)
    array_len_bytes = sock.recv(4)
    # print(array_len_bytes, 'xx', len(array_len_bytes))
    array_size = struct.unpack("i", array_len_bytes)[0]
    # print(array_size, 'sss')
    array_bytes = sock.recv(array_size)

    return img_bytes, array_bytes

def parser_img_array_bytes(img_bytes, array_bytes):
    print(len(img_bytes), len(array_bytes))
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    array = np.frombuffer(array_bytes, dtype=np.float32)
    return img, array


# 客户端代码  多线程
def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', 12345))
        # 接收并解析 numpy array
        img_bytes, array_bytes = receive_img_array_bytes(s)

        img, array = parser_img_array_bytes(img_bytes, array_bytes)
        print(array)
        cv2.imwrite("gxgx23.png", img)
        print("successed")

main()
