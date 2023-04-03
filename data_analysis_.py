import cv2
import os

from K2P.dataset.celebafaedata import get_all_embeding, get_facial_priors, inverse_whiten, check_pca


"""
从易水寒论文给出的例子，可以看出，角色的表情比较丰富，且角色本身存在各种配饰，胡须等其他物件。
其论文实现的效果，比较神似，有眼睛和嘴巴的控制。
连续参数控制的具体构成(表示shift, orientation and scale)：
    大板块上分五部分：眉毛， 眼睛， 鼻子， 嘴巴， 脸
    其各自的细节有     24,   51,   30,    42,   61 选项,共 208， 实际手机上貌似是174
        眉毛：s眉头、眉体、眉尾
        眼睛：整个，上眼睑外侧，上眼睑内侧，下眼睑，内侧角，外角
        鼻子： 整体、桥、翼、尖端、底部
        嘴巴： 整个，中上唇，外上唇，中下唇，外下唇，嘴唇，角落
        脸： 前额、眉间、颧骨、笑肌、脸颊、下颌、下颌， 下颌角、外颌
    连续参数的各个部分维度应该不同。
离散参数(158)：
    女性： 22种头发， 36中眉毛样式， 19种口红样式， 25种口红颜色
    男性： 23种头发， 26种眉毛样式， 7种胡须样式
    
你的数据：
    略 
    
"""

def img2movie(img_dir, save_path, format="mp4"):
    # 获取图片路径列表
    img_list = sorted(os.listdir(img_dir))

    # 获取第一张图片，用于设置视频分辨率
    img_path = os.path.join(img_dir, img_list[0])
    img = cv2.imread(img_path)
    height, width, channels = img.shape

    # 创建视频编写器对象
    video_name = f"{save_path}.{format}"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30
    video_writer = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    # 逐帧添加图片到视频中
    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        video_writer.write(img)

    # 释放视频编写器对象
    video_writer.release()
    
    
def visual_embedding(embeddings, method="tsne", n_components=2):
    import matplotlib.pyplot as plt
    from sklearn import manifold
    from sklearn.decomposition import PCA

    from mpl_toolkits.mplot3d import Axes3D

    if method == "tsne":
        # 使用t-SNE算法进行降维
        tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=501)
        embeddings_xd = tsne.fit_transform(embeddings)
    elif method == "pca":
        # 使用PCA算法进行降维
        pca = PCA(n_components=n_components)
        embeddings_xd = pca.fit_transform(embeddings)
    else:
        raise ValueError("Unsupported dimension reduction method: " + method)
    
    # 可视化降维后的结果d
    if n_components == 2:
        plt.scatter(embeddings_xd[:, 0], embeddings_xd[:, 1], c=range(len(embeddings_xd)), cmap='viridis')
        plt.colorbar()
        # plt.show()
        plt.savefig("visual_embedding_" + method + str(n_components) + ".png")
    elif n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embeddings_xd[:, 0], embeddings_xd[:, 1], embeddings_xd[:, 2], c=range(len(embeddings_xd)), cmap='viridis')
        # plt.show()
        plt.savefig("visual_embedding_" + method + str(n_components) + ".png")

    else:
        raise ValueError("Unsupported dimension: " + str(n_components))


if __name__ == "__main__":
    
    # img2movie("/data/data_320/test/Texture", "./data_320_test")
    data_dir = "/data/data_320_test_male/Data"
    # data_dir = "/data/data_320/test/Data"
    all_data_normal = get_all_embeding(data_dir, normal=False)
    print(all_data_normal.max(), all_data_normal.min())
    # print(all_data_normal.shape)
    # visual_embedding(all_data_normal, method="pca", n_components=3)
    print()
    data_whitened, projection_matrix, mean = get_facial_priors(all_data_normal, keep_rate=0.98)
    print(data_whitened.shape, mean, projection_matrix.shape)
    o_d = inverse_whiten(data_whitened, projection_matrix, mean)
    x_red, x_rec = check_pca(all_data_normal, reduction_dim=26)
    
