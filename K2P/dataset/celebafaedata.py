import os
import cv2
import torch
import numpy as np
from K2P.fileio.io import load, dump
from  torch.utils.data import Dataset
from torchvision.transforms import transforms
from K2P.faceanalysis.parsing.dml_csr.utils import transforms as tf
# from torchvision.datasets.celeba import CelebA
# from torchvision.datasets.folder import DatasetFolder
# from K2P.module.Translator import translator
from K2P.module.Imitator import imatator
import copy
import random

def parserface_embeding(data_dir, method):
    info = method(data_dir)
    return info["embedings"]

def get_all_embeding(data_dir, normal=True):
    all_data = []
    minvs, maxvs = None,  None
    for jp in os.listdir(data_dir):
        jpf = os.path.join(data_dir, jp)
        _, gt_values, minvs, maxvs = get_parameters_info(jpf) 

        all_data.append(gt_values.reshape(1, 28))
    all_data = np.concatenate(all_data, axis=0)
    if normal:
        all_data_normal = normalize_vec(minvs, maxvs, all_data)
    else:
        all_data_normal = all_data
    return all_data_normal

def save_get_embeding(embedings, save_dir, save_name="celeba_embeding.npy"):
    file_name = os.path.join(save_dir, save_name)
    if os.path.exists(file_name):
        return np.load(file_name)
    else:
        os.makedirs(save_dir)
        np.save(file=file_name, arr=embedings)
        return True


def get_facial_priors(data, keep_rate=0.9):
    """
    获取给定数据集的 projection matrix，保持信息率为 keep_rate 和均值
    """
    # 1. 计算数据集的均值
    mean = np.mean(data, axis=0)

    # 2. 将数据集中心化
    data_centered = data - mean  # 这里，原始数据已经被做了0，1归一化了。而且是按照各自的取值范围做了min-max归一化

    # 3. 计算数据集的协方差矩阵
    cov = np.cov(data_centered, rowvar=False)
    # cov = np.dot(np.transpose(data_centered), data_centered)

    # 4. 对协方差矩阵进行特征值分解
    # eig_values, eig_vectors = np.linalg.eigh(cov)  # 复数
    eig_values, eig_vectors = np.linalg.eig(cov)
    
    # 5. 对特征值进行排序，选择信息量占比前 keep_rate 的特征向量
    sorted_indices = np.argsort(eig_values)[::-1]
    sorted_eig_values = eig_values[sorted_indices]
    sorted_eig_vectors = eig_vectors[:, sorted_indices]
    variance_ratio = np.cumsum(sorted_eig_values) / np.sum(sorted_eig_values)
    num_keep = np.argmax(variance_ratio >= keep_rate)
    projection_matrix = sorted_eig_vectors[:, :num_keep]
    # 6. 对数据进行白化处理
    data_whitened = np.dot(data_centered, projection_matrix)
    # data_whitened = np.dot(projection_matrix.T, data_centered.T)  # 论文中公式处理方式数据是列表示？或者论文错了

    return data_whitened, projection_matrix, mean

def inverse_whiten_wrong(whitened_data, projection_matrix, mean):
    # 论文中的方式计算出来是错误的。 具体细节待查
    P_ = np.dot(np.linalg.inv(np.dot(projection_matrix, projection_matrix.T)), projection_matrix)
    original_data = np.dot(P_, whitened_data).T + mean
    return original_data.T

def inverse_whiten(whitened_data, projection_matrix, mean):
    original_data = np.dot(whitened_data, projection_matrix.T) + mean
    print(original_data.max(), original_data.min())
    return original_data

def check_pca(data, reduction_dim=26):
    # 此函数说明inverse_whiten白化处理实现正确
    from sklearn.decomposition import PCA
    pca = PCA(n_components=reduction_dim, whiten=True).fit(data)
    x_reduction = pca.transform(data)
    recdata = pca.inverse_transform(x_reduction)
    print(x_reduction.shape, recdata.shape, recdata.max(), recdata.min())
    return x_reduction, recdata

def whitening_transformation_pca(P, x, m):
    """
    用给定数据的P, m，映射单个数据的，达到降维的目的
    一条x或者x为矩阵，一个batch均可。
    """
    x_bar = P.T @ (x - m)
    return x_bar

def whitening_transformation_pca_inverse(P, x_bar, m):
    """
    将被降维的参数x_bar，映射回去
    """
    x = P@x_bar + m
    # P.T @ P = I
    return x

def normalize_vec(minv, maxv, x):
    """
    x序列值和 对应的最小最大范围， 向量化生成0-1
    """
    return (x - minv ) / (maxv - minv)

def innormalize_vec(minv, maxv, pred_v):
    """
    np.array, minv, maxv 按照给定顺序给出
    minv = np.array([-1, -1, -1, ])
    """
    return pred_v * (maxv - minv) + minv

def get_parameters_info(jp):
    j = load(jp)
    minvs = []
    maxvs = []
    name_map = dict()
    gt_values = []
    for parameter in j["Datas"]:
        minv = parameter["MinValue"]
        maxv = parameter["MaxValue"]
        name_map[parameter["SkeletonName"]] = parameter["NameZH"]
        minvs.append(minv)
        maxvs.append(maxv)
        gt_values.append(parameter["CurValue"])
    dp = np.float64
    minvs, maxvs = np.array(minvs, dtype=dp), np.array(maxvs, dtype=dp)
    gt_values = np.array(gt_values, dtype=dp)
    return name_map, gt_values, minvs, maxvs

def get_parameters(jp, minvs, maxvs, normal=True):
    """
    最好是提前处理，写为文件格式，直接读取
    """
    info = load(jp)
    gt_values = []
    for parameter in info["Datas"]:
        gt_values.append(parameter["CurValue"])
    gt_values = np.array(gt_values, dtype=np.float64)
    if normal:
        gt_values = normalize_vec(minvs, maxvs, gt_values)
    return gt_values

def parser_jp(jp):
    name_map, gt_values, minvs, maxvs = get_parameters_info(jp)
    gt_normal = normalize_vec(minvs, maxvs, gt_values)
    gt_normal = torch.from_numpy(gt_normal).double().unsqueeze(0).cuda()
    return gt_normal, name_map

def write_default_json(jp_template, pred_v, save_file="pred_parameters.json"):
    jp = load(jp_template)
    jp_temp  = copy.deepcopy(jp)
    for i,p in enumerate(jp_temp["Datas"]):
        p["CurValue"] = pred_v[i]
        print(p["SkeletonName"])
    dump(jp_temp, save_file)
    return

def write_result_json(pred_v, save_file="./pred_parameters.json"):
    import json
    json_struct = dict() # write your own json struct TODO see here 

    for i,p in enumerate(json_struct["Datas"]):
        p["CurValue"] = pred_v[i].item()
    # return dump(json_struct, save_file)  # true is successed
    with open(save_file, 'w') as f:
        json.dump(json_struct, f, indent=4, ensure_ascii=False)

    
class FaceParameters(Dataset):
    def __init__(self, cfg, mode="train"):
        """
        """
        if mode == "train":
            self.path = cfg.faceparameter_train_dir
        elif mode == "test":
            self.path = cfg.faceparameter_test_dir
        else:
            raise ("not such mode for dataset")
        self.params  = os.listdir(os.path.join(self.path, "Data"))
        self.names = os.listdir(os.path.join(self.path, "Texture"))
        random_json = random.sample(self.params, 1)[0]
        one_json = os.path.join(self.path, "Data", random_json)
        self.name_map, self.ge_values, self.minvs, self.maxvs = get_parameters_info(one_json)

        self.cfg = cfg
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.transformer = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float32), normalize])
        if self.cfg.use_whitening:
            embedings = parserface_embeding(cfg.celebafrontal_dir, method=None)  # TODO
            P, mean = get_facial_priors(embedings, keep_rate=cfg.keep_rate)
            # self.params = P.T @ (np.array(self.params) - mean)
            self.params = whitening_transformation_pca(P, self.params, mean)  # TODO replace self.params matrix
        
        self.aspect_ratio = 1.0
        
    def _xywh2cs(self, x: float, y: float, w: float, h: float) -> tuple:
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)

        return center, scale
    
    def _box2cs(self, box: list) -> tuple:
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)
    
    def __getitem__(self, idx):
        jp = os.path.join(self.path, "Data", self.params[idx])
        imgp = os.path.join(self.path, "Texture", self.names[idx])
        image = cv2.imread(imgp, cv2.IMREAD_COLOR)
        # 图片做了resize, 参数没有.. 
        h, w, _ = image.shape
        # Get center and scale
        center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0
        
        trans = tf.get_affine_transform(center, s, r, (self.cfg.dst_size, self.cfg.dst_size))
        image = cv2.warpAffine(
            image,
            trans,
            (self.cfg.dst_size, self.cfg.dst_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        
        # image = cv2.resize(image, (self.cfg.dst_size, self.cfg.dst_size), interpolation=cv2.INTER_LINEAR)
        param = get_parameters(jp, self.minvs, self.maxvs, normal=True)
        param = torch.from_numpy(param).float()  # change 

        image = self.transformer(image)
        # image = image * 2 - 1  # to [-1, 1] for the last layer is tanh
        return image, param

    def __len__(self):
        return len(self.names)

class CelebaFace(Dataset):
    def __init__(self, cfg, frontal=False):
        """
        Celeba data does not need face parameter, use a loopback_loss
        but may be giving some parameter from the K2P_simple(first method)?
        """
        super().__init__()
        if frontal:
            self.celeba_dir = cfg.celebafrontal_dir
        else:
            self.celeba_dir = cfg.celeba_dir
        self.size = (cfg.dst_size, cfg.dst_size)
        self.img_lists = os.listdir(self.celeba_dir)
        self.transformer = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=self.size)])

    def __getitem__(self, index):
        img_p = os.path.join(self.celeba_dir, self.img_lists[index])
        img = cv2.imread(img_p)
        img = self.transformer(img)
        # label = torch.from_numpy(np.zeros(28)).float()
        label = 0
        return img, label


    def __len__(self):
        return len(self.img_lists)