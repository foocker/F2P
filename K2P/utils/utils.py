import torch
import cv2
import numpy as np
from K2P.dataset.celebafaedata import innormalize_vec, get_parameters_info

def points2bbox(points, points_scale=None):
    if points_scale:
        assert points_scale[0]==points_scale[1]
        points = points.clone()
        points[:,:,:2] = (points[:,:,:2]*0.5 + 0.5)*points_scale[0]
    min_coords, _ = torch.min(points, dim=1)
    xmin, ymin = min_coords[:, 0], min_coords[:, 1]
    max_coords, _ = torch.max(points, dim=1)
    xmax, ymax = max_coords[:, 0], max_coords[:, 1]
    center = torch.stack([xmax + xmin, ymax + ymin], dim=-1) * 0.5

    width = (xmax - xmin)
    height = (ymax - ymin)
    # Convert the bounding box to a square box
    size = torch.max(width, height).unsqueeze(-1)
    return center, size

def crop_face(kp, img, expand=8):
    pass


def img2tensor(image):
    """
    PIL :transform = transforms.Compose([
    transforms.PILToTensor()
    ])
  
    """
    from torchvision import transforms
    # # Convert BGR image to RGB image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([transforms.ToTensor()])
    # Convert the image to Torch tensor
    tensor = transform(image)
    return tensor

    
def tensor2image(tensor):
    import numpy as np
    """
    tensor to cv2 numpy array
    :param tensor: [batch, c, w, h]
    :return: [batch, h, w, c]
    """
    batch = tensor.size(0)
    images = []
    for i in range(batch):
        img = tensor[i].cpu().detach().numpy()
        img = np.swapaxes(img, 0, 2)  # [h, w, c]
        img = np.swapaxes(img, 0, 1)  # [w, h, c]
        images.append(img * 255)
    return images

def save_img(img, save_p):
    cv2.imwrite(save_p, img)

import dlib

def face_frontal_intensity(image, face_detector, shape_predictor):
    # 使用dlib的人脸检测器检测人脸
    pass

def sigmoid(ary):
    return 1 / (1 + np.exp(-ary))

def evaluate_parameters(gt_p, pred_p, sign_rate=0.5, dist_threshold=0.1*28, **kwargs):
    """
    预测的人脸参数与客户端给出的人脸参数的欧式距离 + 符号距离。
    计算距离在给定阈值之下的占比，以及符号距离占比sign_rate
    最后可考虑 两者的比例

    """
    lower_dis_threshold_rate = 1 - np.greater(np.power(gt_p-pred_p, 2).sum(axis=1), dist_threshold).mean()
    dim = gt_p.shape[1]
    minvs, maxvs = kwargs.get("minvs", None), kwargs.get("maxvs", None)s
    if minvs is not  None and maxvs is not None:
        gt_p_original = innormalize_vec(minvs, maxvs, gt_p)
        pred_p_original = innormalize_vec(minvs, maxvs, pred_p)
        gt_p_sign  = np.sign(gt_p_original)
        pred_p_sign = np.sign(pred_p_original)
        keep_sign = np.equal(gt_p_sign,pred_p_sign).sum(axis=1).mean() / dim
        combine = lower_dis_threshold_rate * (1-sign_rate ) + keep_sign * sign_rate
    else:
        keep_sign = -1
        combine = -1
    
    return lower_dis_threshold_rate, keep_sign, combine

def mv_img():
    import shutil
    pass


from K2P.faceanalysis.insightface.iface import FaceInformation
from K2P.fileio.path import mkdir_or_exist
Ff = FaceInformation(detect_only=True)
import glob, os
img_dir = "D:/code/GameCharacterAuto-Creation/img_align_celeba"
img_list = sorted(glob.glob(img_dir + "/*.jpg"))
img_align_celeba_crop = os.path.join(os.path.dirname(img_dir),"img_align_celeba_crop")
mkdir_or_exist(img_align_celeba_crop)

def hand_func(img_sublist):
    for imgf in img_sublist:
        img = cv2.imread(imgf)
        try:
            _, aimg, _, _, _ = Ff.info(img)
            dst_p = os.path.join(img_align_celeba_crop, os.path.basename(imgf))
            Ff.save_img(aimg, dst_p)
        except:
            print("Wrong")
            continue

def main_process(process_num=8):
    from multiprocessing import Process
    one_split_num = len(img_list) // process_num

    img_lists = [img_list[i*one_split_num:(i+1)*one_split_num] for i in range(process_num+1)]
    process = [Process(target=hand_func, args=(img_lists[i],)) for i in range(process_num)]
    [p.start() for p in process]  # 开启进程
    [p.join() for p in process]   # 等待进程依次结束


# from abc import ABC, abstractmethod
# class BaseMutilProcess(ABC):
#     def __init__(self, process_num=8):
#         self.process_num = process_num
#         one_split_num = 100
#         self.img_lists = [img_list[i*one_split_num:(i+1)*one_split_num] for i in range(process_num+1)]
    
#     @abstractmethod
#     def hand_func(self):
#         pass

#     def run(self):
#         from multiprocessing import Process
#         process = [Process(target=hand_func, args=(self.img_lists[i],)) for i in range(self.process_num)]


if __name__ == "__main__":
    main_process(process_num=16)  # 主进程写在if 内部
    # print()
    # hand_func(img_list[:10])
    