import torch
import cv2
import numpy as np
from K2P.faceanalysis.faceparsing.model import BiSeNet
from torchvision import transforms
from einops import rearrange

def load_BiSeNet(cfg):
    net = BiSeNet(n_classes=19)
    if cfg.cuda:
        net.to("cuda")
        net.load_state_dict(torch.load(cfg.seg_checkpoint))
    else:
        net.load_state_dict(torch.load(cfg.seg_checkpoint, map_location="cpu"))
    net.eval()
    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),  # [H, W, C]->[C, H, W]
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    return net, to_tensor

def vis_parsing_maps(im, parsing):
    """
    # 显示所有部位
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 0, 85], [255, 0, 170], [0, 255, 0], [85, 255, 0],
                   [170, 255, 0], [0, 255, 85], [0, 255, 170], [0, 0, 255], [85, 0, 255], [170, 0, 255], [0, 85, 255],
                   [0, 170, 255], [255, 255, 0], [255, 255, 85], [255, 255, 170], [255, 0, 255], [255, 85, 255],
                   [255, 170, 255], [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    """
    # # 只显示鼻子 眼睛 眉毛 嘴巴
    # part_colors = [[255, 255, 255], [255, 255, 255], [25, 170, 0], [255, 170, 0], [254, 0, 170], [254, 0, 170],
    #                [255, 255, 255],
    #                [255, 255, 255], [255, 255, 255], [255, 255, 255], [0, 0, 254], [85, 0, 255], [170, 0, 255],
    #                [0, 85, 255],
    #                [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255],
    #                [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]]
    
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 0, 85], [255, 0, 170], [0, 255, 0], [85, 255, 0],
                   [170, 255, 0], [0, 255, 85], [0, 255, 170], [0, 0, 255], [85, 0, 255], [170, 0, 255], [0, 85, 255],
                   [0, 170, 255], [255, 255, 0], [255, 255, 85], [255, 255, 170], [255, 0, 255], [255, 85, 255],
                   [255, 170, 255], [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    
    """
    part_colors = [[255, 255, 255], [脸], [左眉], [右眉], [左眼], [右眼],
                   [255, 255, 255],
                   [左耳], [右耳], [255, 255, 255], [鼻子], [牙齿], [上唇], 
                   [下唇],
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255],
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]]
    """

    im = np.array(im)
    vis_parsing = parsing.copy().astype(np.uint8)
    vis_parsing_anno_color = np.zeros((vis_parsing.shape[0], vis_parsing.shape[1], 3)) + 255
    num_of_class = np.max(vis_parsing)
    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    return vis_parsing_anno_color, im

def faceparsing_ndarray(img, cfg):
    """
    :param img: numpy array, np.uint8
    """
    net, _to_tensor = load_BiSeNet(cfg)
    input_ = _to_tensor(img)
    input_ = torch.unsqueeze(input_, 0)
    if cfg.cuda:
        input_ = input_.to("cuda")
    out = net(input_)
    parsing = out.squeeze(0).cpu().detach().numpy().argmax(0)
    return vis_parsing_maps(img, parsing)

def faceparsing_tensor(input, cfg, w):
    """
    :param input: torch tensor [B, H, W, C] rang: [0-1], not [0-255]
    :param cp: args.parsing_checkpoint, str
    :param w: tuple len=6 [eyebrow,eye,nose,teeth,up lip,lower lip]
    :return  tensor, shape:[H, W]
    """
    net, _ = load_BiSeNet(cfg)
    out = net(input)
    out = out.squeeze()

    return w[0] * out[3] + w[1] * out[4] + w[2] * out[10] + out[11] + out[12] + out[13], out[1]

def infer_face_seg(cfg, file_p):
    device = torch.device('cuda:0')
    with torch.no_grad():
        img = cv2.imread(file_p)
        image = cv2.resize(img, (256, 256), cv2.INTER_LINEAR)
        img_array, im = faceparsing_ndarray(image, cfg)
        cv2.imwrite("seg_out_ndarray.png", img_array)
        cv2.imwrite("seg_out_ndarray_im_original.png", im)
        image = image[np.newaxis, :, :, :]
        image = np.swapaxes(image, 1, 3)
        image = np.swapaxes(image, 2, 3)
        tensor = torch.from_numpy(image).float() / 255.
        if cfg.cuda:
            tensor = tensor.to(device)
        weight = [1.2, 1.4, 1.1, .7, 1., 1.]
        lsi = faceparsing_tensor(tensor, cfg, weight)
        for idx, tensor in enumerate(lsi):
            map = tensor.cpu().detach().numpy()
            cv2.imwrite("seg_out_tensor_{}.png".format(idx), map * 10)
    return lsi




class InferFaceSeg(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.model, _ = load_BiSeNet(cfg)
        self.w = cfg.w  # 人脸区域每一块的权重

    def __call__(self, imgs):
        """
        param imgs: torch tensor [B, H, W, C] rang: [0-1], not [0-255]
        param w: tuple len=6 [eyebrow,eye,nose,teeth,up lip,lower lip]
        return  tensor, shape:[H, W]
        """
        # imgs = rearrange(imgs, 'b c h w -> b h w c')
        # print(imgs.shape, self.__class__.__name__, imgs.max())
        out = self.model(imgs)  # B, 19, 256, 256
        out = self.w[0] * out[:, 3, ...] + self.w[1] * out[:,4, ...] + self.w[2] * out[:,10, ...] + out[:, 11, ...] + out[:,12,...] + out[:,13,...]
        # (B,112, 112), (B, H, W)
        return out
    
    def filter_imgs(self, imgs_dir):
        """
        因分割网络在公开数据集上，效果不好，用来筛选出有效数据
        """
        pass

if __name__ == "__main__":
    import glob
    # infer_face_seg()
    print("infer_seg")
    
