import torch
from K2P.faceanalysis.lightcnn.light_cnn import LightCNN_29Layers_v2
from torchvision import transforms
import cv2
import numpy as np


def get_tensor(file):
    import os
    if not os.path.exists(file):
        raise ("{} is not exist".format(file))
    transform = transforms.Compose([transforms.ToTensor()])

    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)  # bgr or gray 
    img = cv2.resize(img, (128, 128))   # orignal image should crop to 128*128
    img = img[:, :, np.newaxis]
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    return img

def get_feature(cfg, img):
    """
    img is img file or img array
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LightCNN_29Layers_v2(num_classes=cfg.num_classes)
    model.eval()
    import os
    if not os.path.exists(cfg.lightcnn_checkpoint):
        raise ("no checkpoint found at '{}'".format(cfg.lightcnn_checkpoint))
    if cfg.cuda:
        model = model.to(device)
        model = torch.nn.DataParallel(model)  # fdr some special weight key
        state_dict = torch.load(cfg.lightcnn_checkpoint, map_location="cuda")
        model.load_state_dict(state_dict["state_dict"])
    else:
        model = model.to("cpu")
        checkpoint = torch.load(cfg.lightcnn_checkpoint, map_location="cpu")
        new_state_dict = model.state_dict()
        for k, v in checkpoint["state_dict"].items():
            _name = k[7:]  # remove `module.`
            new_state_dict[_name] = v
        model.load_state_dict(new_state_dict)

    if not img.is_cuda and cfg.cuda:
        img = img.to(device)
    with torch.no_grad():
        _, feature = model(img)
    return feature.squeeze(0).cpu().detach().numpy()

def load_model(model, mpath, device="cuda"):
    import os
    if not os.path.exists(mpath):
        raise ("no checkpoint found at '{}'".format(mpath))
    if device == "cuda":
        model = model.to(device)
        model = torch.nn.DataParallel(model)  # fdr some special weight key
        state_dict = torch.load(mpath, map_location="cuda")
        model.load_state_dict(state_dict["state_dict"])
    else:
        model = model.to("cpu")
        checkpoint = torch.load(mpath, map_location="cpu")
        new_state_dict = model.state_dict()
        for k, v in checkpoint["state_dict"].items():
            _name = k[7:]  # remove `module.`
            new_state_dict[_name] = v
        model.load_state_dict(new_state_dict)

    return model


class LightCNNFeature(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.model = LightCNN_29Layers_v2(num_classes=cfg.num_classes)
        self.model.eval()
        self.model = load_model(self.model, cfg.lightcnn_checkpoint)
        if cfg.task == "Infer":
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128)), transforms.Grayscale()])
        else:
            self.transform = transforms.Compose([transforms.Resize((128, 128)), transforms.Grayscale()])
    
    def __call__(self, imgs):
        imgs = self.transform(imgs)
        if self.cfg.task == "Infer":
            imgs = imgs.unsqueeze(0).to('cuda')
        _, features = self.model(imgs)
        return features