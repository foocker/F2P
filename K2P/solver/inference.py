import torch 
import cv2

# import sys
# sys.path.append('../../')

from K2P.module.Imitator.imatator import Imitator
from K2P.module.Translator.translator import Translator
from K2P.faceanalysis.insightface.iface import FaceInformation
from K2P.faceanalysis.facealignment.dlibalign import align_face
from K2P.faceanalysis.lightcnn.extract_features import get_feature, LightCNNFeature
from K2P.config.config import Config
from K2P.dataset.celebafaedata import write_default_json, write_result_json, innormalize_vec, get_parameters_info
from K2P.utils.utils import sigmoid

import os


def infer(cfg, img_file):
    img = cv2.imread(img_file)
    aligned_face = align_face(img, size=(256, 256))
    if cfg.face_embeding_method == "lightcnn":
        feature = LightCNNFeature(cfg)(aligned_face)
    else:
        feature = FaceInformation().face_embeding()  # TODO 
    feature = feature.to('cuda')
    T = Translator(cfg)
    model_T = T.from_pretrained(T, cfg.inference_checkpoint).to('cuda')
    model_T.eval()
    with torch.no_grad():
        parameters = model_T(feature)
        parameters_array = parameters.cpu().numpy().flatten()
        _, _, minvs, maxvs = get_parameters_info("./template.json")
        # p_sigmoid = sigmoid(parameters_array)
        # pred_v = innormalize_vec(minvs, maxvs, p_sigmoid)
        pred_v = innormalize_vec(minvs, maxvs, parameters_array)
        base_name = os.path.basename(img_file).split(".")[0]
        write_result_json(pred_v, save_file="./pred_parameters_{}.json".format(base_name))

    return parameters_array

if __name__ == "__main__":
    cfg_path = "/data/F2P/configs/config.yaml"
    cfg = Config.fromfile(cfg_path)
    img_p = "/data/data_320/T_/5058.png"
    parameters = infer(cfg, img_p)
    print(parameters)
    print()
    # import numpy as np
    # pred_v = np.random.randn(28)
    # print(pred_v)
    # write_result_json(pred_v, save_file="./pred_parameters_j.json")
