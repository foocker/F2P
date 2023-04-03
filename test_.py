from K2P.faceanalysis.faceparsing.infer import infer_face_seg,vis_parsing_maps, load_BiSeNet
from K2P.faceanalysis.parsing.dml_csr.infer import DML_CSR_Infer
from K2P.config.config import Config
import glob
import random

from K2P.module.Imitator.imatator import Imitator
import cv2
from K2P.dataset.celebafaedata import parser_jp, write_default_json, get_parameters_info, normalize_vec, innormalize_vec
import os
import shutil
import torch
 

def test_imitator(model_i, mode="celebA"):
    """
    mode: "celebA" or "character"
    """
    if mode != "celebA":
        pred_jp = "/data/data_320/test/Data/*.json"
        jp_list = glob.glob(pred_jp)
        jp_list_choosed = random.sample(jp_list, 20)
        name_map, gt_values, minvs, maxvs = get_parameters_info(jp_list_choosed[0])
        save_dir = "./test_result"
        gt_choosed_dir = "./gt_test_result"
        shutil.rmtree(gt_choosed_dir)
        shutil.rmtree(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        if not os.path.exists(gt_choosed_dir):
            os.makedirs(gt_choosed_dir, exist_ok=True)
        src_dir = "/data/data_320/test/Texture"

        for jp in jp_list_choosed:
            parameters, name_map = parser_jp(jp)
            pred_img = model_i.infer(model_i, parameters)
            name_index = os.path.basename(jp).split(".")[0]
            shutil.copyfile(os.path.join(src_dir, str(name_index) + ".png"), os.path.join(gt_choosed_dir, str(name_index) + ".png"))
            cv2.imwrite(os.path.join(save_dir, "imitator_70_pred_{}.png".format(name_index)), pred_img)
    elif mode == "celebA":
        pred_jp = "/data/data_320/test/Data/*.json"
        jp_list = glob.glob(pred_jp)
        jp_list_choosed = random.sample(jp_list, 20)
        
    else:
        raise NotImplementedError(f"mode {mode} is not implemented")


# def infer_hand_func(img_sublist, new_dir="/data/seg_celebA_crop_frontal"):
#     if not os.path.exists(new_dir):
#         os.makedirs(new_dir, exist_ok=True)

#     for imgf in img_sublist:
#         img = cv2.imread(imgf)
#         image = cv2.resize(img, (256, 256), cv2.INTER_LINEAR)
#         input_ = Transform(image)
#         input_ = torch.unsqueeze(input_, 0).to("cuda")
#         out = model(input_)
#         parsing = out.squeeze(0).cpu().detach().numpy().argmax(0)
#         vis_parsing_anno_color, _ = vis_parsing_maps(img, parsing)
#         img_name = os.path.basename(imgf)
#         cv2.imwrite(os.path.join(new_dir, img_name), vis_parsing_anno_color)

if __name__ == "__main__":
    cfgf = "configs/config.yaml"
    cfg = Config.fromfile(cfgf)
    
    # test imitator
    model_i = Imitator(cfg)
    test_imitator(model_i, cfg, mode="celebA")

    # # test face seg
    # img_p = "/data/data_320/test/Texture/10023.png"
    # img_d = "/data/img_align_celeba_crop_frontal"
    # img_list = glob.glob(img_d + "/*.jpg")
    # img_p = random.sample(img_list , 1)[0]
    # s = infer_face_seg(cfg, img_p)
    # # test xxx
    

# def main_process(process_num=8):
#     from multiprocessing import Process
#     one_split_num = len(img_list) // process_num

#     img_lists = [img_list[i*one_split_num:(i+1)*one_split_num] for i in range(process_num+1)]
#     process = [Process(target=infer_hand_func, args=(img_lists[i],)) for i in range(process_num)]
#     [p.start() for p in process]  # 开启进程
#     [p.join() for p in process]   # 等待进程依次结束

# if __name__ == "__main__":
#     import glob
#     cfgf = "configs/config.yaml"
#     cfg = Config.fromfile(cfgf)
#     # infer_face_seg()
#     print("infer_seg")
#     model, Transform = load_BiSeNet(cfg)
#     img_list = glob.glob('/data/img_align_celeba_crop_frontal/*.jpg')
    
#     main_process(8)
    

