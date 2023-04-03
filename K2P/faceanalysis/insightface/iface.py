from insightface.app import FaceAnalysis
import cv2
import shutil
from K2P.fileio.path import mkdir_or_exist
import torch
import numpy as np


class FaceInformation:
    def __init__(self, detect_only=False) -> None:
        self.detect_only = detect_only
        if self.detect_only:
            self.app = FaceAnalysis(root="./weights/onnx",allowed_modules=['detection', 'recognition'])
        else:
            self.app = FaceAnalysis(root="./weights/onnx",allowed_modules=['detection','recognition', 'genderage', 'landmark_3d_68'])
        self.app.prepare(ctx_id=0, det_size=(256, 256))

    def info(self, img):
        faces = self.app.get(img)
        assert len(faces) == 1 , "there are more than one face or has not detected face."
        face = faces[0]
        aimg = face.aimg
        if not self.detect_only:
            kps = face.kps
            normed_embedding = face.normed_embedding
            pose = face.pose  # 
            sex = face.sex
        else:
            kps = None
            normed_embedding = None
            pose = None
            sex = None
        return normed_embedding, aimg, kps, pose, sex
    
    def __call__(self, imgs):
        # img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
        imgs = imgs.cpu().numpy().transpose((1, 2, 0)) * 255  # TODO to see this is right
        normed_embeddings = []
        for img in imgs:
            normed_embedding = self.info(img)[0]
            normed_embeddings.append(normed_embedding)
        return torch.from_numpy(np.array(normed_embeddings, dtype=np.float32)).to("cuda")
        
    def face_embeding(self):
        pass

    def face_frontal_intensity(self, pose, kps):
        """
        simple 
        """
        x_r, y_r, z_r = pose
        if abs(x_r) < 10.5 and abs(y_r) < 10 and abs(z_r):
            return True
        else:
            return False
    def save_img(self, img, save_p):
        cv2.imwrite(save_p, img)
    
    def cp_img(self, cp_src, cp_dst):
        shutil.copyfile(src=cp_src, dst=cp_dst)

    def make_frontal_data(self, img_dir, savebasedir_name="img_align_celeba_crop_frontal"):
        """
        img_dir: celeda data dir 
        savebasedir_name: new dir base name
        """
        import glob, os

        img_list = sorted(glob.glob(img_dir + "/*.jpg"))
        img_align_celeba_crop_frontal = os.path.join(os.path.dirname(img_dir),savebasedir_name)
        mkdir_or_exist(img_align_celeba_crop_frontal)

        for imgf in img_list:
            img = cv2.imread(imgf)
            try:
                normed_embedding, aimg, kps, pose, sex = self.info(img)
                frontal_flag = self.face_frontal_intensity(pose, kps)
            except:
                frontal_flag = False
                continue
            dst_p = os.path.join(img_align_celeba_crop_frontal, os.path.basename(imgf))
            if frontal_flag:
                # self.cp_img(imgf, dst_p)
                self.save_img(aimg, dst_p)
