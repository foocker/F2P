# import face_alignment
from skimage import io
import numpy as np
import torch

# https://github.com/atksh/onnx-facial-lmk-detector


# sfd for SFD, dlib for Dlib and folder for existing bounding boxes.
# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, face_detector="sfd")
# input = io.imread("../test/assets/aflw-test.jpg")
# preds = fa.get_landmarks(input)

class FAN(object):
    def __init__(self):
        import face_alignment
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    def run(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        out = self.model.get_landmarks(image)
        if out is None:
            return [0], 'kpt68'
        else:
            kpt = out[0].squeeze()
            left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
            top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
            bbox = [left,top, right, bottom]
            return bbox, 'kpt68'




