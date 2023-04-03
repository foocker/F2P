import dlib
import cv2
import numpy as np


def generate_detector():
    predictor = dlib.shape_predictor('./weights/dat/shape_predictor_68_face_landmarks.dat')
    facerec = dlib.face_recognition_model_v1('./weights/dat/dlib_face_recognition_resnet_model_v1.dat')
    detector = dlib.get_frontal_face_detector()
    return detector, predictor, facerec

def align_face(img, size=(512, 512)):
    """
    :param img:  input photo, numpy array
    :param size: output shape
    :return: output align face image
    """
    if img.shape[0] * img.shape[1] > 512 * 512:
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    detector, predictor, facerec = generate_detector()
    dets = detector(img, 1)
    d = dets[0]  # default the first one
    bb = np.zeros(4, dtype=np.int32)

    ext = 8
    bb[0] = np.maximum(d.left() - ext, 0)
    bb[1] = np.maximum(d.top() - ext - 20, 0)
    bb[2] = np.minimum(d.right() + ext, img.shape[1])
    bb[3] = np.minimum(d.bottom() + 2, img.shape[0])

    # rec = dlib.rectangle(bb[0], bb[1], bb[2], bb[3])
    # shape = predictor(img, rec)  # get landmark
    # face_descriptor = facerec.compute_face_descriptor(img, shape)  # use resNet get 128d face feature 
    # face_array = np.array(face_descriptor).reshape((1, 128))

    # for testing 
    # cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (255, 255, 255), 1)
    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
    scaled = cv2.resize(cropped, size, interpolation=cv2.INTER_CUBIC)  # INTER_LINEAR
    return scaled
