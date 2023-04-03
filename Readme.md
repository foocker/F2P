# Using
## data
prepare your own dataset and write your own cleabafacedata.py(class of FaceParameters)
## train, infer
1. change the config.yaml 
    1.5 task I, T, Infer
    1.9 python test_.py
2. python run.py
3. mediapipe_test.py is a hardcode method(added new plan), not compelete yet, but does not affect the original program.

# reference code
face-nn, mmcv config. 

## weights dir
1. effect of factorch is very general
```
./weights
./dat
├── 79999_iter.pth
├── dlib_face_recognition_resnet_model_v1.dat
├── LightCNN_29Layers_V2_checkpoint.pth.tar
├── resnet18-5c106cde.pth
└── shape_predictor_68_face_landmarks.dat
./onnx
└── models
    └── buffalo_l
        ├── 1k3d68.onnx
        ├── 2d106det.onnx
        ├── det_10g.onnx
        ├── genderage.onnx
        └── w600k_r50.onnx
./torchscript
├── detector
│   ├── 1
│   │   └── model.pt
│   ├── 16
│   │   └── model.pt
│   └── 2
│       └── model.pt
└── predictor
    ├── align
    │   └── 1
    │       └── model.pt
    ├── antispoof
    │   └── 1
    │       └── model.pt
    ├── au
    │   ├── 1
    │   │   └── model.pt
    │   └── 2
    │       └── model.pt
    ├── deepfake
    │   └── 1
    │       └── model.pt
    ├── embed
    │   └── 1
    │       └── model.pt
    ├── fer
    │   ├── 1
    │   │   └── model.pt
    │   └── 2
    │       └── model.pt
    ├── segmentation
    └── verify
        ├── 1
        │   └── model.pt
        └── 2
            └── model.pt
```