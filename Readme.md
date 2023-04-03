# Using
reproduce [Face-to-Parameter-V2](https://github.com/FuxiCV/Face-to-Parameter-V2)
## data
prepare your own dataset and write your own cleabafacedata.py(class of FaceParameters)
## train, infer
1. change the config.yaml(task I, T, Infer)
2. python run.py
3. mediapipe_test.py is a hardcode method(added new plan), not compelete yet, but does not affect the original program.
4. python test_.py for test imitator and other
# reference code
face-nn, mmcv config. 

## Directory Structure and weights dir 
1. effect of factorch is very general
```
# cd F2P, tree -L 2 -d
.
├── configs
├── K2P
│   ├── config
│   ├── dataset
│   ├── enginenet
│   ├── faceanalysis
│   ├── fileio
│   ├── logs
│   ├── module
│   ├── solver
│   ├── utils
│   └── visual
├── mediapipe_results
├── output
│   ├── checkpoints
│   ├── logs
│   └── testvisual
├── test_imgs
├── test_result
│   ├── embeding_imgs
│   ├── gt_test_result
│   ├── imitator
│   ├── line_imgs
│   ├── parameter_add
│   ├── simulate_imgs
│   └── video
└── weights
    ├── dat
    ├── onnx
    └── torchscript
```

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
