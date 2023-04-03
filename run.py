from K2P.solver.trainer import Trainer
from K2P.solver.inference import infer
from K2P.config.config import Config

if __name__ == "__main__":
    cfgf = "configs/config.yaml"
    cfg = Config.fromfile(cfgf)
    # img_p = "/data/data_320/test/Texture/5113.png"
    # img_p = "/data/data_320_test_male/Texture/604.png"
    # img_p = "/data/data_320/Texture/218.png"
    img_p = "./test_imgs/jk.jpg"  # 000353 , 000919, jk, emo
    
    if cfg.task == "Infer":
        parameters = infer(cfg, img_p)
        print(parameters)
    else:
        T = Trainer(cfg)
        if cfg.task == "I":
            T.train_single()
        elif cfg.task == "T":
            T.train_mutil()
        else:
            raise NotImplementedError
        # tensorboard --logdir_spec=./output/logs/