import os
import torch
from torch.utils.data.dataloader import DataLoader
from K2P.dataset.celebafaedata import FaceParameters,CelebaFace
import K2P.logs.logit as log
from K2P.module.Imitator.imatator import Imitator
from K2P.module.loss.losses import * 
from K2P.module.Translator.translator import Translator
from K2P.faceanalysis.insightface.iface import FaceInformation
from K2P.faceanalysis.faceparsing.infer import InferFaceSeg
from K2P.faceanalysis.lightcnn.extract_features import LightCNNFeature

from torch.optim import Adam
from tqdm import tqdm

from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingWarmRestarts, StepLR, CosineAnnealingLR

from K2P.utils.utils import evaluate_parameters
import numpy as np
import copy

from K2P.faceanalysis.parsing.dml_csr.infer import DML_CSR_Infer, valid, valid_
from K2P.faceanalysis.parsing.dml_csr.dataset.datasets import FaceDataSet

from torchvision.transforms import transforms

import torch.backends.cudnn as cudnn

from torchvision.utils import make_grid, save_image


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lr = cfg.lr
        self.initial_epoch = 0
        cudnn.benchmark = True
        cudnn.enabled = True

        if self.cfg.task == "I":
            self.model_I = Imitator(cfg)
            self.model = self.model_I.to('cuda')  # double()

        elif self.cfg.task == "T":
            self.model_I = Imitator(cfg).to('cuda')
            checkpoint = self.model_I.load_pretrained()
            self.model_I.load_state_dict(checkpoint["model"])
            self.model_I = self.model_I.to("cuda").float()  # float()
            self.freeze_(self.model_I)
            
            self.model_recg = NotImplemented
            # self.model_seg = InferFaceSeg(cfg) # TODO 效果差，需重新训练
            self.model_seg = DML_CSR_Infer(cfg)
            self.freeze_(self.model_seg.model)  # for other block 
            self.model_T = Translator(cfg)  # TODO 
            self.model = self.model_T.to('cuda').float()
            if self.cfg.face_embeding_method == "lightcnn":
                print(self.cfg.face_embeding_method)
                self.model_recg = LightCNNFeature(cfg)
                self.freeze_(self.model_recg.model)
            elif self.cfg.face_embeding_method == "insightface":
                self.model_recg = FaceInformation()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        self.configure_optimizers()
            
        self.faceparameters_dataset = FaceParameters(cfg, mode="train")
        self.minvs, self.maxvs = self.faceparameters_dataset.minvs, self.faceparameters_dataset.maxvs

        self.faceparameters_data_loader = DataLoader(dataset=self.faceparameters_dataset, batch_size=self.cfg.batch_size, 
        shuffle=False,  num_workers=8, pin_memory=True, drop_last=True)
        self.faceparameters_dataset_test = FaceParameters(cfg, mode="test")
        self.faceparameters_data_test_loader = DataLoader(dataset=self.faceparameters_dataset_test, batch_size=self.cfg.batch_size, 
        shuffle=False,  num_workers=8, pin_memory=True, drop_last=True)
        self.loss = g_loss
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225])
        # self.dataset_ = FaceDataSet("/data/data_320/test/Texture", "test", crop_size=(256, 256), transform=transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float32),normalize]))
        # self.valloader = DataLoader(self.dataset_, batch_size=6, shuffle=False, pin_memory=True)
        
        
        self.celeba_frontal_dataset = CelebaFace(self.cfg, frontal=True)
        self.celeba_dataset = CelebaFace(self.cfg, frontal=False)
        self.celeba_front_loader = DataLoader(dataset=self.celeba_frontal_dataset,batch_size=self.cfg.batch_size, 
        shuffle=False,  num_workers=8, pin_memory=True, drop_last=True)
        self.celeba_loader = DataLoader(dataset=self.celeba_dataset,batch_size=self.cfg.batch_size, 
        shuffle=False,  num_workers=8, pin_memory=True, drop_last=True)

        self.writer = SummaryWriter("./output/logs")
        
    def freeze_(self, module):
        for p in module.parameters():
            p.requires_grad  = False
    
    def train_step(self, x, y, *args, **kwargs):
        """
        single model training and mutil input for more than one model,do not considering distribution
        """
        self.optimizer.zero_grad()
        out = self.model.forward(x)
        loss = self.loss(y, out)
        loss.backward()
        self.optimizer.step()
        return loss, out
    def evaluate(self, data_loader, model, threshold_list= [0.1*28, 0.2*28, 0.3*28]):
        gt_p = []
        pred_p = []
        model = copy.deepcopy(model)
        model = model.double()
        results = []  # lower_dis_threshold_rate, keep_sign, combine
        for step, (imgs, params) in enumerate(tqdm(data_loader)):
            imgs = imgs.to('cuda')   # (B, 3, 112, 112)
            face_embeding = self.model_recg(imgs.float()) 
            face_parameters = model(face_embeding.double())
            gt_p.append(params.cpu().detach().numpy())
            pred_p.append(face_parameters.cpu().detach().numpy())
            
        gt_p = np.concatenate(gt_p, axis=0)
        pred_p = np.concatenate(pred_p, axis=0)
        
        for th in threshold_list:
            result = evaluate_parameters(gt_p, pred_p, sign_rate=0.5, dist_threshold=th, minvs=self.minvs, maxvs=self.maxvs)
            results.append(result)
        return results
    
    def train_single(self):
        if self.cfg.finetune:
            self.load_checkpoint()
        self.model.float()
        # self.model.double()
        self.model.train()
        log.info("Start from {} epoch".format(self.initial_epoch))
        for ep in range(self.initial_epoch, self.cfg.epoch):
            # for params, images in tqdm(self.faceparameters_data_loader):
            for params, images in tqdm(self.faceparameters_data_test_loader):
                images, params = params.to('cuda'), images.to('cuda')
                loss, out = self.train_step(params, images)
                self.writer.add_scalar('imitator/loss', loss, ep)
                self.writer.add_scalar('imitator/lr', self.optimizer.param_groups[0]["lr"], ep)
                if (ep+1) % self.cfg.save_epoch_freq == 0:
                    self.writer.add_images('imitator/imgs', out, ep)
                    self.writer.add_embedding(mat=params,metadata=None,label_img=images, global_step=ep, tag='imitator/embedding')
                if (ep+1) % self.cfg.save_epoch_freq == 0:
                    self.save_checkpoint(ep+1)
                    log.info('save')
            self.scheduler.step()
            log.info("compeleted {} and loss is {}".format(ep, loss))
        self.writer.close()

    def train_mutil(self):
        l_1, l_2, l_3, l_4, l_5 = 0.5, 2, 0.1, 1.5, 3  # 0.01, 1, 1, 0
        if self.cfg.finetune:
            self.load_checkpoint()
            # with open("./W_10.txt", 'w') as f:
            #     print(self.model.state_dict(), file=f)

        self.model.train()
        log.info("Start from {} epoch".format(self.initial_epoch))
        
        # 1. 提供先验，先让T在有标注的Character上训练，主要是loopback_loss 需要首先完成这一步，
        # 2. 然后在CelebA上训练， 使T在通用照片上能将映射的参数和Character对齐
        # 3. 最后两者混合训练，在第2步，可以将通用数据CelebA做PCA处理
        # epoch >= 50 = 10 + 30 + 10 , initial_epoch = 0
        # 4. 最后在正面人脸上训练20 ep
        
        log.info("Start Evaluating on trian for imitator dataset")
        # results = self.evaluate(self.faceparameters_data_loader, self.model)
        results = self.evaluate(self.faceparameters_data_test_loader, self.model)  # 新增数据
        
        for r in results:
            lower_dis_threshold_rate, keep_sign, combine = r
            log.info("model_{} evaulate base 0.1, 0.2, 0.3 * 28 on dis_rate, keep_sign, combine are {}, {}, {}".
                        format(self.initial_epoch, lower_dis_threshold_rate,keep_sign, combine))
        
        for ep in range(self.initial_epoch, self.cfg.epoch):
            # if ep <= 10:
            #     data_loader = self.faceparameters_data_test_loader
            # elif 10 < ep <= 60:
            #     if ep % 4 == 0:
            #         l_2 = 0
            #         data_loader = self.celeba_front_loader
            #     else:
            #         l_2 = 1
            #         data_loader = self.celeba_loader
            # else:
            #     data_loader = self.faceparameters_data_test_loader
            
            # for finetune only on celeba_front_loader
            data_loader = self.celeba_front_loader
            for step, (imgs, labels) in enumerate(tqdm(data_loader)):
                self.optimizer.zero_grad()
                imgs = imgs.to('cuda')   # (B, 3, 112, 112)
                # imgs_cp = copy.deepcopy(imgs)
                # face_seg = self.model_seg(imgs_cp)

                face_seg = self.model_seg(imgs)  # (B, 112, 112), in original model is 448 
                
                face_embeding = self.model_recg(imgs)  # (B, feature_embeding_dim), (B, 256) 0r (B, 512)
                face_parameters = self.model(face_embeding) 
                
                fake_face = self.model_I(face_parameters)  # 出256 or 512 大小图 2**n
                # add test imitator on the celebA dataset 
                # save_image(fake_face, f"./{step}_x.png")
                fake_face_embeding = self.model_recg(fake_face)  #infer 
                fake_face_seg = self.model_seg(fake_face)  # infer  this fake img can evaluate performance of the model
                fake_face_parameters = self.model(fake_face_embeding)
                L_idt = facial_identity_loss(face_embeding, fake_face_embeding)
                # if torch.sum(labels[0])!=0:
                if labels.shape==(self.cfg.batch_size, 28):
                    labels = labels.to('cuda')
                    L_loop_label = loopback_loss(face_parameters, labels)
                else:
                    L_loop_label = 0
                L_loop = loopback_loss(face_parameters, fake_face_parameters)
                L_ctt = facial_content_loss(face_seg, fake_face_seg)
                L_cef = circle_enforce_loss(imgs, fake_face)
                loss = l_1 * L_idt + l_2 * L_loop + l_3 * L_ctt + l_4 * L_cef + l_5 * L_loop_label
                # loss = l_1 * L_idt + l_2 * L_loop + l_4 * L_cef + l_5 * L_loop_label
                # log.info("seperate loss of eooch {} is {}, {}, {}, {}, {}".format(ep+1, l_1*L_idt, l_2*L_loop, l_3*L_ctt, l_4*L_cef, l_5 * L_loop_label))
                
                self.writer.add_scalar('translator/loss', loss, ep)
                if (ep+1) % self.cfg.save_epoch_freq == 1:
                    # np.array(fake_face, dtype=np.uint8)
                    self.writer.add_images('translator/imgs_fake_face', fake_face, ep)
                    self.writer.add_images('translator/imgs_fake_seg', fake_face_seg.unsqueeze(1), ep, dataformats="NCHW")
                    self.writer.add_images('translator/imgs_fake_seg_np', fake_face_seg.unsqueeze(1).cpu().numpy().astype(np.uint8)*4, ep, dataformats="NCHW")
                    
                    self.writer.add_embedding(mat=face_embeding,metadata=None,label_img=imgs, global_step=ep+1, tag='translator/faceembedding')
                    self.writer.add_embedding(mat=fake_face_embeding,metadata=None,label_img=fake_face, global_step=ep+1, tag='translator/fakeembedding')
                    
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()
            if (ep + 1 ) % self.cfg.eval_freq == 1:
                log.info("Start Evaluating")
                results = self.evaluate(self.faceparameters_data_test_loader, self.model)
                for r in results:
                    lower_dis_threshold_rate, keep_sign, combine = r
                    log.info("model_{} evaulate base 0.1, 0.2, 0.3 * 28 on dis_rate, keep_sign, combine are {}, {}, {}".
                             format(ep+1, lower_dis_threshold_rate,keep_sign, combine))

            log.info("total loss of epoch {} is {}".format(ep+1, loss))
            log.info("seperate loss of eooch {} is {}, {}, {}, {}, {}".format(ep+1, l_1*L_idt, l_2*L_loop, l_3*L_ctt, l_4*L_cef, l_5 * L_loop_label))
            # log.info("seperate loss of eooch {} is {}, {}, {}".format(ep+1, l_1*L_idt, l_2*L_loop, l_4*L_cef))
            
            if (ep + 1) % self.cfg.save_epoch_freq == 0:
                self.save_checkpoint(ep+1) 
            
        self.writer.close()

                
    def configure_optimizers(self):
        if self.cfg.task == "I":
            self.optimizer = Adam(self.model.parameters(), self.lr)
            # self.scheduler = CyclicLR(self.optimizer, base_lr=0.001, max_lr=0.01,step_size_up=5,mode="triangular2", cycle_momentum=False)
            self.scheduler = CyclicLR(self.optimizer, base_lr=0.001, max_lr=0.05,step_size_up=10,mode="triangular2", cycle_momentum=False)
            # self.scheduler = CosineAnnealingWarmRestarts()
            # self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
            # self.scheduler = CosineAnnealingLR()
        elif self.cfg.task == "T":
            self.optimizer = Adam(self.model.parameters(), self.lr)
            self.scheduler = CyclicLR(self.optimizer, base_lr=0.001, max_lr=0.05,step_size_up=5,mode="triangular2", cycle_momentum=False)
        else:
            raise NotImplementedError
        
    def load_checkpoint(self):
        path_ = self.cfg.translator_checkpoint_dir if self.cfg.task == "T" else self.cfg.imatator_checkpoint_dir
        name_ = self.cfg.infer_model_name_t if self.cfg.task == "T" else self.cfg.infer_model_name_i
        path_ = os.path.join(path_, name_)  # choose model name
        if not os.path.exists(path_):
            raise ("not exist checkpoint of imitator with path " + path_)
        if self.cfg.cuda:
            checkpoint = torch.load(path_)
        else:
            checkpoint = torch.load(path_, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.initial_epoch = checkpoint["epoch"]

    def save_checkpoint(self, epoch):
        """"
        self.model = self.model_T, 更新T会将参数更新到model? 浅拷贝
        """
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
        }
        mdoel_name = self.model.__class__.__name__
        path_ = self.cfg.imatator_checkpoint_dir if self.cfg.task == "I" else self.cfg.translator_checkpoint_dir
        if not os.path.exists(path_):
            os.makedirs(path_, exist_ok=True)
        torch.save(
            state, "{}/{}_{}_{}.pth".format(path_, mdoel_name, epoch, self.cfg.save_model_tag_info)
        )

    def visual(self):
        pass
