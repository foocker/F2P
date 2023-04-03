import torch
import torch.backends.cudnn as cudnn

from inplace_abn import InPlaceABN

from K2P.faceanalysis.parsing.dml_csr.networks import dml_csr
import numpy as np
import cv2
import os

def img_edge(img):
    """
    提取原始图像的边缘
    :param img: input image
    :return: edge image
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    x_grad = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)
    y_grad = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)
    return cv2.Canny(x_grad, y_grad, 40, 130)


def valid(model, valloader, input_size, num_samples, dir=None, dir_edge=None, dir_img=None):

    height = input_size[0]
    width  = input_size[1]
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, \
        record_shapes=False, profile_memory=False) as prof:
        model.eval()
        parsing_preds = np.zeros((num_samples, height, width), dtype=np.uint8)
        scales = np.zeros((num_samples, 2), dtype=np.float32)
        centers = np.zeros((num_samples, 2), dtype=np.int32)

        idx = 0
        interp = torch.nn.Upsample(size=(height, width), mode='bilinear', align_corners=True)

        with torch.no_grad():
            for index, batch in enumerate(valloader):
                image, meta = batch  # B, 3, 256, 256
                print(image.max(), image.min(), 'maxmin')

                num_images = image.size(0)
                if index % 10 == 0:
                    print('%d  processd' % (index * num_images))

                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                scales[idx:idx + num_images, :] = s[:, :]
                centers[idx:idx + num_images, :] = c[:, :]

                results = model(image.cuda())
                outputs = results

                if isinstance(results, list):
                    outputs = results[0]
                    print('jsk')

                if isinstance(outputs, list):
                    for k, output in enumerate(outputs):
                        parsing = output
                        nums = len(parsing)
                        parsing = interp(parsing).data.cpu().numpy()
                        parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                        parsing_preds[idx:idx + nums, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)
                        idx += nums
                else:
                    w = torch.tensor([0, 1, 1, 1, 2, 2, 2, 2, 1.5, 2, 0], dtype=torch.float32).to("cuda")
                    parsing = results
                    b = parsing.shape[0]
                    parsing = parsing.permute(0, 2, 3, 1)
                    weighted_parsing = w * parsing
                    
                    max_index = torch.argmax(weighted_parsing, dim=-1, keepdim=True)
                    parsing_ = torch.gather(weighted_parsing, -1, max_index).squeeze(-1)
                    for i in range(b):
                        img = parsing_[i].cpu().numpy().astype(np.uint8)
                        ied = img_edge(img)
                        cv2.imwrite(os.path.join("./", 'gg_ed' + str(i)+'.png'), ied)
                    for i in range(b):
                        cv2.imwrite(os.path.join("./", 'gg_' + str(i)+'.png'), parsing_[i].cpu().numpy().astype(np.uint8)* 5)
                    
                    idx += num_images
        parsing_preds = parsing_preds[:num_samples, :, :]

    return parsing_preds, scales, centers

def valid_(model, imgs):

    with torch.no_grad():
        image = imgs
        print(image.max(), image.min(), 'maxmin')
        results = model(image.cuda())

        w = torch.tensor([0, 1, 1, 1, 2, 2, 2, 2, 1.5, 2, 0], dtype=torch.float32).to("cuda")
        parsing = results
        b = parsing.shape[0]
        parsing = parsing.permute(0, 2, 3, 1)
        weighted_parsing = w * parsing
        
        max_index = torch.argmax(weighted_parsing, dim=-1, keepdim=True)
        parsing_ = torch.gather(weighted_parsing, -1, max_index).squeeze(-1)
        for i in range(b):
            img = parsing_[i].cpu().numpy().astype(np.uint8)
            ied = img_edge(img)
            cv2.imwrite(os.path.join("./", 'gg_ed' + str(i)+'.png'), ied)
        for i in range(b):
            cv2.imwrite(os.path.join("./", 'gg_' + str(i)+'.png'), parsing_[i].cpu().numpy().astype(np.uint8)* 5)
    return parsing_


class DML_CSR_Infer(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        cudnn.benchmark = True
        cudnn.enabled = True
        self.model = dml_csr.DML_CSR(cfg.seg_num_classes, InPlaceABN, False)  # seg 
        state_dict = torch.load(cfg.seg_checkpoint, map_location='cuda:0')
        print(cfg.seg_checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.cuda()
        self.model.eval()
        self.interp = torch.nn.Upsample(size=(self.cfg.dst_size, self.cfg.dst_size), mode='bilinear', align_corners=True)
        self.w = torch.tensor(self.cfg.w, dtype=torch.float32).to("cuda")
        # for p in self.model.parameters():
        #     print(p)
        #     break
            
    def __call__(self, imgs):
        # s = valid_(self.model, imgs)
        # return s
        
        with torch.no_grad():
            results = self.model(imgs)
            # parsing = self.interp(results)
            parsing = results
            b = parsing.shape[0]
            # parsing = interp(results)
            
            parsing = parsing.permute(0, 2, 3, 1)
            weighted_parsing = self.w * parsing
            max_index = torch.argmax(weighted_parsing, dim=-1, keepdim=True)
            parsing_ = torch.gather(weighted_parsing, -1, max_index).squeeze(-1)  # experiance
            # print(parsing_.shape, parsing_.max(), parsing_.min())
            # it is better to normalize it to (0~1)?
            # parsing_ = torch.sigmoid(parsing_)
    
            # for i in range(b):
            #     im = parsing_[i].cpu().numpy().astype(np.uint8)*4
            #     # print(im.max(), im.min())
            #     cv2.imwrite(os.path.join("./", 'gg_' + str(i)+'.png'), im)

        return parsing_
    
    def infer_img(self, img):
        pass
