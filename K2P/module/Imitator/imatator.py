import torch.nn as nn
import torch
import os


def deconv_layer(in_chanel, out_chanel, kernel_size, stride=1, pad=0):
    """
    反卷积layer, CT->BN->Relu 主要用于上采样
    公式： output = (input - 1) * stride + outputpadding - 2 * padding + kernel_size
    :param in_chanel: chanel in, int
    :param out_chanel: chanel out, int
    :param kernel_size: triple or int
    :param stride: conv stride
    :param pad: pad
    :return: nn.Sequential
    """
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_chanel, out_chanel, kernel_size=kernel_size, stride=stride, padding=pad
        ),
        # nn.PixelShuffle(2)
        nn.BatchNorm2d(out_chanel),
        # nn.ReLU(),
        nn.LeakyReLU()  # change 
    )

class Imitator(nn.Module):
    def __init__(self, cfg):
        """
        imitator
        :param name: imitator name
        :param cfg: argparse options
        """
        super().__init__()
        self.cfg = cfg
        self.model = nn.Sequential(
            deconv_layer(cfg.params_cnt, 512, kernel_size=4),  # 1. (batch, 512, 4, 4)
            deconv_layer(
                512, 512, kernel_size=4, stride=2, pad=1
            ),  # 2. (batch, 512, 8, 8)
            deconv_layer(
                512, 512, kernel_size=4, stride=2, pad=1
            ),  # 3. (batch, 512, 16, 16)
            deconv_layer(
                512, 256, kernel_size=4, stride=2, pad=1
            ),  # 4. (batch, 256, 32, 32)
            deconv_layer(
                256, 128, kernel_size=4, stride=2, pad=1
            ),  # 5. (batch, 128, 64, 64)
            deconv_layer(
                128, 64, kernel_size=4, stride=2, pad=1
            ),  # 6. (batch, 64, 128, 128)
            deconv_layer(
                64, 3, kernel_size=4, stride=2, pad=1
            ),  # 7. (batch, 64, 256, 256)  (batch, 3, 256, 256)
            # nn.ConvTranspose2d(
            #     64, 3, kernel_size=4, stride=2, padding=1
            # ),  # 8. (batch, 3, 512, 512)
            nn.Sigmoid(),
            # nn.Tanh(),   # change for[-1, 1] if image normalize to [-1, 1]
        )
        self.model.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # if module.bias is not None:
        #     torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)


    def forward(self, parameters):
        """
        forward module
        :param params: [batch, params_cnt]
        :return: (batch, 3, 512, 512)
        """
        batch = parameters.size(0)
        length = parameters.size(1)
        _params = parameters.reshape((batch, length, 1, 1))
        return self.model(_params)

    @staticmethod
    def capture(self):
        pass

    def load_pretrained(self):
        path_ = os.path.join(self.cfg.imatator_checkpoint_dir, self.cfg.infer_model_name_i)
        if not os.path.exists(path_):
            raise ("not exist checkpoint of imitator with path " + path_)
        if self.cfg.cuda:
            checkpoint = torch.load(path_)
        else:
            checkpoint = torch.load(path_, map_location="cpu")
        # return self.model.load_state_dict(checkpoint["model"])  # TODO 
        return checkpoint

    def infer(self, model, parameters):
        import numpy as np
        checkpoint = self.load_pretrained()
        model.load_state_dict(checkpoint["model"])
        model = model.to("cuda").double()
        with torch.no_grad():
            out = model(parameters)
        out = out.cpu().squeeze(0).permute(1, 2, 0).numpy() * 255
        return out.astype(np.uint8)