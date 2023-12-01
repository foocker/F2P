import torch
import torch.nn as nn


class Translator(nn.Module):
    """
    1. Continuous head. This head firstly predicts a vector in an orthogonal space, then this vector is projected into the original
    parameter space,then this vector is projected into the original parameter space by Eq (6) (main paper) as continuous parameters
    2.Discrete head. This head predicts a vector containing several groups, each group corresponds a kind of makeup, e.g.
    hairstyle, eyebrow style
    3.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.res_att_1 = Residual_attention_block()
        self.res_att_2 = Residual_attention_block()
        self.res_att_3 = Residual_attention_block()
        self.res_att_4 = Residual_attention_block()
        self.res_att_5 = Residual_attention_block()
        self.res_att_6 = Residual_attention_block()
        # self.res_att_7 = Residual_attention_block()
        # self.res_att_8 = Residual_attention_block()
        self.fc1 = nn.Linear(cfg.embeding_dim, 512)
        self.fc_cn = nn.Linear(512, cfg.cn)
        if cfg.dn > 0:
            self.fc_dn = nn.Linear(512, cfg.dn)  # female, male. continus, discrete
        else:
            self.fc_dn = None
        # self.bn = nn.BatchNorm1d(cfg.cn)
        self.active = nn.Sigmoid()
        # self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # elif isinstance(module, nn.Embedding):
        #     torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # elif isinstance(module, nn.LayerNorm):
        #     torch.nn.init.zeros_(module.bias)
        #     torch.nn.init.ones_(module.weight)


    def continuous_projection(self):
        pass

    def forward(self, embeding):
        d1 = self.fc1(embeding)
        out = self.res_att_1(d1)+d1
        out = self.res_att_2(out)+out
        out = self.res_att_3(out)+out
        out = self.res_att_4(out)+out
        out = self.res_att_5(out)+out
        out = self.res_att_6(out)+out
        # ra_7 = self.res_att_7(ra_6 + ra_5)
        # ra_8 = self.res_att_8(ra_7 + ra_6)
        p_continuous = self.fc_cn(out)
        if self.cfg.projection_reduce:
            pass # TODO continuous_projection
        if self.fc_dn is not None:
            p_discrete = self.fc_dn(out)
            parameters = torch.cat((p_continuous, p_discrete), dim=1)
        else:
            parameters = p_continuous
        parameters = self.active(parameters)  # for normalize (0~1)
        return parameters

    # @classmethod
    def from_pretrained(self, model, inference_checkpoint):
        import os
        path_ = inference_checkpoint
        if not os.path.exists(path_):
            raise ("not exist checkpoint of imitator with path " + path_)
        if self.cfg.cuda:
            checkpoint = torch.load(path_)
        else:
            checkpoint = torch.load(path_, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        return model

class Residual_attention_block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 512)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        # self.bn = nn.BatchNorm2d(512)
        self.bn = nn.BatchNorm1d(512,track_running_stats=True)
        self.bn_ = nn.BatchNorm1d(512,track_running_stats=True)
        # self.bn.reset_parameters()
        # self.bn = nn.LayerNorm(512)
        self.bn.reset_parameters()
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.bn(self.fc(x)))
        out_f = self.bn_(self.fc1(out))
        out_ = self.relu(self.fc2(out_f))
        out_sig = self.sigmoid(self.fc3(out_))
        out_skip_att = out_f * out_sig  # TODO
        # out_skip_att = torch.mul(out_f, out_sig)
        return out_skip_att
