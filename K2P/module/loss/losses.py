import torch
import torch.nn.functional as F


def g_loss(g_x, engine_x):
    """
    x is face parameter from unity or other engine output by bone driver system
    g is genertor model Imatator

    g_x is the G(x), maybe replace by x_bar =  P^T(x-m)
    """
    return F.l1_loss(g_x,engine_x)

def facial_identity_loss(I_embeding, I_prime_embeding):
    """
     LightCNN-29v2 256-d or insightface 512-d as F_recg model
    """
    distance = torch.cosine_similarity(I_embeding, I_prime_embeding)
    return 1 - torch.mean(distance)

def circle_enforce_loss(I, I_prime):
    """
    I is input image traing generator model, G or Imatator
    # I_prime = G(T(F_recg(I)))
    """
    return F.l1_loss(I_prime, I)

def facial_content_loss(f_1, f_2):
    """
    I is input image 
    f_1 = F_seg(I)
    f_2 = F_seg(I_prime)
    """
    return F.l1_loss(f_1, f_2).sum()

def loopback_loss(x, x_prime):
    """
    inspired by the unsupervised 3D face reconstruction method, to further improve 
    the robustness of the parameter
    x is face parameter:
        x = T(f_recg(I))
    x_prime is as follows:
        I_prime = G(x) -> 
        x_prime = T(F_recg(I_prime))
    """
    return F.l1_loss(x, x_prime)

def mutil_loss():
    """
    l_1 * facial_identity_loss + l_2 * facial_content_loss + l_3 * loopback_loss + l_4 * circle_enforce_loss
    when training T:
        align at first 
        l_2 = 0 steps % 4 == 0, sampling from CelebA traning set 
        l_2 != 0, sampling from the high-quality frontal faces in CelebA dataset 
    """
    pass