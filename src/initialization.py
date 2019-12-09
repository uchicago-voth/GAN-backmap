import torch
import torch.nn as nn


def weights_init_G(m):
    classname = m.__class__.__name__
    if classname == "Linear":
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu'))




def weights_init_D(m):
    classname = m.__class__.__name__
    if classname == "Linear":
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu'))
