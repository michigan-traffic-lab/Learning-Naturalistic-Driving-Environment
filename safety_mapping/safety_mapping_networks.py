import torch
import torch.nn as nn
from .bert import Embeddings, Block, Config


class PositionalMapping(nn.Module):
    """
    Positional mapping Layer.
    This layer map continuous input coordinates into a higher dimensional space
    and enable the prediction to more easily approximate a higher frequency function.
    See NERF paper for more details (https://arxiv.org/pdf/2003.08934.pdf)
    """

    def __init__(self, input_dim, L=5, pos_scale=0.01, heading_scale=1.0):
        super(PositionalMapping, self).__init__()
        self.L = L
        self.input_dim = input_dim
        self.output_dim = input_dim * (L*2 + 1)
        # self.scale = scale
        self.pos_scale = pos_scale
        self.heading_scale = heading_scale

    def forward(self, x):

        if self.L == 0:
            return x

        x_scale = x.clone()
        x_scale[:, :, :int(2 * self.input_dim / 3)] = x_scale[:, :, :int(2 * self.input_dim / 3)] * self.pos_scale
        x_scale[:, :, int(2 * self.input_dim / 3):] = x_scale[:, :, int(2 * self.input_dim / 3):] * self.heading_scale

        h = [x]
        PI = 3.1415927410125732
        for i in range(self.L):
            x_sin = torch.sin(2**i * PI * x_scale)
            x_cos = torch.cos(2**i * PI * x_scale)

            x_sin[:, :, :int(2 * self.input_dim / 3)], x_sin[:, :, int(2 * self.input_dim / 3):] = x_sin[:, :, :int(2 * self.input_dim / 3)] / self.pos_scale, x_sin[:, :, int(2 * self.input_dim / 3):] / self.heading_scale
            x_cos[:, :, :int(2 * self.input_dim / 3)], x_cos[:, :, int(2 * self.input_dim / 3):] = x_cos[:, :, :int(2 * self.input_dim / 3)] / self.pos_scale, x_cos[:, :, int(2 * self.input_dim / 3):] / self.heading_scale

            h.append(x_sin)
            h.append(x_cos)

        return torch.cat(h, dim=-1)


def load_pretrained_weights(net_G, checkpoint_dir, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    print('loading pre-trained safety mapper...')
    checkpoint = torch.load(checkpoint_dir, map_location=device)
    net_G.load_state_dict(checkpoint['model_G_state_dict'])
    return net_G


def define_safety_mapping_networks(model, input_dim, output_dim, m_tokens):

    h_dim = 256

    # define input positional mapping
    M = PositionalMapping(input_dim=input_dim, L=4, pos_scale=0.01, heading_scale=0.01)

    # define backbone networks
    if model == 'transformer':
        bert_cfg = Config()
        bert_cfg.dim = h_dim
        bert_cfg.n_layers = 4
        bert_cfg.n_heads = 4
        bert_cfg.max_len = m_tokens
        Backbone = Transformer(input_dim=M.output_dim, output_dim=output_dim, m_tokens=m_tokens, cfg=bert_cfg)
    else:
        raise NotImplementedError(
            'Wrong backbone model name %s (choose one from [transformer])' % model)

    return nn.Sequential(M, Backbone)


class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""

    def __init__(self, input_dim, output_dim, m_tokens, cfg=Config()):
        super().__init__()
        self.in_net = nn.Linear(input_dim, cfg.dim, bias=True)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.m_tokens = m_tokens
        self.out_net = nn.Linear(in_features=cfg.dim, out_features=output_dim, bias=True)

    def forward(self, x):
        # shape x: batch_size x m_token x m_state
        h = self.in_net(x)
        for block in self.blocks:
            h = block(h, None)
        y = self.out_net(h)
        return y