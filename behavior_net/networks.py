import torch
import torch.nn as nn
from .bert import Embeddings, Block, Config
from safety_mapping.safety_mapping_networks import define_safety_mapping_networks, load_pretrained_weights

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def define_G(model, input_dim, output_dim, m_tokens):

    h_dim = 256

    # define input positional mapping
    M = PositionalMapping(input_dim=input_dim, L=4, pos_scale=0.01, heading_scale=1.0)

    # define backbone networks
    if model == 'simple_mlp':
        Backbone = SimpleMLP(input_dim=M.output_dim, h_dim=h_dim, m_tokens=m_tokens)
    elif model == 'bn_mlp':
        Backbone = BnMLP(input_dim=M.output_dim, h_dim=h_dim, m_tokens=m_tokens)
    elif model == 'transformer':
        bert_cfg = Config()
        bert_cfg.dim = h_dim
        bert_cfg.n_layers = 4
        bert_cfg.n_heads = 4
        bert_cfg.max_len = m_tokens
        Backbone = Transformer(input_dim=M.output_dim, m_tokens=m_tokens, cfg=bert_cfg)
    else:
        raise NotImplementedError(
            'Wrong backbone model name %s (choose one from [simple_mlp, bn_mlp, transformer])' % model)

    # define prediction heads
    P = PredictionsHeads(h_dim=h_dim, output_dim=output_dim)

    return nn.Sequential(M, Backbone, P)


def define_D(input_dim):

    # define input positional mapping
    M = PositionalMapping(input_dim=input_dim, L=4, pos_scale=0.01, heading_scale=1.0)
    D = Discriminator(input_dim=M.output_dim)

    return nn.Sequential(M, D)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()

        self.mlp = nn.Sequential(
            # input is batch x m_token x input_dim
            nn.Linear(input_dim, 1024, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1, bias=True)
        )

    def forward(self, input):
        return self.mlp(input)


def define_safety_mapper(safety_mapper_ckpt_dir, m_tokens, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    # define neural safety mapping network
    SafetyMapping_M = SafetyMapper(safety_mapper_ckpt_dir, m_tokens, device)
    SafetyMapping_M.eval()
    set_requires_grad(SafetyMapping_M, requires_grad=False)
    return SafetyMapping_M


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
        self.output_dim = input_dim * (L*2 + 1)
        # self.scale = scale
        self.pos_scale = pos_scale
        self.heading_scale = heading_scale

    def forward(self, x):

        if self.L == 0:
            return x

        # x = x * self.scale
        x_scale = x.clone()
        x_scale[:, :, :10] = x_scale[:, :, :10] * self.pos_scale
        x_scale[:, :, 10:] = x_scale[:, :, 10:] * self.heading_scale

        h = [x]
        PI = 3.1415927410125732
        for i in range(self.L):
            x_sin = torch.sin(2**i * PI * x_scale)
            x_cos = torch.cos(2**i * PI * x_scale)

            x_sin[:, :, :10], x_sin[:, :, 10:] = x_sin[:, :, :10] / self.pos_scale, x_sin[:, :, 10:] / self.heading_scale
            x_cos[:, :, :10], x_cos[:, :, 10:] = x_cos[:, :, :10] / self.pos_scale, x_cos[:, :, 10:] / self.heading_scale

            h.append(x_sin)
            h.append(x_cos)

        # return torch.cat(h, dim=-1) / self.scale
        return torch.cat(h, dim=-1)


class PredictionsHeads(nn.Module):
    """
    Prediction layer with two output heads, one modeling mean and another one modeling std.
    Also prediction cos and sin headings.
    """

    def __init__(self, h_dim, output_dim):
        super().__init__()
        # x, y position
        self.out_net_mean = nn.Linear(in_features=h_dim, out_features=int(output_dim/2), bias=True)
        self.out_net_std = nn.Linear(in_features=h_dim, out_features=int(output_dim/2), bias=True)
        self.elu = torch.nn.ELU()

        # cos and sin heading
        self.out_net_cos_sin_heading = nn.Linear(in_features=h_dim, out_features=int(output_dim/2), bias=True)

    def forward(self, x):

        # shape x: batch_size x m_token x m_state
        out_mean = self.out_net_mean(x)
        out_std = self.elu(self.out_net_std(x)) + 1

        out_cos_sin_heading = self.out_net_cos_sin_heading(x)

        return out_mean, out_std, out_cos_sin_heading


class SafetyMapper(nn.Module):
    def __init__(self, checkpoint_dir, m_tokens, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(SafetyMapper, self).__init__()
        self.net = define_safety_mapping_networks(model='transformer', input_dim=3, output_dim=3, m_tokens=m_tokens)
        self.net = load_pretrained_weights(self.net, checkpoint_dir, device)

    def safety_mapper_in_the_training_loop_forward(self, x_sampled):
        """
        Run forward pass of the safety mapping networks when training the base behavior model.
        Parameters
        ----------
        x_sampled

        Returns
        -------

        """

        # shape x: batch_size x m_token x m_state
        bs, m_token, nc = x_sampled.shape

        for step in range(int(nc/4)):  # x, y, cos heading, sin heading
            x_lat, x_lon = x_sampled[:, :, step], x_sampled[:, :, int(nc/4) + step]
            x_cos_heading, x_sin_heading = x_sampled[:, :, int(2*nc/4) + step], x_sampled[:, :, int(3*nc/4) + step]
            heading = torch.atan2(x_sin_heading, x_cos_heading)
            heading = torch.rad2deg(heading) % 360.
            x_one_moment = torch.stack([x_lat, x_lon, heading], dim=-1) # bs x m_token x 3

            output = self.net(x_one_moment)
            delta_position = output[:, :, :2]
            delta_position = delta_position / 100

            x_sampled[:, :, step] += delta_position[:, :, 0]
            x_sampled[:, :, int(nc/4) + step] += delta_position[:, :, 1]

        return x_sampled

    def forward(self, pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, pred_vid, device):
        pred_lat = torch.tensor(pred_lat, dtype=torch.float32).to(device)
        pred_lon = torch.tensor(pred_lon, dtype=torch.float32).to(device)
        pred_cos_heading = torch.tensor(pred_cos_heading, dtype=torch.float32).to(device)
        pred_sin_heading = torch.tensor(pred_sin_heading, dtype=torch.float32).to(device)
        pred_vid = torch.tensor(pred_vid, dtype=torch.float32).to(device)

        # non-existence vehicle to 0,0
        pred_lat[torch.isnan(pred_vid)] = 0
        pred_lon[torch.isnan(pred_vid)] = 0
        pred_cos_heading[torch.isnan(pred_vid)] = 0
        pred_sin_heading[torch.isnan(pred_vid)] = 0

        # shape x: batch_size x m_token x m_state
        m_token, nc = pred_lat.shape

        delta_position_mask = []
        for step in range(nc):
            x_lat, x_lon, x_cos_heading, x_sin_heading = pred_lat[:, step], pred_lon[:, step], pred_cos_heading[:, step], pred_sin_heading[:, step]
            heading = torch.atan2(x_sin_heading, x_cos_heading)
            heading = torch.rad2deg(heading) % 360.
            x_one_moment = torch.stack([x_lat, x_lon, heading], dim=-1)  # m_token x 4
            x_one_moment = x_one_moment.unsqueeze(dim=0)  # make sure the input has a shape of batch_size x m_token x state

            output = self.net(x_one_moment)
            delta_position = output[:, :, :2]
            delta_position = (delta_position.squeeze(dim=0)) / 100

            pred_lat[:, step] += delta_position[:, 0]
            pred_lon[:, step] += delta_position[:, 1]

            delta_position_mask_tmp = torch.logical_or(torch.abs(delta_position[:, 0]) > 0.01, torch.abs(delta_position[:, 1]) > 0.01)
            delta_position_mask.append(delta_position_mask_tmp.reshape(-1, 1))

        delta_position_mask = torch.cat(delta_position_mask, dim=1)

        return pred_lat.detach().cpu().numpy(), pred_lon.detach().cpu().numpy(), delta_position_mask.detach().cpu().numpy()


class SimpleMLP(nn.Module):
    """
    A two-layer simple MLP (Baseline, which ignores associations between tokens)
    """

    def __init__(self, input_dim, h_dim, m_tokens):
        super().__init__()
        self.linear1 = nn.Linear(in_features=input_dim, out_features=h_dim, bias=True)
        self.linear2 = nn.Linear(in_features=h_dim, out_features=h_dim, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        # shape x: batch_size x m_token x m_state
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return x


class BnMLP(nn.Module):
    """
    A two-layer MLP with BN (Baseline, which ignores associations between tokens)
    """

    def __init__(self, input_dim, h_dim, m_tokens):
        super().__init__()
        self.linear1 = nn.Linear(in_features=input_dim, out_features=h_dim, bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=m_tokens)
        self.linear2 = nn.Linear(in_features=h_dim, out_features=h_dim, bias=True)
        self.bn2 = nn.BatchNorm1d(num_features=m_tokens)
        self.relu = nn.ReLU()

    def forward(self, x):
        # shape x: batch_size x m_token x m_state
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        return x


class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""

    def __init__(self, input_dim, m_tokens, cfg=Config()):
        super().__init__()
        self.in_net = nn.Linear(input_dim, cfg.dim, bias=True)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.m_tokens = m_tokens

    def forward(self, x):
        # shape x: batch_size x m_token x m_state
        h = self.in_net(x)
        for block in self.blocks:
            h = block(h, None)
        return h
