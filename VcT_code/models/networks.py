import os
from torch_scatter import scatter_mean
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler
import functools
from einops import rearrange
from utils import make_numpy_grid
import models
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d, TransformerCross
from models.gcnlayers import *
from models.kmeans import *

###############################################################################
# Helper Functions
###############################################################################
indice0 = torch.tensor(np.loadtxt("indice0.txt"), dtype=int).cuda(0)
indice1 = torch.tensor(np.loadtxt("indice1.txt"), dtype=int).cuda(0)


def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs // 3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if args.net_G == 'base_resnet18':
        net = ResNet(input_nc=3, output_nc=2, output_sigmoid=False)

    elif args.net_G == 'Reliable_transformer':
        net = Reliable_Transformer(input_nc=3, output_nc=2, resnet_stages_num=4,
                                   with_pos='learned', enc_depth=1, dec_depth=8)

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)


###############################################################################
# main Functions
###############################################################################


class ResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False, True, True])
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False, True, True])
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False, True, True])
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x)  # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4)  # 1/8, in=64, out=128

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8)  # 1/8, in=128, out=256

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8)  # 1/32, in=256, out=512
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)
        else:
            x = x_8
        # output layers
        x = self.conv_pred(x)
        return x


class Reliable_Transformer(ResNet):

    def __init__(self, input_nc, output_nc, with_pos, resnet_stages_num=5, token_trans=True,
                 enc_depth=1, dec_depth=8,
                 dim_head=64, decoder_dim_head=64,
                  if_upsample_2x=True,
                 backbone='resnet18',
                 decoder_softmax=True,
                 with_decoder=True, k_nums=1000, cluster=10):
        super(Reliable_Transformer, self).__init__(input_nc, output_nc, backbone=backbone,
                                                   resnet_stages_num=resnet_stages_num,
                                                   if_upsample_2x=if_upsample_2x,
                                                   )
        self.k = k_nums
        self.cluster_nums = cluster
        # self.tokenizer = tokenizer
        # if not self.tokenizer:
        #     #  if not use tokenzier，then downsample the feature map into a certain size
        #     self.pooling_size = pool_size
        #     self.pool_mode = pool_mode


        self.token_trans = token_trans
        self.with_decoder = with_decoder
        dim = 32
        mlp_dim = 2 * dim

        self.with_pos = with_pos
        if with_pos is 'learned':
            # self.pos_embedding = nn.Parameter(torch.randn(1, self.cluster_nums, 32))
            self.pos_embedding = nn.Parameter(torch.randn(1, self.cluster_nums * 2, 32))
            # self.pos_embedding = nn.Parameter(torch.randn(1, 1, 32))
        # decoder_pos_size = 256 // 4
        # self.with_decoder_pos = with_decoder_pos
        # if self.with_decoder_pos == 'learned':
        #     self.pos_embedding_decoder = nn.Parameter(torch.randn(1, 32,
        #                                                           decoder_pos_size,
        #                                                           decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformerCross = TransformerCross(dim=dim, depth=self.enc_depth, heads=8,
                                       mlp_dim=mlp_dim, dropout=0, softmax=True)

        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                                                      heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim,
                                                      dropout=0, softmax=decoder_softmax)
        self.gc1 = GraphConvolution(in_features=32, out_features=1)

    def _forward_tokens(self, x, index):
        b, c, h, w = x.shape

        x = x.reshape(b, c, -1)  # x1/2:  b  c   hw

        select_k_x = torch.gather(x, 2, index.repeat(1, c, 1))  # torch.Size([8, c, k])
        # tokens = select_k_x.mean(2, keepdim=True).transpose(1, 2)  # torch.Size([8, 1, 32])
        tokens = select_k_x.transpose(1, 2)

        return tokens

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode == 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def forwardCorss(self, x, m):
        # if self.with_pos:
        #     x = x + self.pos_embedding
        x = self.transformerCross(x, m)
        return x

    def _forward_transformer(self, x):
        if self.with_pos:
            x = x + self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        # if self.with_decoder_pos == 'fix':
        #     x = x + self.pos_embedding_decoder
        # elif self.with_decoder_pos == 'learned':
        #     x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h, w, b, l, c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def kmeansToken(self, x, num_clusters):
        cluster_ids_x, _ = kmeans(X=x.detach(), num_clusters=num_clusters, distance='euclidean')
        c = scatter_mean(x, cluster_ids_x.squeeze(), dim=1, dim_size=num_clusters)
        return c

    def knngraph(self, a):
        b, c, h, w = a.shape
        n = h * w
        a = a.permute(0, 2, 3, 1).reshape(-1, c)
        vals = torch.bmm(a[indice0[0]].unsqueeze(1), a[indice0[1]].unsqueeze(2))
        vals = vals.reshape(-1)

        d = torch.sparse.FloatTensor(indice1, vals, torch.Size([b * n, n])).to_dense()
        d = d.reshape(b, n, n)

        return d

    def forward(self, x1, x2):
        # forward backbone resnet
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        b, c, h, w = x1.shape
        x3 = torch.abs(x1 - x2)  # b  c  h  w
        x4 = x3.reshape(b, c, -1).transpose(1, 2)  # b  hw  c
        A = self.knngraph(x3)  # b hw hw
        A = normalize_adj(A)
        F = self.gc1(x4, A)  # b hw 1
        F = F.transpose(1, 2)
        _, indices = F.topk(k=self.k, dim=2, largest=False)  # indices:(8,1,k)

        token1 = self._forward_tokens(x1, indices)
        token2 = self._forward_tokens(x2, indices)  # (b,k,c)

        token1 = self.kmeansToken(token1, self.cluster_nums)
        token2 = self.kmeansToken(token2, self.cluster_nums)


        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder

        token1_ = self.forwardCorss(token1, token2)
        token2_ = self.forwardCorss(token2, token1)

        x1 = self._forward_transformer_decoder(x1, token1_)
        x2 = self._forward_transformer_decoder(x2, token2_)

        # if self.with_decoder:
        #     x1 = self._forward_transformer_decoder(x1, token1_)
        #     x2 = self._forward_transformer_decoder(x2, token2_)
        # else:
        #     x1 = self._forward_simple_decoder(x1, token1)
        #     x2 = self._forward_simple_decoder(x2, token2)
        # feature differencing
        x = torch.abs(x1 - x2)
        # if not self.if_upsample_2x:
        #     x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        # forward small cnn
        x = self.classifier(x)
        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x
