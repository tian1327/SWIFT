# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import copy
import math

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=256, mlp_dim=4096, K=65536, T=1.0, mlp=False):
        """
        dim: feature dimension (default: 256)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.T = T

        # print the parameters of the base encoder, i.e. OpenCLIP
        # for name, param in base_encoder.named_parameters():
        #     print(name, param.shape)


        # free the patch embedding of the base encoder, based on MoCo V3
        # to improve the training stability of ViT,
        # OpenCLIP uses a conv layer to replace the patch embedding
        base_encoder.visual.conv1.weight.requires_grad = False

        ## this is the original code for MoCo V3, for DeiT model
        # base_encoder.patch_embed.proj.weight.requires_grad = False
        # base_encoder.patch_embed.proj.bias.requires_grad = False

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder
        self.encoder_k = copy.deepcopy(base_encoder)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        # if mlp:  # hack: brute-force replacement
        #     dim_mlp = self.encoder_q.fc.weight.shape[1]
        #     self.encoder_q.fc = nn.Sequential(
        #         nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
        #     )
        #     self.encoder_k.fc = nn.Sequential(
        #         nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
        #     )


        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize key encoder with query encoder
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass


    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)


    @torch.no_grad()
    def _momentum_update_key_encoder(self, m):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    # @torch.no_grad()
    # def _batch_shuffle_ddp(self, x):
    #     """
    #     Batch shuffle, for making use of BatchNorm.
    #     *** Only support DistributedDataParallel (DDP) model. ***
    #     """
    #     # gather from all gpus
    #     batch_size_this = x.shape[0]
    #     x_gather = concat_all_gather(x)
    #     batch_size_all = x_gather.shape[0]

    #     num_gpus = batch_size_all // batch_size_this

    #     # random shuffle index
    #     idx_shuffle = torch.randperm(batch_size_all).cuda()

    #     # broadcast to all gpus
    #     torch.distributed.broadcast(idx_shuffle, src=0)

    #     # index for restoring
    #     idx_unshuffle = torch.argsort(idx_shuffle)

    #     # shuffled index for this gpu
    #     gpu_idx = torch.distributed.get_rank()
    #     idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

    #     return x_gather[idx_this], idx_unshuffle

    # @torch.no_grad()
    # def _batch_unshuffle_ddp(self, x, idx_unshuffle):
    #     """
    #     Undo batch shuffle.
    #     *** Only support DistributedDataParallel (DDP) model. ***
    #     """
    #     # gather from all gpus
    #     batch_size_this = x.shape[0]
    #     x_gather = concat_all_gather(x)
    #     batch_size_all = x_gather.shape[0]

    #     num_gpus = batch_size_all // batch_size_this

    #     # restored index for this gpu
    #     gpu_idx = torch.distributed.get_rank()
    #     idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

    #     return x_gather[idx_this]

    def forward(self, im_q, im_k, m):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        # q = self.encoder_q(im_q)  # queries: NxC
        q = self.predictor(self.encoder_q_projector(self.encoder_q.encode_image(im_q)))  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder(m)  # update the key encoder

            # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            # k = self.encoder_k(im_k)  # keys: NxC
            k = self.encoder_k_projector(self.encoder_k.encode_image(im_k))  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach().to(q.device)])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


# utils
# @torch.no_grad()
# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     tensors_gather = [
#         torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
#     ]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

#     output = torch.cat(tensors_gather, dim=0)
#     return output

def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m


# class MoCo_ResNet(MoCo):
#     def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
#         hidden_dim = self.base_encoder.fc.weight.shape[1]
#         del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

#         # projectors
#         self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
#         self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

#         # predictor
#         self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):

        # get the hidden dim of the visual encoder of CLIP
        # hidden_dim = self.encoder_q.visual.proj.weight.shape[1]
        # hidden_dim = self.encoder_q.head.weight.shape[1]
        hidden_dim = 512 # hack solution for CLIP output dimension

        # Do not remove the encoder head, as we are joint training with the classification task using CE loss
        # del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.encoder_q_projector = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.encoder_k_projector = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        # self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)