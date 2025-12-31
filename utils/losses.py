import torch
import torch.nn as nn
import torch
from utils.extras import aves_hard_classes_set
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import MinMaxScaler



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, fea_u_w, fea_u_s, logit_scale, labels, confidence_mask):
        """Compute loss for model.
        This is a modified version of the original SupConLoss.

        https://github.com/HobbitLong/SupContrast/blob/master/losses.py

        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            fea_u_w: features of the weakly augmented samples of shape [bsz, dim].
            fea_u_s: features of the strongly augmented samples of shape [bsz, dim].
            logit_scale: logit scale for scalig logits for calculating contrastive loss.
            labels: pseudo_labels from the weakly augmenetd samples of shape [bsz].
            confidence_mask: confidence mask of shape [bsz], 1 for confident samples, 0 for unconfident samples.

            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        # convert logit_scale to temperature
        self.temperature = 1.0 / logit_scale.exp()

        # filter fea_u_w and fea_u_s based on confidence_mask
        fea_u_w = fea_u_w[confidence_mask==1]
        fea_u_s = fea_u_s[confidence_mask==1]
        labels = labels[confidence_mask==1]

        if len(fea_u_w) == 0:
            return torch.tensor(0.0).to(fea_u_w.device)

        # Normalize along feature dimension
        fea_u_w = F.normalize(fea_u_w, dim=-1, p=2)
        fea_u_s = F.normalize(fea_u_s, dim=-1, p=2)

        device = fea_u_w.device

        # combine fea_u_w and fea_u_s of shape [bsz, dim] into [bsz, 2, dim]
        features = torch.stack([fea_u_w, fea_u_s], dim=1)
        features = features.contiguous()
        # print('features.shape=', features.shape) # bsz, 2, 512

        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
        # print('mask.shape=', mask.shape) # 256, 256

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # print('contrast_feature.shape=', contrast_feature.shape) # 512, 512
        # print('contrast_count=', contrast_count) # 2
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1

        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count

        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # print('anchor_feature.shape=', anchor_feature.shape) # 512, 512
        # print('anchor_count=', anchor_count) # 2

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # print('anchor_dot_contrast.shape=', anchor_dot_contrast.shape) # 512, 512


        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # print('logits.shape=', logits.shape) # 512, 512
        # print('logits_max', logits_max) # 10, 10, ..., 10

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # print('mask.shape=', mask.shape) # 256, 256

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs_updated = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs_updated

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss, mask_pos_pairs



def set_loss(args):
    if args.loss_name == 'CE':
        loss = nn.CrossEntropyLoss()
    elif args.loss_name == 'WeightedCE':
        loss = WeightedCELoss(fewshot_weight=args.fewshot_weight)
    elif args.loss_name == 'Focal':
        loss = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    elif args.loss_name == 'BalancedSoftmax':
        loss = BalancedSoftmaxLoss(cls_num_list=args.cls_num_list)
    elif args.loss_name == 'LCA':
        # load the LCA matrix
        tree = None
        lca_matrix_distance = np.load('hierarchy/semi-aves_lca_matrix.npy')
        lca_matrix = process_lca_matrix(lca_matrix_distance)
        loss = lca_alignment_loss(tree, lca_matrix.cuda())
    else:
        raise NotImplementedError(f'Loss {args.loss_name} not implemented.')

    args.loss = loss

    return loss


def process_lca_matrix(lca_matrix_raw, tree_prefix=None, temperature=25.0):
    if lca_matrix_raw is None:
        return None
    # if tree_prefix!='WordNet':
    #     result_matrix=np.max(lca_matrix_raw)-lca_matrix_raw # convert from depth to distance
    # else:
    #     result_matrix=lca_matrix_raw
    result_matrix=lca_matrix_raw
    result_matrix=result_matrix**temperature

    scaler = MinMaxScaler()
    result_matrix=scaler.fit_transform(result_matrix)
    # print(result_matrix)
    return torch.from_numpy(result_matrix)


class lca_alignment_loss(nn.Module):
    def __init__(self, tree, lca_matrix, alignment_mode=2):
        super().__init__()
        self.lca_matrix=lca_matrix # this is already minmax normalized lca matrix
        self.tree=tree
        self.alignment_mode=alignment_mode
        self.reverse_matrix=1-self.lca_matrix # take reverse of distance such that groundtruth index has the highest value of 1

    def forward(self,logits,targets,lambda_weight=0.03):
        reverse_matrix=self.reverse_matrix

        # Ensure distance matrix is float32
        distance_matrix = self.lca_matrix.float()

        # Compute the predicted probabilities
        probs = F.softmax(logits, dim=1)

        # One-hot encode the targets
        one_hot_targets = F.one_hot(targets, num_classes=logits.size(1)).float()

        # Compute the standard cross-entropy loss
        standard_loss = -torch.sum(one_hot_targets * torch.log(probs + 1e-9), dim=1)


        # Compute the alignment soft loss
        if self.alignment_mode==0: # not using alignment soft loss
            total_loss=standard_loss
        else:
            if self.alignment_mode==1: # BCE-form alignment soft loss
                criterion=nn.BCEWithLogitsLoss(reduction='none')
                alignment_loss = torch.mean(criterion(logits, reverse_matrix[targets]),dim=1)

            elif self.alignment_mode==2: # CE-form alignment soft loss
                alignment_loss=-torch.mean(reverse_matrix[targets] * torch.log(probs + 1e-9), dim=1)

            assert lambda_weight<=1
            assert lambda_weight>=0

            total_loss = lambda_weight* standard_loss + alignment_loss
            # total_loss = 0.5* standard_loss + (1.0 - 0.5)*alignment_loss
            # total_loss = lambda_weight* standard_loss + alignment_loss

        # Return the mean loss over the batch
        return torch.mean(total_loss)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        # Ensure numerical stability with epsilon
        ce_loss = torch.clamp(ce_loss, min=1e-8)

        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * ((1-pt)**self.gamma) * ce_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class WeightedLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # assign higher weights to hard classes in aves_hard_classes_set
        weights = torch.ones(inputs.shape[0])
        for i in range(inputs.shape[0]):
            if str(targets[i].item()) in aves_hard_classes_set:
                weights[i] = 3
                # print('hard class:', targets[i])
                # stop
        weights = weights.to(inputs.device)
        W_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        W_loss = W_loss * weights

        if self.reduction == 'mean':
            return torch.mean(W_loss)
        elif self.reduction == 'sum':
            return torch.sum(W_loss)
        else:
            return W_loss


class WeightedCELoss(nn.Module):
    def __init__(self, fewshot_weight=1.0, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.fewshot_weight = fewshot_weight

    # source in the text file uses 1 as fewshot data, 0 as retrived data
    def forward(self, inputs, targets, source):
        # print('inputs.shape:', inputs.shape)
        # print('targets.shape:', targets.shape)

        # fewshot data has higher weight, retrived data has weight 1.0
        weights = source * self.fewshot_weight + (1.0 - source)
        W_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        weights = weights.to(inputs.device)
        W_loss = W_loss * weights

        if self.reduction == 'mean':
            return torch.mean(W_loss)
        elif self.reduction == 'sum':
            return torch.sum(W_loss)
        else:
            return W_loss


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, cls_num_list):
        super().__init__()
        cls_num_list = torch.from_numpy(np.array(cls_num_list)).float().cuda()
        cls_prior = cls_num_list / sum(cls_num_list)
        self.log_prior = torch.log(cls_prior).unsqueeze(0)
        # self.min_prob = 1e-9
        # print(f'Use BalancedSoftmaxLoss, class_prior: {cls_prior}')

    def forward(self, logits, labels):
        adjusted_logits = logits + self.log_prior
        label_loss = F.cross_entropy(adjusted_logits, labels)

        return label_loss