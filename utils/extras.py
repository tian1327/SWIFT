import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, RandomHorizontalFlip, RandomCrop
from PIL import Image, ImageFilter, ImageOps

import open_clip, clip
import argparse
import json
import os
from torchvision import transforms
from .randaugment import RandAugmentMC
import math
import random
from torchvision import transforms
import torchvision.transforms.v2 as v2

# import torchvision.transforms.functional as tf

# linear probing 30 hard classes
# aves_hard_classes = ['95', '60', '90', '94', '143', '169', '14', '30', '65', '76', '87', '132', '11', '80', '47', '78', '106', '123', '135', '171', '2', '29', '58', '62', '91', '100', '105', '139', '146', '44']

# finetuning 30 hard classes, truth on test set
aves_hard_classes = ['90', '87', '42', '78', '146', '95', '139', '11', '30', '51', '58', '63', '76', '80', '94', '132', '143', '169', '14', '39', '62', '65', '123', '149', '71', '105', '131', '0', '2', '38']

aves_hard_classes_set = set(aves_hard_classes)



class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _convert_image_to_rgb(image):
    return image.convert("RGB")


class TransformFixMatch(object):
    def __init__(self, n_px, is_openclip=True):
        self.weak = transforms.Compose([
            RandomResizedCrop(n_px,
                            #   scale=(0.9, 1.0),
                            #   ratio=(0.75, 1.3333),
                            #   interpolation=Image.BICUBIC
                              ),
            RandomHorizontalFlip(),
            # _convert_image_to_rgb
            ])

        self.strong = transforms.Compose([
            RandomResizedCrop(n_px,
                            #   scale=(0.9, 1.0),
                            #   ratio=(0.75, 1.3333),
                            #   interpolation=Image.BICUBIC
                              ),
            RandomHorizontalFlip(),
            # _convert_image_to_rgb,
            RandAugmentMC(n=2, m=10) # add RandAugmentMC here for strong augmentation !!!
            # v2.RandAugment(num_ops=2, magnitude=9),
            ])

        if is_openclip:
            self.normalize = transforms.Compose([
                transforms.ToTensor(),
                # CLIP mean and std
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))])
        else:
            self.normalize = transforms.Compose([
                transforms.ToTensor(),
                # ImageNet mean and std
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)



# class TransformFixMatch:
#     def __init__(self, n_px, is_openclip=True):
#         self.resize = v2.Resize((n_px, n_px))  # Resize only once at the start

#         self.weak = v2.Compose([
#             v2.RandomHorizontalFlip(),
#             # v2.ColorJitter(0.4, 0.4, 0.4, 0.1),  # Faster than multiple separate jitter ops
#         ])

#         self.strong = v2.Compose([
#             v2.RandomHorizontalFlip(),
#             v2.RandAugment(num_ops=2, magnitude=9),
#         ])

#         self.to_tensor = v2.Compose([
#             v2.ToTensor(),
#             v2.ConvertImageDtype(torch.float32),  # Efficient tensor conversion
#         ])

#         if is_openclip:
#             self.normalize = v2.Normalize(
#                 mean=(0.48145466, 0.4578275, 0.40821073),
#                 std=(0.26862954, 0.26130258, 0.27577711))
#         else:
#             self.normalize = v2.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225])

#     def __call__(self, x):
#         x = self.resize(x)  # Resize once
#         weak = self.weak(x)
#         strong = self.strong(x)

#         weak = self.to_tensor(weak)
#         strong = self.to_tensor(strong)

#         return self.normalize(weak), self.normalize(strong)




class MoCoTransform(object):
    def __init__(self, n_px, is_openclip=True):

        if is_openclip:
            self.normalize = transforms.Compose([
                # transforms.ToTensor(),
                # CLIP mean and std
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))])
        else:
            self.normalize = transforms.Compose([
                # transforms.ToTensor(),
                # ImageNet mean and std
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

        # from MoCo v3 stronger transformations: https://github.com/facebookresearch/moco-v3/blob/c349e6e24f40d3fedb22d973f92defa4cedf37a7/main_moco.py
        # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
        augmentation1 = [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize
        ]
        self.transform1 = transforms.Compose(augmentation1)

        augmentation2 = [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            transforms.RandomApply([Solarize()], p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize
        ]
        self.transform2 = transforms.Compose(augmentation2)

    def __call__(self, x):
        q_images = self.transform1(x)
        k_images = self.transform2(x)
        return [q_images, k_images]


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)



def transform(n_px , mode='train'):
    normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    if mode == 'train':
        return Compose([
            RandomResizedCrop(n_px, scale=(0.9, 1.0), ratio=(0.75, 1.3333),
                              interpolation=Image.BICUBIC),
            # CenterCrop(n_px), # change to Center Crop
            RandomHorizontalFlip(),
            _convert_image_to_rgb,
            ToTensor(),
            normalize
        ])



OpenCLIP_model_dic_LAION400M = {
    'ViT-B/32': ('ViT-B-32-quickgelu', 'laion400m_e32'),
    'ViT-B/16': ('ViT-B-16', 'laion400m_e32'),
    'ViT-L/14': ('ViT-L-14', 'laion400m_e32')
}

OpenCLIP_model_dic_LAION2B = {
    'ViT-B/32': ('ViT-B-32', 'laion2b_e16'),
}

OPENCLIP_MODEL_DIC = {
    'laion400m': {
        'vitb32': ('laion400m_e32','ViT-B-32-quickgelu'),
        'vitb16': ('laion400m_e32','ViT-B-16'),
        'vitl14': ('laion400m_e32','ViT-L-14'),
    },
    'openai': {
        'vitb32': ('openai','ViT-B-32-quickgelu'),
        'vitb16': ('openai','ViT-B-16'),
        'vitl14': ('openai','ViT-L-14')
    },
    'laion2b': {
        'vitb32': ('laion2b_s34b_b79k','ViT-B-32'),
        'vitb16': ('laion2b_s34b_b88k','ViT-B-16'),
        'vitl14': ('laion2b_s32b_b82k','ViT-L-14')
    },
}

CLIP_MODEL_DIC ={
    'vitb32': 'ViT-B/32',
    'vitb16': 'ViT-B/16',
    'resnet50': 'RN50',
}



def get_engine(model_cfg, device = 'cuda', mode='val'):

    cfgs = model_cfg.split('_')
    arch = cfgs[0]
    model_name = cfgs[1]
    pretraining_dataset = cfgs[2] if len(cfgs) == 3 else None

    if model_name == 'clip':
        arch = CLIP_MODEL_DIC[arch]
        model, preprocess = clip.load(arch, device)
        tokenizer = clip.tokenize
        # get the train preprocess for CLIP
        # train_preprocess = transform(224, mode='train')
        train_preprocess = preprocess

        # load Oindrila's checkpoint
        if pretraining_dataset is not None:
            if cfgs[0] == 'vitb32':
                arc = 'b32'
            elif cfgs[0] == 'vitb16':
                arc = 'b16'
            else:
                raise NotImplementedError

            if pretraining_dataset == 'cub':
                ckpt = torch.load(f'/scratch/group/real-fs/model_ckpts/AdaptCLIPZS/CUB_{arc}.pth', map_location=device)
            elif pretraining_dataset == 'inat':
                ckpt = torch.load(f'/scratch/group/real-fs/model_ckpts/AdaptCLIPZS/INaturalist21_{arc}.pth', map_location=device)
            elif pretraining_dataset == 'nabirds':
                ckpt = torch.load(f'/scratch/group/real-fs/model_ckpts/AdaptCLIPZS/NABirds_{arc}.pth', map_location=device)
            else:
                raise NotImplementedError

            # print(ckpt.keys())
            model.load_state_dict(ckpt, strict=True)

    elif model_name == 'openclip':
        corpus_config, model_arch = OPENCLIP_MODEL_DIC[pretraining_dataset][arch]
        model, train_preprocess, preprocess = open_clip.create_model_and_transforms(model_arch, pretrained=corpus_config)
        # print('train_preprocess:', train_preprocess)
        tokenizer = open_clip.get_tokenizer(model_arch)

    else:
        raise NotImplementedError

    # not using mixed precision
    model.float()
    model.to(device)

    if mode == 'train':
        return model, train_preprocess, preprocess, tokenizer
    elif mode == 'val':
        return model, preprocess, tokenizer
    else:
        raise NotImplementedError


def get_worstk_class(score, confusion_matrix, N=30, fname=None):

    confusing_pairs = {}

    # id to scname mapping
    with open('data/semi-aves/id_scname_dict.json', 'r') as f:
        id_scname_dict = json.load(f)

    # get top-k confusing classes of the confusion matrix for each class
    topk = 20
    topk_classes = {}
    for classid, row in enumerate(confusion_matrix):
        # each item in the dict is a tuple (str(classid), str(count))
        topk_classes[str(classid)] = sorted([(i, row[i]) for i in range(len(row))], key=lambda x: x[1], reverse=True)[:topk]

    for k, v in topk_classes.items():
        info = dict()
        for classid, count in v:
            info[classid] = (id_scname_dict[str(classid)], str(count))
        topk_classes[k] = info

    # get the target dict
    target_dict = {}
    for classid, row in enumerate(confusion_matrix):
        target_dict[str(classid)] = dict()
        target_dict[str(classid)]['sname'] = id_scname_dict[str(classid)][0]
        target_dict[str(classid)]['cname'] = id_scname_dict[str(classid)][1]
        target_dict[str(classid)]['zs_acc'] = score['per_class_recall'][classid]*100
        target_dict[str(classid)]['topk_confusion'] = topk_classes[str(classid)]
        # target_dict[str(classid)]['freq'] = sorted_classid_freq_dict_downstream[str(classid)]['count']
        target_dict[str(classid)]['freq'] = 10

    # sort the target dict by the increasing zs_acc
    target_dict = sorted(target_dict.items(), key=lambda x: x[1]['zs_acc'])

    C=15
    for k, v in target_dict[:N]: # here worst N classes
        # print(f"\n{k}, {v['cname']}, {round(v['zs_acc'], 1)}, {v['freq']}")
        # print(f"\n{k}, {v['cname']}, {round(v['zs_acc'], 1)}")
        classid = k
        cname = v['cname']
        if classid not in confusing_pairs:
            confusing_pairs[classid] = []
        # print('its top k confusions:')
        ct = 0
        max_conf_count = 0
        for kk, vv in v['topk_confusion'].items():
            # print(f"{kk}, {vv[0][1]}, {vv[1]}")
            conf_id = kk
            conf_cname = vv[0][1]
            conf_count = int(vv[1])
            # collecting the confusing pairs, excluding the self-confusion, including the tiers
            if int(conf_id) == int(classid):
                pass
            elif conf_count >= max_conf_count:
                max_conf_count = conf_count
                confusing_pairs[classid].append(conf_id) # this is a list
            else:
                pass

            ct += 1
            if ct == C:
                break

    # print the confusing pairs
    # print()
    # for k, v in confusing_pairs.items():
    #     print(f"{k}: {v}")

    return confusing_pairs


# single-match precision based on classid
def cal_single_precision(true, obtained):

    obtained_keys = set(obtained.keys())
    true_keys = set(true.keys())

    # count how many true keys are in the obtained keys
    intersection = obtained_keys.intersection(true_keys)
    # print(f"intersection: {intersection}")
    # print(f'matching ct: {len(intersection)}/{len(obtained_keys)}')

    precision = len(intersection) / len(obtained_keys)
    # print(f"precision: {round(precision, 4)}")
    return precision


def cal_pair_precision(true, obtained):

    # calculate the candidate pair precision
    obtained_keys = set(obtained.keys())
    true_keys = set(true.keys())

    # count how many true keys are in the obtained keys
    intersection = obtained_keys.intersection(true_keys)
    if len(intersection) == 0:
        return 0, 0

    pair_match_ct = 0
    for k in intersection:
        candidates = set(obtained[k])
        target = set(true[k])

        if len(candidates.intersection(target)) > 0:
            pair_match_ct += 1
            # print(f"{k}: {candidates.intersection(target)}")

    rel_precision = pair_match_ct / len(intersection)
    # print(f"rel candidate pair precision: {round(rel_precision, 4)}")

    abs_precision = pair_match_ct / len(obtained_keys)
    # print(f"abs candidate pair precision: {round(abs_precision, 4)}")

    return rel_precision, abs_precision

def cal_hard_avg_acc(score):
    acc = 0
    # calculate the avg recall for the class in the aves_hard_classes
    for classid in aves_hard_classes_set:
        acc += score['per_class_recall'][int(classid)]
    acc /= len(aves_hard_classes_set)

    return acc

def cal_easy_avg_acc(score):
    acc = 0.0
    # calculate the avg recall for the class not in the aves_hard_classes
    ct = 0
    for classid in score['per_class_recall'].keys():
        if str(classid) not in aves_hard_classes_set:
            ct += 1
            acc += score['per_class_recall'][int(classid)]
    acc /= (len(score['per_class_recall']) - len(aves_hard_classes_set))

    # if acc > 1.0:
    #     print(score['per_class_recall'])
    #     print('acc > 1.0')
    #     print('ct:', ct)
    #     print(acc)
    #     print(len(score['per_class_recall']) - len(aves_hard_classes_set))
    #     print(len(score['per_class_recall']))
    #     print(len(aves_hard_classes_set))
    #     exit()

    return acc

def get_class_num_list(test_file):

    with open(test_file, 'r') as f:
        lines = f.readlines()

    class_id_set =set()
    for line in lines:
        entries = line.strip('\n').split()
        # filename = entries[0]
        classid = int(entries[1])
        # source = int(entries[2])
        class_id_set.add(classid)
    # print('len(class_id_set):', len(class_id_set))

    cls_num_list = [0] * len(class_id_set)
    for line in lines:
        entries = line.strip('\n').split()
        classid = int(entries[1])
        cls_num_list[classid] += 1

    return cls_num_list
