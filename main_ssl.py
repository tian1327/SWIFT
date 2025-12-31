import os
import torch
from utils.models import set_model, set_classifier, MyLinear, save_test_scores, save_best_model, DinoVisionTransformer#, save_head_weights
import time
import numpy as np
from utils.parser import parse_args
from utils.logger import set_logger
from testing import validate, load_model
from testing import calculate_scores
from utils.datasets.dataset_utils import NUM_CLASSES_DICT
from utils.prompt import set_prompt
import copy
from utils.losses import set_loss
import torch.nn.functional as F
import cv2
from utils.training import set_training_seed, train_probing, run_zeroshot, train_CMLP, \
    train_dataset_cls, train_ce, train_cutmix, train_flyp, train_ce_mixed, train_fixmatch, train_debiasPL, \
    train_ce_multitask, train_mixup, train_mixup_fs, train_cutmix_fs, train_resizemix, \
    train_saliencymix2, train_attentivemix2, train_CMO, train_supervised_contrastive, train_balanced_contrastive, \
    train_probing2, train_fixmatch2
from utils.dataloader import get_unlabeled_dataloader, extract_train_dataloader, extract_dataloader, set_dataloaders, set_text_dataloader, get_retrieve_fewshot_dataloader
from utils.optimizers import set_optimizer, set_params
# from gem import create_gem_model
# import pickle

from torchvision.models import resnet50
from torchvision.models.vision_transformer import vit_b_32
from torchvision.models import ViT_B_32_Weights

from collections import OrderedDict
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from utils.datasets.dataset_utils import load_dataset
from testing import extract_confidence



# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

dinvo_fts = {
    "vits14": 384,
    "vitb14": 768,
    "vitl14": 1024,
    "vitg14": 1536,
}


def run_tau_normalization(args, best_head, best_model, val_loader, test_loader, logit_scale, logger):

    best_tau_head = copy.deepcopy(best_head)
    best_tau = 0.0
    best_tau_val_acc = 0.0
    best_tau_test_acc = 0.0
    if args.tau_norm:
        logger.info(f"Check Tau Normalization ......")
        tau_list = np.arange(0.0, 2.2, 0.2).tolist()
        for tau in tau_list:
            tau_head = copy.deepcopy(best_head)
            tau_head.linear.weight.data /= torch.pow(tau_head.linear.weight.data.norm(dim=-1, keepdim=True), tau)
            # does not affect FLYP because head is already normalized, thus the norm=1

            # check on val set
            tau_val_acc, _, _ = validate(args,data_loader=val_loader,
                                        model=best_model, logger=logger,
                                        loss=args.loss, logit_scale=logit_scale,
                                        classifier_head=tau_head,
                                        dataset=args.dataset,
                                        output_dir=args.output_dir, device=args.device,
                                        )
            # check on test set
            tau_test_acc, _, tau_test_confusion_matrix = validate(args,data_loader=test_loader,
                                            model=best_model, logger=logger,
                                            loss=args.loss, logit_scale=logit_scale,
                                            show_confusion_matrix=True,
                                            classifier_head=tau_head,
                                            dataset=args.dataset,
                                            output_dir=args.output_dir, device=args.device,
                                            )
            logger.info(f"Tau: {round(tau,2)}, Val Acc: {round(tau_val_acc, 3)}, Test Acc: {round(tau_test_acc, 3)}")
            if tau_val_acc > best_tau_val_acc:
                best_tau = tau
                best_tau_val_acc = tau_val_acc
                best_tau_test_acc = tau_test_acc
                best_tau_head = copy.deepcopy(tau_head)
                best_tau_scores = calculate_scores(tau_test_confusion_matrix)
                best_tau_confusion_matrix = copy.deepcopy(tau_test_confusion_matrix)

        logger.info(f"+++++ Best Tau: {round(best_tau,1)}, Val Acc: {round(best_tau_val_acc, 3)}, Test Acc: {round(best_tau_test_acc, 3)}")
        # save_test_scores(best_tau_scores, best_tau_confusion_matrix, args.output_dir, 'best_tau_test')
        # save_head_weights(best_tau_head, output_dir, 'best_tau')

    return best_tau_head, best_tau, best_tau_test_acc


def ensemble_model(best_model, zeroshot_model, alpha):
    """Ensemble the best_model and zeroshot_model"""

    wsft_model = copy.deepcopy(best_model)
    # Load models
    zeroshot = zeroshot_model
    finetuned = best_model
    theta_0 = zeroshot.state_dict()
    theta_1 = finetuned.state_dict()

    # make sure checkpoints are compatible
    assert set(theta_0.keys()) == set(theta_1.keys())

    # interpolate between checkpoints with mixing coefficient alpha
    theta = {
        key: (1.0-alpha) * theta_0[key] + alpha * theta_1[key]
        for key in theta_0.keys()
    }

    # update the model acccording to the new weights
    wsft_model.load_state_dict(theta)

    return wsft_model

def run_wsft(args, best_model, best_head, test_loader, zeroshot_model, zeroshot_weights, logit_scale, logger, alpha=0.5):

    learned_head_weights = best_head.linear.weight.data.to(args.device)
    wsft_head_weights = alpha * learned_head_weights + (1.0 - alpha) * zeroshot_weights
    wsft_head = MyLinear(weights=wsft_head_weights)
    wsft_head.to(args.device)
    logger.info(f'WiSE-FT classifier done. alpha: {alpha}')
    if args.freeze_visual:
        wsft_model = best_model
    else:
        # ensemble the best_model and zeroshot_model
        wsft_model = ensemble_model(best_model, zeroshot_model, alpha)
        logger.info(f'WiSE-FT model done. alpha: {alpha}')

    wsft_test_acc, _, _ = validate(args,data_loader=test_loader,
                                                        model=wsft_model,
                                                        classifier_head=wsft_head, # here use the wsft_head
                                                        logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        show_confusion_matrix=False,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        )
    logger.info(f"+++++ WiSE-FT Test Acc: {round(wsft_test_acc, 3)}")
    # wsft_test_scores = calculate_scores(wsft_test_confusion_matrix)
    # save_test_scores(wsft_test_scores, wsft_test_confusion_matrix, args.output_dir, 'wsft_test')
    # save_head_weights(wsft_head, output_dir, 'wsft')

    return wsft_model, wsft_head, wsft_test_acc

def run_wsft_alpha(args, best_model, best_head, val_loader, test_loader, zeroshot_model, zeroshot_head, logit_scale, logger, step=0.1):
    logger.info(f"Checking WSFT ......")

    ensemble_val_acc = []
    ensemble_test_acc = []
    learned_head_weights = best_head.linear.weight.data.to(args.device)
    zeroshot_weights = zeroshot_head.linear.weight.data.to(args.device)
    best_alpha = 0.0
    best_wsft_test_acc = 0.0
    best_wsft_val_acc = 0.0
    best_wsft_head = best_head
    best_wsft_model = best_model
    # for alpha in np.arange(0.0, 1.0+step, step):
    for alpha in [0.5]:

        wsft_head_weights = alpha * learned_head_weights + (1.0 - alpha) * zeroshot_weights
        wsft_head = MyLinear(weights=wsft_head_weights)
        wsft_head.to(args.device)

        # wsft_head = best_head # use the best_head, do not ensemble the head
        # wsft_head = zeroshot_head # use the zeroshot_head, do not ensemble the head

        if args.freeze_visual:
            wsft_model = best_model
        else:
            # ensemble the best_model and zeroshot_model
            wsft_model = ensemble_model(best_model, zeroshot_model, alpha)

        wsft_val_acc, _, _ = validate(args,data_loader=val_loader,
                                        model=wsft_model,
                                        classifier_head=wsft_head, # here use the wsft_head
                                        logger=logger,
                                        loss=args.loss, logit_scale=logit_scale,
                                        show_confusion_matrix=False,
                                        dataset=args.dataset,
                                        output_dir=args.output_dir, device=args.device,
                                        )

        wsft_test_acc, _, _ = validate(args,data_loader=test_loader,
                                        model=wsft_model,
                                        classifier_head=wsft_head, # here use the wsft_head
                                        logger=logger,
                                        loss=args.loss, logit_scale=logit_scale,
                                        show_confusion_matrix=False,
                                        dataset=args.dataset,
                                        output_dir=args.output_dir, device=args.device,
                                        )
        ensemble_val_acc.append(wsft_val_acc)
        ensemble_test_acc.append(wsft_test_acc)
        logger.info(f"Alpha:{round(alpha, 3)}, Val Acc: {round(wsft_val_acc, 3)}, Test Acc: {round(wsft_test_acc, 3)}")
        if wsft_val_acc > best_wsft_val_acc:
            best_wsft_val_acc = wsft_val_acc
            best_wsft_test_acc = wsft_test_acc
            best_alpha = alpha
            best_wsft_head = copy.deepcopy(wsft_head)
            best_wsft_model = copy.deepcopy(wsft_model)

    logger.info(f"+++++ Best Alpha: {round(best_alpha, 2)}, Val Acc: {round(best_wsft_val_acc, 3)}, Test Acc: {round(best_wsft_test_acc, 3)}")
    # print(f'ensemble_val_acc', ensemble_val_acc)
    # print(f'ensemble_test_acc', ensemble_test_acc)

    return best_wsft_model, best_wsft_head, best_wsft_test_acc


def run_stage1_finetuning2(args, logger, model, classifier_head, data_transforms):

    results=dict()
    results['mask'] = 0.0
    results['impurity'] = 0.0

    # dataloaders
    train_dataset = load_dataset(dataset_root=args.dataset_root,
                                split=args.train_split,
                                preprocess=data_transforms['train'],
                                tokenized_text_prompts=None,
                                )

    val_dataset = load_dataset(dataset_root=args.dataset_root,
                                split=args.val_split,
                                preprocess=data_transforms['val'],
                                tokenized_text_prompts=None,
                                )

    test_dataset = load_dataset(dataset_root=args.dataset_root,
                                split=args.test_split,
                                preprocess=data_transforms['test'],
                                tokenized_text_prompts=None,
                                )

    train_loader = DataLoader(train_dataset, batch_size=args.bsz, pin_memory=True,
                            shuffle=True, drop_last=True, num_workers=args.num_workers)

    val_loader = DataLoader(val_dataset, batch_size=128, drop_last=False, pin_memory=True,
                            shuffle=False, num_workers=args.num_workers)

    test_loader = DataLoader(test_dataset, batch_size=128, drop_last=False, pin_memory=True,
                            shuffle=False, num_workers=args.num_workers)

    if args.method == 'fixmatch' or args.method == 'debiasPL':

        u_train_dataloader = get_unlabeled_dataloader(args, args.u_train_split, False)
        logger.info(f'len(u_train_dataloader): {len(u_train_dataloader)}')
        train_loader = (train_loader, u_train_dataloader)

        """
        train_loader_fewshot = get_retrieve_fewshot_dataloader(args, args.fewshot_data)
        train_loader_retrieve = get_retrieve_fewshot_dataloader(args, args.retrieval_data)
        u_train_dataloader = get_unlabeled_dataloader(args, args.u_train_split, False)

        logger.info(f'len(train_loader_fewshot): {len(train_loader_fewshot)}')
        logger.info(f'len(train_loader_retrieve): {len(train_loader_retrieve)}')
        logger.info(f'len(u_train_dataloader): {len(u_train_dataloader)}')
        train_loader = (train_loader_fewshot, train_loader_retrieve, u_train_dataloader) # overwrite train_loader
        """

    test_loader_copy = copy.deepcopy(test_loader)

    loss = set_loss(args)

    if args.method == 'probing':
        for param in model.parameters():
            param.requires_grad = False
        for param in classifier_head.parameters():
            param.requires_grad = True
        params = [{'params': classifier_head.parameters(), 'lr': args.lr_classifier}]

        # logit_scale  = torch.tensor([0.0]).to(device=args.device)
        # logit_scale = torch.tensor([4.60517]).to(device=args.device) # 4.60517 = np.log(100) = np.log(1 / 0.01), 0.01 is the temperature

        if args.temp_scheme == "none":
            logit_scale = torch.ones([]) * np.log(1.0 / args.temperature) # ln(1/0.01)=4.6052, ln(1/1)=0.0
        else:
            logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / args.temperature)) # ln(1/0.07)=2.65926, as in CLIP and OpenCLIP
            params.append({'params': [logit_scale], 'lr': args.lr_temp})

        args.logit_scale = logit_scale

    elif args.method == 'finetune' or args.method == 'cutmix' \
        or args.method == 'fixmatch' or args.method == 'debiasPL':
        for param in model.parameters():
            param.requires_grad = True
        for param in classifier_head.parameters():
            param.requires_grad = True
        params = [{'params': model.parameters(), 'lr': args.lr_backbone},
                  {'params': classifier_head.parameters(), 'lr': args.lr_classifier}]

        cfgs = args.model_cfg.split('_')
        if cfgs[0] == 'resnet50' or cfgs[0] == 'dinov2' or (cfgs[0] == 'vitb32' and cfgs[1] == 'imagenet'):
            if args.temp_scheme == "none":
                logit_scale_x = torch.ones([]) * np.log(1.0 / args.temperature) # exp(0.0) = 1.0/1.0, temp=1.0
                logit_scale_r = torch.ones([]) * np.log(1.0 / args.temperature) # set args.temperature=1.0, ln(1.0)= 0.0
                logit_scale_u = torch.ones([]) * np.log(1.0 / args.temperature)

            elif args.temp_scheme == "fewshot":
                logit_scale_x = nn.Parameter(torch.ones([]) * np.log(1.0 / args.temperature))
                logit_scale_r = torch.tensor([0.0]).to(device=args.device)
                logit_scale_u = torch.tensor([0.0]).to(device=args.device)
                params.append({'params': [logit_scale_x], 'lr': args.lr_temp})

            elif args.temp_scheme == "fewshot+retrieved":
                logit_scale_x = nn.Parameter(torch.ones([]) * np.log(1.0 / args.temperature))
                logit_scale_r = nn.Parameter(torch.ones([]) * np.log(1.0 / args.temperature))
                logit_scale_u = torch.tensor([0.0]).to(device=args.device)
                params.append({'params': [logit_scale_x], 'lr': args.lr_temp})
                params.append({'params': [logit_scale_r], 'lr': args.lr_temp})

            elif args.temp_scheme == "fewshot+retrieved+unlabeled":
                logit_scale_x = nn.Parameter(torch.ones([]) * np.log(1.0 / args.temperature))
                logit_scale_r = nn.Parameter(torch.ones([]) * np.log(1.0 / args.temperature))
                logit_scale_u = nn.Parameter(torch.ones([]) * np.log(1.0 / args.temperature))

                params.append({'params': [logit_scale_x], 'lr': args.lr_temp})
                params.append({'params': [logit_scale_r], 'lr': args.lr_temp})
                params.append({'params': [logit_scale_u], 'lr': args.lr_temp})
            else:
                raise NotImplementedError(f'Temperature scheme {args.temp_scheme} not implemented.')

            logit_scale = logit_scale_x
            args.logit_scale = logit_scale_x
            args.logit_scale_r = logit_scale_r
            args.logit_scale_u = logit_scale_u

        else:
            raise NotImplementedError(f"Model {args.model_cfg} not implemented.")

    else:
        raise NotImplementedError(f"Method {args.method} not implemented.")


    optimizer, scheduler, total_iter = set_optimizer(args, params, train_loader)

    args.loss = loss
    args.optimizer = optimizer
    args.scheduler = scheduler

    if args.skip_stage1:
        return -1, None, test_loader_copy

    if args.model_path is not None:
        load_model(args, logger, model, test_loader, classifier_head, False)    

    # check zeroshot acc
    args.pre_extracted = False
    if args.check_zeroshot or args.method == 'zeroshot':
        logger.info(f"Check Zero-shot Acc ......")
        run_zeroshot(args, test_loader, model, logger, loss, logit_scale, classifier_head, False)
    if args.zeroshot_only or args.method == 'zeroshot':
        exit()

    # get the confidence on the test set of different pretrained models
    if args.check_confidence:
        logger.info(f"calculate confidence ......")
        extract_confidence(args, model, classifier_head, test_loader)
        exit()




    reload_model = True if args.model_path else False
    #---------- Training
    if args.method == 'probing':
        best_model, best_head, best_records, \
            best_logit_scale, val_loader, test_loader = train_probing2(args, logger, loss_logger, model, classifier_head, \
                                                                      train_loader, val_loader, test_loader, reload_model)

    elif args.method == 'CMLP': # cross modal linear probing
        best_model, best_head, best_records, \
            best_logit_scale, val_loader, test_loader = train_CMLP(args, logger, loss_logger, model, classifier_head, \
                                                                   preprocess, tokenized_text_prompts, \
                                                                   train_loader, val_loader, test_loader, False, text_dataloader)

    elif args.method == 'finetune':
        best_model, best_head, \
            best_records, best_logit_scale = train_ce(args, logger, loss_logger, model, classifier_head, \
                                                      train_loader, val_loader, test_loader, reload_model)

    elif args.method == 'finetune-mixed': # half batch is retrieved, half batch is fewshot
        best_model, best_head, \
            best_records, best_logit_scale = train_ce_mixed(args, logger, loss_logger, model, classifier_head, \
                                                            train_loader, val_loader, test_loader)

    elif args.method == 'fixmatch': # bs is labeled, bs*mu is unlabeled
        best_model, best_head, \
            best_records, best_logit_scale, mask, impurity = train_fixmatch(args, logger, loss_logger, model, classifier_head, \
                                                            train_loader, val_loader, test_loader, reload_model)
        results['mask'] = mask
        results['impurity'] = impurity

    elif args.method == 'debiasPL':
        best_model, best_head, \
            best_records, best_logit_scale = train_debiasPL(args, logger, loss_logger, model, classifier_head, \
                                                            train_loader, val_loader, test_loader, reload_model)

    elif args.method == 'finetune-multitask': # 1 backbone 2 output heads

        best_model, best_head, \
            best_records, best_logit_scale = train_ce_multitask(args, logger, loss_logger, model, classifier_head, \
                                                                train_loader, val_loader, test_loader, dataset_classifier_head)

    elif args.method == 'mixup': # random mixup
        best_model, best_head, \
            best_records, best_logit_scale = train_mixup(args, logger, loss_logger, model, classifier_head, \
                                                         train_loader, val_loader, test_loader)

    elif args.method == 'mixup-fs': # mix retrieved with few-shot
        best_model, best_head, \
            best_records, best_logit_scale = train_mixup_fs(args, logger, loss_logger, model, classifier_head, \
                                                             train_loader, val_loader, test_loader)

    elif args.method == 'cutmix': # cutmix
        best_model, best_head, \
            best_records, best_logit_scale = train_cutmix(args, logger, loss_logger, model, classifier_head, \
                                                          train_loader, val_loader, test_loader)

    elif args.method == 'cutmix-fs': # cutmix with few-shot data
        best_model, best_head, \
            best_records, best_logit_scale = train_cutmix_fs(args, logger, loss_logger, model, classifier_head, \
                                                             train_loader, val_loader, test_loader)

    elif args.method == 'CMO': # CMO
        best_model, best_head, \
            best_records, best_logit_scale = train_CMO(args, logger, loss_logger, model, classifier_head, \
                                                       train_loader, val_loader, test_loader)

    elif args.method == 'resizemix': # resizemix
        best_model, best_head, \
            best_records, best_logit_scale = train_resizemix(args, logger, loss_logger, model, classifier_head, \
                                                             train_loader, val_loader, test_loader)

    elif args.method == 'saliencymix': # saliencymix
        #----- paper code, use first image saliency for entire batch
        # best_model, best_head, best_records, best_logit_scale = train_saliencymix(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader)
        #----- my code, use individual image saliency for each image in the batch
        best_model, best_head, best_records, best_logit_scale = train_saliencymix2(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader)

    elif args.method == 'attentivemix': # attentivemix
        # irregular binary mask
        # best_model, best_head, best_records, best_logit_scale = train_attentivemix(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader)

        # rectangular patches as SaliencyMix2
        best_model, best_head, best_records, best_logit_scale = train_attentivemix2(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader)

    elif args.method == 'FLYP':
        best_model, best_head, best_records, best_logit_scale = train_flyp(args, logger, loss_logger, model, tokenizer,
                                                                            train_loader, val_loader, test_loader, text_prompts)
    elif args.method == 'SupContrastive':
        best_model, best_head, best_records, best_logit_scale = train_supervised_contrastive(args, logger, loss_logger, model, classifier_head,
                                                                                             logit_scale, loss, optimizer, scheduler,
                                                                                             train_loader, val_loader, test_loader)
    elif args.method == 'BalancedContrastive':
        best_model, best_head, best_records, best_logit_scale = train_balanced_contrastive(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader)

    else:
        raise NotImplementedError(f"Method {args.method} not implemented.")

    if args.method == 'dataset-cls':
        exit()

    #---------- Test the wsft, cannot preextract feature, as the model backbone weights is ensembled
    wsft_test_acc = -1
    wsft_backbone, wsft_head = None, None
    # wsft_backbone, wsft_head, wsft_test_acc = run_wsft(args, best_model, best_head, test_loader, zeroshot_model, zeroshot_weights, best_logit_scale, logger)
    # wsft_backbone, wsft_head, wsft_test_acc = run_wsft_alpha(args, best_model, best_head, val_loader, \
    #                                                          test_loader, zeroshot_model, zeroshot_head, \
    #                                                         best_logit_scale, logger)

    # Here we re-extract the val, test dataloader after training, for fast checking of tau normalization
    # new_val_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_val_features_new.pth'
    # new_test_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_test_features_new.pth'
    # val_loader = extract_dataloader(args, best_model, args.val_split, new_val_fea_path, preprocess, tokenized_text_prompts)
    # test_loader = extract_dataloader(args, best_model, args.test_split, new_test_fea_path, preprocess, tokenized_text_prompts)
    # logger.info(f'Extracted val, test dataloader for fast testing after training.')

    #---------- Testing
    test_acc, test_loss, test_confusion_matrix = validate(args,data_loader=test_loader,
                                                        model=best_model,
                                                        classifier_head=best_head,
                                                        logger=logger,
                                                        loss=args.loss, logit_scale=best_logit_scale,
                                                        show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        is_encoder=False,
                                                        )
    test_scores = calculate_scores(test_confusion_matrix)
    logger.info(f"+++++ Test Acc: {round(test_acc, 3)}")
    save_test_scores(test_scores, test_confusion_matrix, args.output_dir, 'test')
    # save_head_weights(best_head, args.output_dir, 'best_val')

    #---------- Tau normalization
    best_tau, best_tau_test_acc = -1, -1
    best_tau_head = None
    # best_tau_head, best_tau, best_tau_test_acc = run_tau_normalization(args, best_head, best_model, val_loader, \
    #                                                                    test_loader, best_logit_scale, logger)

    # print the logit_scale
    # logger.info(f"logit_scale: {round(args.logit_scale.item(), 8)}")
    logger.info(f"best_logit_scale: {round(best_logit_scale.item(), 8)}")

    #----------- save stage 1 best model
    best_model_path = None
    best_model_path = save_best_model(args, best_records,
                                    best_model, best_head, best_logit_scale,
                                    test_acc, best_tau, best_tau_test_acc, wsft_test_acc,
                                    best_tau_head, wsft_backbone, wsft_head, stage=1)
    logger.info(f'Stage 1 Best Model saved to: {best_model_path}')

    # remove the extracted features
    # os.remove(new_val_fea_path)
    # os.remove(new_test_fea_path)

    return test_acc, best_model_path, test_loader_copy, results



def run_stage2_probing2(stage1_best_model_path, test_loader):

    logger.info(f"Run stage 2 classifier retraining ......")

    args.model_path = stage1_best_model_path
    load_model(args, logger, model, test_loader, classifier_head, is_encoder=False)

    # re-extract the train_loader, val_loader, test_loader
    train_dataset = load_dataset(dataset_root=args.dataset_root,
                                split=args.fewshot_data, # note here we use fewshot data !!!
                                preprocess=data_transforms['train'],
                                tokenized_text_prompts=None,
                                )

    test_dataset = load_dataset(dataset_root=args.dataset_root,
                                split=args.test_split,
                                preprocess=data_transforms['test'],
                                tokenized_text_prompts=None,
                                )

    train_loader = DataLoader(train_dataset, batch_size=args.bsz, pin_memory=True,
                            shuffle=True, drop_last=True, num_workers=args.num_workers)

    test_loader = DataLoader(test_dataset, batch_size=128, drop_last=False, pin_memory=True,
                            shuffle=False, num_workers=args.num_workers)

    val_loader = train_loader

    # reset the pre_extracted flag
    args.method = 'probing'
    args.pre_extracted = True
    logger.info(f'Reset args.pre_extracted: {args.pre_extracted}')
    args.epochs = 10
    args.early_stop = False

    # update the lr, wd, optim for classifier retraining in stage 2
    if args.model_cfg.split('_')[0] == 'resnet50':
        args.lr_classifier = 1e-4
    else:
        args.lr_classifier = 1e-4

    args.wd = 1e-4
    args.optimizer = 'AdamW'

    # Imporatnt! Need to reset the params, optimizer, scheduler, loss, logit_scale
    loss = set_loss(args)

    if args.method == 'probing':
        for param in model.parameters():
            param.requires_grad = False
        for param in classifier_head.parameters():
            param.requires_grad = True
        params = [{'params': classifier_head.parameters(), 'lr': args.lr_classifier}]

    else:
        raise ValueError(f"Stage 2 method must be probing.")

    # for RN50 backbone, do not learn the temp/logit_scale
    # logit_scale  = torch.tensor([4.60517]).to(device=args.device)
    logit_scale = torch.tensor([0.0]).to(device=args.device)

    optimizer, scheduler, total_iter = set_optimizer(args, params, train_loader)

    args.loss = loss
    args.logit_scale = logit_scale
    args.optimizer = optimizer
    args.scheduler = scheduler

    #---------- Training
    best_model, best_head, best_records, _, _, test_loader = train_probing2(args, logger, loss_logger, model, classifier_head,
                                                                 train_loader, val_loader, test_loader,
                                                                 reload_model=False, is_stage2=True)

    # test the best model after probing, using the returned test_loader above
    test_acc, test_loss, test_confusion_matrix = validate(args,data_loader=test_loader,
                                                        model=best_model,
                                                        classifier_head=best_head,
                                                        logger=logger,
                                                        loss=args.loss,
                                                        logit_scale=args.logit_scale,
                                                        show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir,
                                                        device=args.device,
                                                        is_encoder=False,
                                                        )
    test_scores = calculate_scores(test_confusion_matrix)
    logger.info(f"+++++ Stage 2 LP Test Acc: {round(test_acc, 3)}")
    save_test_scores(test_scores, test_confusion_matrix, args.output_dir, 'test', stage=2)

    #----------- save stage 2 best model
    best_model_path = save_best_model(args, best_records,
                                    best_model, best_head, logit_scale,
                                    test_acc, best_tau=None, best_tau_test_acc=-1, wsft_test_acc=-1,
                                    best_tau_head=None, wsft_backbone=None, wsft_head=None, stage=2)

    logger.info(f'stage 2 Best Model saved to: {best_model_path}')

    # remove the extracted features
    # remove any file ending with _stage2.pth in the path f'{args.dataset_root}/pre_extracted/
    file_list = os.listdir(f'{args.dataset_root}/pre_extracted/')
    files = [f for f in file_list if f.endswith('_stage2.pth')]
    for f in files:
        os.remove(f'{args.dataset_root}/pre_extracted/{f}')
        logger.info(f"Removed {f}")

    return test_acc, best_model_path


def run_stage2_FSFT2(stage1_best_model_path, test_loader):

    # reset the pre_extracted flag
    args.method = 'finetune'
    args.pre_extracted = False
    logger.info(f'Reset args.pre_extracted: {args.pre_extracted}')
    args.epochs = 10
    args.early_stop = False
    args.save_ckpt = False

    logger.info(f"Run stage 2 few-shot finetuning ......")

    args.model_path = stage1_best_model_path
    # load_model(args, logger, model, test_loader, classifier_head)
    load_model(args, logger, model, None, classifier_head)

    # re-extract the train_loader, val_loader, test_loader
    train_dataset = load_dataset(dataset_root=args.dataset_root,
                                split=args.fewshot_data, # note here we use fewshot data !!!
                                preprocess=data_transforms['train'],
                                tokenized_text_prompts=None,
                                )

    test_dataset = load_dataset(dataset_root=args.dataset_root,
                                split=args.test_split,
                                preprocess=data_transforms['test'],
                                tokenized_text_prompts=None,
                                )

    train_loader = DataLoader(train_dataset, batch_size=args.bsz, pin_memory=True,
                            shuffle=True, drop_last=True, num_workers=args.num_workers)

    test_loader = DataLoader(test_dataset, batch_size=128, drop_last=False, pin_memory=True,
                            shuffle=False, num_workers=args.num_workers)

    val_loader = train_loader

    # update the lr, wd, optim for stage 2 finetuning
    if args.model_cfg.split('_')[0] == 'resnet50':
        args.lr_classifier = 1e-4
        args.lr_backbone = 1e-4
    else:
        args.lr_classifier = 1e-4
        args.lr_backbone = 1e-6

    args.wd = 1e-2
    args.optimizer = 'AdamW'
    # args.optimizer = 'SGD'


    # Imporatnt! Need to reset the params, optimizer, scheduler, loss, logit_scale
    loss = set_loss(args)

    if args.method == 'finetune':
        for param in model.parameters():
            param.requires_grad = True
        for param in classifier_head.parameters():
            param.requires_grad = True
        params = [{'params': classifier_head.parameters(), 'lr': args.lr_classifier},
                  {'params': model.parameters(), 'lr': args.lr_backbone}]

    else:
        raise ValueError(f"Stage 2 method here must be finetune.")

    # for RN50 backbone, do not learn the temp/logit_scale
    # maybe we should follow the temp_scheme here. But this is ImageNet pretrained models, does not matter.
    # logit_scale  = torch.tensor([4.60517]).to(device=args.device)
    logit_scale = torch.tensor([0.0]).to(device=args.device)

    optimizer, scheduler, total_iter = set_optimizer(args, params, train_loader)

    args.loss = loss
    args.logit_scale = logit_scale
    args.optimizer = optimizer
    args.scheduler = scheduler

    #---------- Training
    best_model, best_head, best_records, _  = train_ce(args, logger, loss_logger, model, classifier_head, \
                                                      train_loader, val_loader, test_loader, reload_model=False)

    # test the best model after FSFT
    test_acc, test_loss, test_confusion_matrix = validate(args, data_loader=test_loader,
                                                        model=best_model,
                                                        classifier_head=best_head,
                                                        logger=logger,
                                                        loss=args.loss,
                                                        logit_scale=args.logit_scale,
                                                        show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir,
                                                        device=args.device,
                                                        is_encoder=False,
                                                        )
    test_scores = calculate_scores(test_confusion_matrix)
    logger.info(f"+++++ Stage 2 FSFT Test Acc: {round(test_acc, 3)}")
    save_test_scores(test_scores, test_confusion_matrix, args.output_dir, 'test', stage=3)

    #----------- save stage 2 best model
    best_model_path = save_best_model(args, best_records,
                                    best_model, best_head, logit_scale,
                                    test_acc, best_tau=None, best_tau_test_acc=-1, wsft_test_acc=-1,
                                    best_tau_head=None, wsft_backbone=None, wsft_head=None,
                                    stage=3 # note here I set the stage to 3 for FSFT
                                    )

    logger.info(f'Stage 2 FSFT Best Model saved to: {best_model_path}')


    return test_acc, best_model_path



if __name__ == '__main__':

    program_start = time.time()
    args = parse_args()
    logger, loss_logger = set_logger(args)
    set_training_seed(args)

    # load model
    cfgs = args.model_cfg.split('_')
    if cfgs[0] == 'resnet50' and cfgs[1] == 'scratch':
        model_ft = resnet50(weights=None)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Identity()

    elif cfgs[0] == 'resnet50' and cfgs[1] == 'imagenet':
        model_ft = resnet50(weights='IMAGENET1K_V2') # also default weights using new training receipe
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Identity()

    elif cfgs[0] == 'resnet50' and cfgs[1] == 'inat':
        model_ft = resnet50(weights=None)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Identity()

        ## loading inat pre-trained model
        model_ft = torch.nn.DataParallel(model_ft)
        checkpoint_filename = '/scratch/group/real-fs/model_ckpts/inat_resnet50.pth.tar'
        # print("=> loading checkpoint '{}'".format(checkpoint_filename))
        checkpoint = torch.load(checkpoint_filename)
        del checkpoint['state_dict']['module.fc.bias']
        del checkpoint['state_dict']['module.fc.weight']
        model_ft.load_state_dict(checkpoint['state_dict'], strict=True)

    elif cfgs[0] == 'vitb32' and cfgs[1] == 'imagenet':
        # Load the pretrained weights
        weights = ViT_B_32_Weights.IMAGENET1K_V1
        model_ft = vit_b_32(weights=weights)
        # print(model_ft.parameters)
        # Get in_features safely
        last_linear = model_ft.heads[-1]
        num_ftrs = last_linear.in_features
        logger.info(f'Loaded model: {args.model_cfg}, num_ftrs: {num_ftrs}')
        model_ft.heads = torch.nn.Identity()

    elif cfgs[0] == 'dinov2':
        model_ft = DinoVisionTransformer(args.model_cfg)
        logger.info(f'Loaded model: {args.model_cfg}')
        num_ftrs = dinvo_fts[cfgs[1]]

    else:
        raise NotImplementedError(f"Model {args.model_cfg} not implemented.")


    model = model_ft
    model = model.to(args.device)

    # create classifier head
    classifier_head = MyLinear(inp_dim=num_ftrs, num_classes=args.num_classes)

    # load the classifier weights using the classifier after probing
    if args.cls_path is not None:
        checkpoint = torch.load(args.cls_path)
        classifier_head.load_state_dict(checkpoint['head'])
        logger.info(f'Loaded classifier weights from {args.cls_path}')

    classifier_head = classifier_head.to(args.device)

    # ==================  Craete data loader ==================================
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            # transforms.ColorJitter(Brightness=0.4, Contrast=0.4, Color=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    data_transforms['l_train'] = data_transforms['train']
    data_transforms['u_train'] = data_transforms['train']
    data_transforms['val'] = data_transforms['test']


    # run finetuning for stage 1
    stage1_acc, stage1_best_model_path, test_loader, results = run_stage1_finetuning2(args, logger, model, classifier_head, data_transforms)
    stage1_method = args.method # record method here, as in stage 2 method will be updated to probing

    # replace the stage1_best_model_path and set stage 1 epoch to 0 to run stage 2 for certrain checkpoints
    if args.skip_stage1:
        # stage1_best_model_path= 'output/CE_cutmix_fewshot+retrieved_resnet50_inat_pretrained/output_semi-aves/semi-aves_cutmix_fewshot+retrieved_random_16shots_seed1_50eps/stage1_model_best.pth'
        stage1_best_model_path = args.stage1_model_path

    test_loader_copy = copy.deepcopy(test_loader)
    stage2_lp_acc = -1
    stage2_fsft_acc = -1

    #---------- run probing for stage 2
    if not args.skip_stage2:
        stage2_lp_acc, stage2_best_model_path = run_stage2_probing2(stage1_best_model_path, test_loader)
    else:
        logger.info(f"Skip stage 2 classifier retraining.")
        stage2_acc = -1
        stage2_best_model_path = 'None'

    #---------- run FSFT for stage 2
    if not args.skip_stage2:
        stage2_fsft_acc, stage2_best_model_path = run_stage2_FSFT2(stage1_best_model_path, test_loader_copy)
    else:
        logger.info(f"Skip stage 2 FSFT.")
        stage2_fsft_acc = -1
        stage2_best_model_path = 'None'


    loss_logger.close()
    program_end = time.time()
    logger.info(f"Total time: {round((program_end-program_start)/60, 1)} mins.")

    result_summary = f'{args.dataset},{stage1_method},{args.model_cfg},{args.data_source},' \
                     f'{args.cls_init},{args.shots},{args.seed},{args.retrieval_split},'\
                     f'{args.temp_scheme},{args.temperature},' \
                     f'{round(stage1_acc,1)},{round(stage2_lp_acc,1)},{round(stage2_fsft_acc,1)}'

    logger.info(f'{result_summary}')
    print(f'{result_summary}')