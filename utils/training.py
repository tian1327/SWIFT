import random
import torch
import numpy as np
import copy
from .models import build_classifier_head, save_model_ckpt, save_test_scores, ProjectionMLP
from .dataloader import extract_dataloader, extract_train_dataloader, get_dataloader_preextracted
from testing import validate, calculate_scores, validate_dataset, load_model
from .features import extract_test_feats2
import os
import torch.nn.functional as F
from utils.extras import AverageMeter
from utils.losses import set_loss, SupConLoss
from utils.optimizers import set_optimizer
# from itertools import cycle
from utils.ema import ModelEMA
from utils.moco_v2 import MoCo_ViT, adjust_moco_momentum
from tqdm import tqdm
import time
import torch.nn as nn
import pickle
import json


def set_training_seed(args):

    # set the seed for training
    random.seed(args.training_seed)
    torch.manual_seed(args.training_seed)
    np.random.seed(args.training_seed)
    torch.cuda.manual_seed(args.training_seed)
    torch.cuda.manual_seed_all(args.training_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def run_zeroshot(args, test_loader, model, logger, loss, logit_scale, classifier_head, is_encoder=True):
    if args.method == 'dataset-cls':
        zs_test_acc, zs_loss, zs_confusion_matrix = validate_dataset(args, data_loader=test_loader, model=model, logger=logger,
                                                            loss=loss, logit_scale=logit_scale,
                                                            classifier_head=classifier_head, show_confusion_matrix=True,
                                                            dataset=args.dataset,
                                                            output_dir=args.output_dir, device=args.device,
                                                            )
    else:
        zs_test_acc, zs_loss, zs_confusion_matrix = validate(args, data_loader=test_loader, model=model, logger=logger,
                                                            loss=loss, logit_scale=logit_scale,
                                                            classifier_head=classifier_head, show_confusion_matrix=True,
                                                            dataset=args.dataset,
                                                            output_dir=args.output_dir, device=args.device,
                                                            is_encoder=is_encoder,
                                                            )

    logger.info(f"+++++ Zero-shot Test Acc: {round(zs_test_acc, 3)}")
    # zs_scores = calculate_scores(zs_confusion_matrix)
    # save_test_scores(zs_scores, zs_confusion_matrix, output_dir, 'zeroshot_test')


def train_probing(args, logger, loss_logger, model, classifier_head, tokenized_text_prompts, preprocess, train_loader, val_loader, test_loader, reload_model=False):
    """ Train the model with Cross-Entropy Loss, linear probing"""

    new_train_fea_path, new_val_fea_path, new_test_fea_path = None, None, None
    if reload_model:
        load_model(args, logger, model, test_loader, classifier_head)

        # Here we reextract the test dataloader for fast testing after training, tau normalization, and WiSE-FT
        new_train_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_train_features_new.pth'
        new_val_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_val_features_new.pth'
        new_test_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_test_features_new.pth'

        train_loader = extract_train_dataloader(args, model, args.train_split, new_train_fea_path,
                                                preprocess, tokenized_text_prompts, bsz=args.bsz)
        val_loader = extract_dataloader(args, model, args.val_split, new_val_fea_path, preprocess, tokenized_text_prompts)
        test_loader = extract_dataloader(args, model, args.test_split, new_test_fea_path, preprocess, tokenized_text_prompts)
        logger.info(f'Extracted train, val, test dataloader for probing.')
        # reset the pre_extracted flag
        args.pre_extracted = True
        logger.info(f'Reset args.pre_extracted: {args.pre_extracted}')
        time.sleep(0.5)

        # # Here we reextract the test dataloader for fast testing after training, tau normalization, and WiSE-FT
        # new_train_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_train_features_new.pth'
        # new_val_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_val_features_new.pth'
        # new_test_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_test_features_new.pth'

        # train_loader = extract_train_dataloader(args, logger, model, args.train_split, new_train_fea_path, args.bsz)
        # val_loader = extract_dataloader(args, model, args.val_split, new_val_fea_path)
        # test_loader = extract_dataloader(args, model, args.test_split, new_test_fea_path)
        # logger.info(f'Extracted train, val, test dataloader for stage 2 training.')
        # # reset the pre_extracted flag
        # args.pre_extracted = True
        # logger.info(f'Reset args.pre_extracted: {args.pre_extracted}')

    logger.info(f"Start Training Linear Probing ......")

    model.eval()
    classifier_head.train()

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    val_loss = -1
    val_acc = -1
    test_acc = -1
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0

    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            if not args.pre_extracted:
                image_features = model.encode_image(images)
                image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            else:
                image_feature = images

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()
            total_loss = loss(logits, labels)
            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        if args.early_stop or epoch == args.epochs:
            val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        )
            scores = calculate_scores(confusion_matrix)

        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        # if epoch == args.epochs:
        test_acc, _, _ = validate(args, data_loader=test_loader, model=model, logger=logger,
                            loss=args.loss, logit_scale=logit_scale,
                            classifier_head=classifier_head,
                            dataset=args.dataset,
                            output_dir=args.output_dir, device=args.device,
                            )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()

        lr = scheduler.get_last_lr()
        lr_classifier = lr[0]

        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, lr_cls: {round(lr_classifier,8)}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

    logger.info(f'Linear Probing done.')

    # remove the new_train_fea_path, new_val_fea_path, new_test_fea_path if exists
    for fea_path in [new_train_fea_path, new_val_fea_path, new_test_fea_path]:
        if fea_path and os.path.exists(fea_path):
            os.remove(fea_path)
            logger.info(f'Removed {fea_path}')
        else:
            pass

    return best_model, best_head, best_records, best_logit_scale, val_loader, test_loader



def train_probing2(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader, reload_model=False, is_stage2=False):
    """
    Train the model with Cross-Entropy Loss, linear probing
    For ResNet or Dinov2, which uses model(input) instead of model.encode_image(input)
    """

    if reload_model:
        load_model(args, logger, model, test_loader, classifier_head)
        # Here we reextract the test dataloader for fast testing after training, tau normalization, and WiSE-FT
        new_train_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_train_features_new.pth'
        new_val_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_val_features_new.pth'
        new_test_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_test_features_new.pth'

        train_loader = extract_train_dataloader(args, logger, model, args.train_split, new_train_fea_path, args.bsz)
        val_loader = extract_dataloader(args, model, args.val_split, new_val_fea_path)
        test_loader = extract_dataloader(args, model, args.test_split, new_test_fea_path)
        logger.info(f'Extracted train, val, test dataloader for stage 2 training.')
        # reset the pre_extracted flag
        args.pre_extracted = True
        logger.info(f'Reset args.pre_extracted: {args.pre_extracted}')

    logger.info(f"Start Training (linear probing) ......")

    model.eval()
    classifier_head.train()

    # pre-extract the train val test features
    args.pre_extracted = True
    if is_stage2:
        suffix = '_stage2'
        args.recal_fea = True # must recalculate the features for stage 2 as stage 1 model is changed !!!
    else:
        suffix = ''
    pre_extract_train_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_train_features{suffix}.pth'
    # pre_extract_val_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_val_features{suffix}.pth'
    pre_extract_test_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_test_features{suffix}.pth'

    if args.recal_fea or not os.path.exists(pre_extract_train_fea_path):
        train_features = extract_test_feats2(model, dataloader=train_loader)
        torch.save(train_features, pre_extract_train_fea_path)
        logger.info(f'Extracted train features to {pre_extract_train_fea_path}')
    else:
        logger.info(f'Found pre-extracted train features at {pre_extract_train_fea_path}')

    # if args.recal_fea or not os.path.exists(pre_extract_val_fea_path):
    #     val_features = extract_test_feats2(model, dataloader=val_loader)
    #     torch.save(val_features, pre_extract_val_fea_path)
    #     logger.info(f'Extracted val features to {pre_extract_val_fea_path}')
    # else:
    #     logger.info(f'Found pre-extracted val features at {pre_extract_val_fea_path}')

    if args.recal_fea or not os.path.exists(pre_extract_test_fea_path):
        test_features = extract_test_feats2(model, dataloader=test_loader)
        torch.save(test_features, pre_extract_test_fea_path)
        logger.info(f'Extracted test features to {pre_extract_test_fea_path}')
    else:
        logger.info(f'Found pre-extracted test features at {pre_extract_test_fea_path}')

    # rebuild the dataloaders
    train_loader, val_loader, test_loader = get_dataloader_preextracted(args, logger,
                                                                        pre_extract_train_fea_path,
                                                                        # pre_extract_val_fea_path,
                                                                        pre_extract_train_fea_path,
                                                                        pre_extract_test_fea_path,
                                                                        args.device)

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    val_loss = -1
    val_acc = -1
    test_acc = -1
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0

    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            optimizer.zero_grad()

            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            if args.pre_extracted:
                image_feature = images
            else:
                image_feature = model(images)
                # image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()
            total_loss = loss(logits, labels)
            train_loss = total_loss.item()
            train_loss_sum += train_loss

            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # if args.early_stop or epoch == args.epochs:
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                    loss=args.loss, logit_scale=logit_scale,
                                                    classifier_head=classifier_head, show_confusion_matrix=True,
                                                    dataset=args.dataset,
                                                    output_dir=args.output_dir, device=args.device,
                                                    is_encoder=False,
                                                    )
        scores = calculate_scores(confusion_matrix)

        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        # if epoch == args.epochs:
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                            loss=args.loss, logit_scale=logit_scale,
                            classifier_head=classifier_head,
                            dataset=args.dataset,
                            output_dir=args.output_dir, device=args.device,
                            is_encoder=False,
                            )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

    logger.info(f'Probing done.')

    return best_model, best_head, best_records, best_logit_scale, val_loader, test_loader



def train_CMLP(args, logger, loss_logger, model, classifier_head, preprocess, \
               tokenized_text_prompts,train_loader, val_loader, test_loader, \
                reload_model=False, text_dataloader=None):
    """ Train the model with Cross-Entropy Loss, linear probing"""

    if reload_model:
        load_model(args, logger, model, test_loader, classifier_head)
        # Here we reextract the test dataloader for fast testing after training, tau normalization, and WiSE-FT
        new_train_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_train_features_new.pth'
        new_val_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_val_features_new.pth'
        new_test_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_test_features_new.pth'

        train_loader = extract_train_dataloader(args, logger, model, args.train_split, new_train_fea_path,
                                                preprocess, tokenized_text_prompts, args.bsz)

        val_loader = extract_dataloader(args, model, args.val_split, new_val_fea_path, preprocess, tokenized_text_prompts)
        test_loader = extract_dataloader(args, model, args.test_split, new_test_fea_path, preprocess, tokenized_text_prompts)
        logger.info(f'Extracted train, val, test dataloader for stage 2 training.')
        # reset the pre_extracted flag
        args.pre_extracted = True
        logger.info(f'Reset args.pre_extracted: {args.pre_extracted}')

    logger.info(f"Start Training (cross-modal linear probing) ......")

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    assert text_dataloader is not None, 'text_dataloader is None.'
    assert args.pre_extracted, 'args.pre_extracted is False.'

    model.eval()
    classifier_head.train()

    best_records = {}
    best_val_acc = -1
    num_iter = 0

    text_loader = text_dataloader
    text_loader_iter = iter(text_loader)

    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            image_feature = images

            try:
                text, text_label = next(text_loader_iter)
            except StopIteration:
                text_loader_iter = iter(text_loader)
                text, text_label = next(text_loader_iter)

            # concatenate image and text features
            combined_feature = torch.cat([image_feature, text], dim=0)
            combined_labels = torch.cat([labels, text_label], dim=0)

            logits = classifier_head(combined_feature)
            logits = logits * logit_scale.exp()
            total_loss = loss(logits, combined_labels)
            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        )
        scores = calculate_scores(confusion_matrix)

        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,

                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

    logger.info(f'Probing done.')

    return best_model, best_head, best_records, best_logit_scale, val_loader, test_loader


def train_ce(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader, reload_model=False):
    """ Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier"""

    cfgs = args.model_cfg.split('_')
    is_encoder = (cfgs[1] == 'openclip' or cfgs[1] == 'clip')

    if reload_model:
        load_model(args, logger, model, test_loader, classifier_head, is_encoder)

    logger.info(f"Start standard finetuning ......")
    logger.info(f"args.train_split: {args.train_split}")

    # model.eval() if args.freeze_visual else model.train()
    model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    val_loss = -1
    val_acc = -1
    test_acc = -1
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0

    for epoch in range(1, args.epochs+1):
        train_loss_sum = 0
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()
            source = source.to(args.device)

            if is_encoder:
                image_features = model.encode_image(images)
                image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            else:
                image_feature = model(images)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()
            if args.loss_name == 'WeightedCE':
                total_loss = loss(logits, labels, source) # for WeightedCE, needs to input the source
            else:
                total_loss = loss(logits, labels)
            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

            # break

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        if args.early_stop or epoch == args.epochs:
            val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        is_encoder=is_encoder,
                                                        )
            scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        # if args.early_stop or epoch == args.epochs:
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                        loss=args.loss, logit_scale=logit_scale,
                        classifier_head=classifier_head,
                        dataset=args.dataset,
                        output_dir=args.output_dir, device=args.device,
                        is_encoder=is_encoder,
                        )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()

        lr = scheduler.get_last_lr()
        # print(lr)
        if args.temp_scheme == 'none' or len(lr) == 2:
            lr_classifier, lr_backbone, lr_temp = lr[0], lr[1], 0.0
        else:
            lr_classifier, lr_backbone, lr_temp = lr[0], lr[1], lr[2]



        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, lr_bkb: {round(lr_backbone, 8)}, "
                    f"lr_cls: {round(lr_classifier,8)}, "
                    f"Trn Loss: {round(train_loss_avg, 6)}, "
                    f"logit_scale: {round(logit_scale.item(), 6)}, "
                    f"Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, "
                    f"Test Acc: {round(test_acc, 3)}")

        # logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale


def train_flyp(args, logger, loss_logger, model, tokenizer,
               train_dataloader, val_dataloader, test_dataloader, text_prompts):
    """
    Finetune like you pretrain
    Train the model with contrastive loss, using the text descriptions from labels.
    Can be modified to lock the text encoder.
    """

    assert (args.loss_name == 'CE' or args.loss_name == 'WeightedCE'), 'FLYP use CE loss for contrastive loss calculation.'


    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    val_loss = -1
    val_acc = -1
    test_acc = -1
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = None
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    # if args.model_path:
    #     model, _ = load_model(args, logger, model, test_dataloader, logit_scale, classifier_head)

    logger.info(f"Start Training FLYP ......")

    model.train()
    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, tokenized_text, source in train_dataloader:
            optimizer.zero_grad()
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()
            tokenized_text = tokenized_text.to(args.device) # currently we use 1 template for semi-aves as in prompt_maker(), can be updated to randomly sample 1 from the 80 prompts

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            prompts = tokenized_text.squeeze()
            text_features = model.encode_text(prompts)
            text_feature = text_features / text_features.norm(dim=-1, keepdim=True) # Normalization

            scale = logit_scale.exp()
            logits_per_image = scale * image_feature @ text_feature.t()
            logits_per_text = logits_per_image.t()
            labels = torch.arange(logits_per_image.shape[0], dtype=torch.long).to(args.device)

            if args.loss_name == 'CE':
                total_loss = (loss(logits_per_image, labels) + loss(logits_per_text, labels)) / 2
            elif args.loss_name == 'WeightedCE':
                total_loss = (loss(logits_per_image, labels, source) + loss(logits_per_text, labels, source)) / 2
            else:
                raise ValueError(f'Loss {args.loss_name} not supported for FLYP training.')

            # total_loss = (loss(logits_per_image, labels) + loss(logits_per_text, labels)) / 2
            train_loss = total_loss.item()
            train_loss_sum += train_loss

            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        # rebuild the classifier head using the updated text encoder
        if args.early_stop or epoch == args.epochs:
            new_head = build_classifier_head(args, model, text_prompts, tokenizer)
            val_acc, val_loss, confusion_matrix = validate(args,data_loader=val_dataloader, model=model, logger=logger,
                                                            loss=loss, logit_scale=logit_scale,
                                                            classifier_head=new_head, show_confusion_matrix=True,
                                                            dataset=args.dataset,
                                                            output_dir=args.output_dir, device=args.device,
                                                            pre_extracted=False,
                                                            )
            scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = logit_scale
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(new_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        if args.early_stop or epoch == args.epochs:
            test_acc, _, _ = validate(args,data_loader=test_dataloader, model=model, logger=logger,
                                loss=loss, logit_scale=logit_scale,
                                classifier_head=new_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,
                                pre_extracted=False,
                                )

        train_loss_avg = train_loss_sum / len(train_dataloader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)}, {round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        # sometime train_loss goes to nan due to numerical instability,
        # here to log the logit_scale and stops the training
        if test_acc == 0.5:
            logger.info(f'logit_scale: {logit_scale.item()}, scale: {scale.item()}')
            logger.info('Test Acc is 0.5, stop training.')
            exit()

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, new_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale


def train_supervised_contrastive(args, logger, loss_logger, model, classifier_head,
                                logit_scale, loss, optimizer, scheduler,
                                train_dataloader, val_dataloader, test_dataloader):
    """ train CLIP visual encoder with supervised contrastive loss, then linear prob to evaluate learned representations """
    assert args.loss == 'SupCon' or args.loss == 'FASupCon',  \
        'Supervised Contrastive Loss is used for training.'





def train_balanced_contrastive(args, logger, loss_logger, model, classifier_head,
                                logit_scale, loss, optimizer, scheduler,
                                train_dataloader, val_dataloader, test_dataloader):
    """ train CLIP visual encoder with supervised contrastive loss, then linear prob to evaluate learned representations """
    exit()




def train_dataset_cls(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    For dataset classification
    """

    logger.info(f"Start standard finetuning ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    val_loss = -1
    val_acc = -1
    test_acc = -1
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0

    for epoch in range(1, args.epochs+1):
        train_loss_sum = 0
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            # labels = labels.to(args.device).long()
            labels = source.to(args.device).long() # use the source as labels
            # source = source.to(args.device)

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()
            total_loss = loss(logits, labels)
            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        if args.early_stop or epoch == args.epochs:
            val_acc, val_loss, confusion_matrix = validate_dataset(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,

                                                        )
            scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        if args.early_stop or epoch == args.epochs:
            test_acc, _, _ = validate_dataset(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,

                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale






def train_ce_mixed(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier.
    Half batch from retrieved data, half batch from few-shot data.
    """

    train_loader, train_dataloader_fs = train_loader
    train_loader_fs = iter(train_dataloader_fs)

    logger.info(f"Start Training ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()
            source = source.to(args.device)

            # get a batch of few-shot data, handle the case when the few-shot data is exhausted, just loop back
            try:
                inputs_fs, labels_fs, text_fs, source_fs = next(train_loader_fs)
            except StopIteration:
                train_loader_fs = iter(train_dataloader_fs)
                inputs_fs, labels_fs, text_fs, source_fs = next(train_loader_fs)
            images_fs = inputs_fs.to(args.device)
            labels_fs = labels_fs.to(args.device).long()
            source_fs = source_fs.to(args.device)

            # concatenate the retrieved data and few-shot data
            images = torch.cat([images, images_fs], dim=0)
            labels = torch.cat([labels, labels_fs], dim=0)
            source = torch.cat([source, source_fs], dim=0)

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()
            if args.loss_name == 'WeightedCE':
                total_loss = loss(logits, labels, source) # for WeightedCE, needs to input the source
            else:
                total_loss = loss(logits, labels)
            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,

                                                        )
        scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,

                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def train_fixmatch3(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader, reload_model):
    """
    Train the model with fixmatch SSL method.
    Part of the batch from labeled data, part is retrieved OOD data, part from unlabeled data
    """

    cfgs = args.model_cfg.split('_')
    is_encoder = (cfgs[1] == 'openclip' or cfgs[1] == 'clip')

    if reload_model:
        load_model(args, logger, model, test_loader, classifier_head, is_encoder)

    train_dataloader, train_loader_retrieve, u_train_dataloader = train_loader

    fewshot_loader = iter(train_dataloader)
    retrieve_loader = iter(train_loader_retrieve)
    u_train_loader = iter(u_train_dataloader)

    logger.info(f"Start FixMatch Training ......")

    # ema_model = ModelEMA(args, model, args.ema_decay)
    # ema_cls = ModelEMA(args, classifier_head, args.ema_decay)
    ema_test_acc = -1

    model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    logit_scale_retrieve = args.logit_scale_r
    logit_scale_u = args.logit_scale_u

    # loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    val_loss = -1
    val_acc = -1
    test_acc = -1
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0

    for epoch in range(1, args.epochs+1):
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_r = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()

        # for iterations in tqdm(range(args.iterations)):
        for iterations in range(args.iterations):

            num_iter += 1

            # load a batch of fewshot data
            try:
                inputs_x, targets_x, _, _ = next(fewshot_loader)
            except StopIteration:
                fewshot_loader = iter(train_dataloader)
                inputs_x, targets_x, _, _ = next(fewshot_loader)

            # load a batch of retrieved data
            try:
                inputs_r, targets_r, _, _ = next(retrieve_loader)
            except StopIteration:
                retrieve_loader = iter(train_loader_retrieve)
                inputs_r, targets_r, _, _ = next(retrieve_loader)

            # load a batch of unlabeled data
            try:
                u_inputs, _, _, _ = next(u_train_loader)
            except StopIteration:
                u_train_loader = iter(u_train_dataloader)
                u_inputs, _, _, _ = next(u_train_loader)

            inputs_u_w, inputs_u_s = u_inputs

            batch_size = inputs_x.shape[0]

            inputs = interleave(
                torch.cat((inputs_x, inputs_r, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)

            targets_x = targets_x.to(args.device)
            targets_r = targets_r.to(args.device)


            if is_encoder:
                image_features = model.encode_image(inputs)
                image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            else:
                image_feature = model(inputs)

            logits = classifier_head(image_feature)

            logits = de_interleave(logits, 2*args.mu+1)

            logits_x = logits[:4]
            logits_r = logits[4:32]
            logits_u_w, logits_u_s = logits[32:].chunk(2)
            del logits

            # fewshot loss
            logits_x = logits_x * logit_scale.exp()
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            # retrieved data loss
            logits_r = logits_r * logit_scale_retrieve.exp()
            Lr = F.cross_entropy(logits_r, targets_r, reduction='mean')

            # unlabeled data loss
            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            logits_u_s = logits_u_s * logit_scale_u.exp()
            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()

            # add up the total loss
            total_loss = Lx + Lr + args.lambda_u * Lu

            losses.update(total_loss.item())
            losses_x.update(Lx.item())
            losses_r.update(Lr.item())
            losses_u.update(Lu.item())
            mask_probs.update(mask.mean().item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            # ema_model.update(model)
            # ema_cls.update(classifier_head)

        if args.early_stop or epoch == args.epochs:
            val_acc, val_loss, confusion_matrix = validate(args,
                                                           data_loader=val_loader,
                                                        #    data_loader=test_loader, # note that here i used test loader for validation
                                                           model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        is_encoder=is_encoder,
                                                        )
            scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        # if args.early_stop or epoch == args.epochs:
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                            loss=args.loss, logit_scale=logit_scale,
                            classifier_head=classifier_head,
                            dataset=args.dataset,
                            output_dir=args.output_dir, device=args.device,
                            is_encoder=is_encoder,
                            )

        # ema_test_acc, _, _ = validate(args,data_loader=test_loader, model=ema_model.ema, logger=logger,
        #                     loss=args.loss, logit_scale=logit_scale,
        #                     classifier_head=ema_cls.ema,
        #                     dataset=args.dataset,
        #                     output_dir=args.output_dir, device=args.device,
        #                     is_encoder=is_encoder,
        #                     )

        train_loss_avg = losses.avg
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()

        lr = scheduler.get_last_lr()
        # logger.info(lr)
        print(lr)
        lr_backbone, lr_classifier = lr[1], lr[0]

        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, lr_bkb: {round(lr_backbone,8)}, lr_cls: {round(lr_classifier,6)}, "
                    f"Trn Loss: {round(train_loss_avg, 6)}, Lx: {round(losses_x.avg, 6)}, Lr: {round(losses_r.avg, 6)}, Lu: {round(losses_u.avg, 6)}, "
                    f"Mask: {round(mask_probs.avg, 6)}, logit_scale_x: {round(logit_scale.item(), 6)}, logit_scale_r: {round(logit_scale_retrieve.item(), 6)}, "
                    f"logit_scale_u: {round(logit_scale_u.item(), 6)}, "
                    f"Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, "
                    f"Test Acc: {round(test_acc, 3)}, EMA Test Acc: {round(ema_test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale


def causal_inference(current_logit, qhat, exp_idx, tau=0.5):
    # de-bias pseudo-labels
    debiased_prob = F.softmax(current_logit - tau*torch.log(qhat), dim=1)
    return debiased_prob


def update_qhat(probs, qhat, momentum, qhat_mask=None):
    if qhat_mask is not None:
        mean_prob = probs.detach()*qhat_mask.detach().unsqueeze(dim=-1)
    else:
        mean_prob = probs.detach().mean(dim=0)
    qhat = momentum * qhat + (1 - momentum) * mean_prob
    return qhat


def get_centroids(prob):
    N, D = prob.shape
    K = D
    cl = prob.argmin(dim=1).long().view(-1)  # -> class index
    Ncl = cl.view(cl.size(0), 1).expand(-1, D)
    unique_labels, labels_count = Ncl.unique(dim=0, return_counts=True)
    labels_count_all = torch.ones([K]).long().cuda() # -> counts of each class
    labels_count_all[unique_labels[:,0]] = labels_count
    c = torch.zeros([K, D], dtype=prob.dtype).cuda().scatter_add_(0, Ncl, prob) # -> class centroids
    c = c / labels_count_all.float().unsqueeze(1)
    return cl, c

def CLDLoss(prob_s, prob_w, mask=None, weights=None):
    cl_w, c_w = get_centroids(prob_w)
    affnity_s2w = torch.mm(prob_s, c_w.t())
    if mask is None:
        loss = F.cross_entropy(affnity_s2w.div(0.07), cl_w, weight=weights)
    else:
        loss = (F.cross_entropy(affnity_s2w.div(0.07), cl_w, reduction='none', weight=weights) * (1 - mask)).mean()
    return loss


def initialize_retrieval(args):

    # use the retrieved concept frequency to estimate the initial qhat

    filepath = f'data/{args.dataset}/{args.dataset}_metrics-LAION400M.json'
    info = json.load(open(filepath, 'r'))

    concept_freq = []
    for key, value in info.items():
        concept_freq.append(value['actual_freq'])

    assert len(concept_freq) == args.num_classes, f'Number of classes {args.num_classes} does not match the concept frequency length {len(concept_freq)}'

    # apply softmax to the concept_freq to get qhat
    count = np.array(concept_freq)
    exps = np.exp(count - np.max(count))  # for numerical stability
    qhat = exps / np.sum(exps)
    qhat = torch.tensor(qhat, dtype=torch.float).cuda().unsqueeze(0)  # make it a 1xC tensor

    return qhat





def train_debiasPL_cutmix(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader, reload_model):
    """
    Train the model with debiasPL SSL method + CutMix
    Part of the batch from labeled data, part from unlabeled data
    """

    # hyperparameters referrred from https://github.com/frank-xwang/debiased-pseudo-labeling/blob/main/scripts/0.2perc-ssl/train_DebiasPL.sh

    args.masked_qhat = False
    # args.qhat_m = 0.99
    args.qhat_m = 0.999
    args.debiased_tau = 0.4
    args.CLDLoss = True
    args.cld_lambda = 0.1
    class_num = args.num_classes

    qhat = (torch.ones([1, class_num], dtype=torch.float)/class_num).cuda() #initialize the logit s offsets
    # qhat = initialize_retrieval(args) # initialize the qhat with the retrieved distribution from concept frequency.

    cfgs = args.model_cfg.split('_')
    is_encoder = (cfgs[1] == 'openclip' or cfgs[1] == 'clip')

    if reload_model:
        load_model(args, logger, model, test_loader, classifier_head, is_encoder)
        # load_model(args, logger, model, None, classifier_head, is_encoder)

    train_dataloader, u_train_dataloader = train_loader
    u_train_loader = iter(u_train_dataloader)

    logger.info(f"Start Training debiasPL-Joint ......")

    model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    logit_scale_u = args.logit_scale_u
    # loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    ema_test_acc = -1
    val_loss = -1
    val_acc = -1
    test_acc = -1
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0

    for epoch in range(1, args.epochs+1):
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_r = AverageMeter()
        losses_u = AverageMeter()
        losses_cld = AverageMeter()
        mask_probs = AverageMeter()
        model.train()
        classifier_head.train()

        for inputs_x, targets_x, _, _ in train_dataloader:

            num_iter += 1

            # load a batch of unlabeled data
            try:
                u_inputs, _, _, _ = next(u_train_loader)
            except StopIteration:
                u_train_loader = iter(u_train_dataloader)
                u_inputs, _, _, _ = next(u_train_loader)

            inputs_u_w, inputs_u_s = u_inputs

            batch_size = inputs_x.shape[0]

            # apply CutMix before interleaving
            images = inputs_x
            labels = targets_x
            r = np.random.rand(1)
            if args.cutmix_beta > 0 and r < args.mix_prob:
                # generate mixed sample
                lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
                # rand_index = torch.randperm(images.size()[0]).cuda()
                rand_index = torch.randperm(images.size()[0])
                target_a = labels
                target_b = labels[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            else:
                target_a = labels
                target_b = labels
                lam = 1.0

            # assign the cutmixed images to inputs_x
            inputs_x = images

            # interleave the labeled and unlabeled data to stabilize training
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)

            # targets_x = targets_x.to(args.device)

            if is_encoder:
                image_features = model.encode_image(inputs)
                image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            else:
                image_feature = model(inputs)

            logits = classifier_head(image_feature)

            logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            logits_x = logits_x * logit_scale.exp()
            # Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
            target_a = target_a.to(args.device)
            target_b = target_b.to(args.device)

            Lx = mixup_criterion(F.cross_entropy, logits_x, target_a, target_b, lam) # use mixup loss

            #----------- unlabeled data loss using DebiasPL

            # producing debiased pseudo-labels, note the /args.T is critical to sharpen the logits so that the mask would be non-zero!!!
            pseudo_label = causal_inference(logits_u_w.detach()/args.T, qhat, exp_idx=0, tau=args.debiased_tau)
            max_probs, pseudo_targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            # print(max_probs)

            # update qhat
            qhat_mask = mask if args.masked_qhat else None
            qhat = update_qhat(torch.softmax(logits_u_w.detach()/args.T, dim=-1), qhat, momentum=args.qhat_m, qhat_mask=qhat_mask)

            per_cls_weights = None # we might be able to use this weights for each class from the true distribution!
            # CLD loss for unlabeled samples (optional)
            if args.CLDLoss:
                prob_s = torch.softmax(logits_u_s, dim=-1)
                prob_w = torch.softmax(logits_u_w.detach(), dim=-1)
                loss_cld = CLDLoss(prob_s, prob_w, mask=None, weights=per_cls_weights)
            else:
                loss_cld = torch.tensor(0.0).cuda()
                args.cld_lambda = 0.0

            # adaptive marginal loss
            delta_logits = torch.log(qhat)
            logits_u_s = logits_u_s * logit_scale_u.exp() # scale the logits for unlabeled data
            logits_u_s = logits_u_s + args.debiased_tau*delta_logits
            Lu = (F.cross_entropy(logits_u_s, pseudo_targets_u, reduction='none', weight=per_cls_weights) * mask).mean()

            #----------- total loss
            total_loss = Lx + args.lambda_u * Lu + args.cld_lambda * loss_cld

            losses.update(total_loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            losses_cld.update(loss_cld.item())
            mask_probs.update(mask.mean().item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # break # for fast debugging

        # validate after 1 epoch
        if args.early_stop or epoch == args.epochs:
            val_acc, val_loss, confusion_matrix = validate(args,
                                                           data_loader=val_loader,
                                                           model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        is_encoder=is_encoder,
                                                        )
            scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        # if args.early_stop or epoch == args.epochs:
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                            loss=args.loss, logit_scale=logit_scale,
                            classifier_head=classifier_head,
                            dataset=args.dataset,
                            output_dir=args.output_dir, device=args.device,
                            is_encoder=is_encoder,
                            )

        train_loss_avg = losses.avg
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()

        lr = scheduler.get_last_lr()
        # print(lr)
        lr_backbone, lr_classifier = lr[1], lr[0]
        qhat_mean = qhat.mean(dim=1).cpu().numpy()

        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, lr_bkb: {round(lr_backbone,8)}, lr_cls: {round(lr_classifier,8)}, "
                    f"Trn Loss: {round(train_loss_avg, 6)}, Lx: {round(losses_x.avg, 6)}, Lu: {round(losses_u.avg, 6)}, "
                    f"L_cld: {round(losses_cld.avg, 6)}, "
                    f"Mask: {round(mask_probs.avg, 6)}, qhat_mean: {qhat_mean}, "
                    f"logit_scale_x: {round(logit_scale.item(), 6)}, "
                    f"logit_scale_u: {round(logit_scale_u.item(), 6)}, "
                    f"Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, "
                    f"Test Acc: {round(test_acc, 3)}, EMA Test Acc: {round(ema_test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale






def train_debiasPL(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader, reload_model):
    """
    Train the model with DebiasPL SSL method.

    https://github.com/frank-xwang/debiased-pseudo-labeling/blob/main/main_DebiasPL.py#L644

    """

    # hyperparameters referrred from https://github.com/frank-xwang/debiased-pseudo-labeling/blob/main/scripts/0.2perc-ssl/train_DebiasPL.sh

    args.masked_qhat = False
    # args.qhat_m = 0.99
    args.qhat_m = 0.999
    args.debiased_tau = 0.4
    args.CLDLoss = True
    args.cld_lambda = 0.1
    class_num = args.num_classes

    qhat = (torch.ones([1, class_num], dtype=torch.float)/class_num).cuda() #initialize the logit s offsets
    # qhat = initialize_retrieval(args) # initialize the qhat with the retrieved distribution from concept frequency.

    cfgs = args.model_cfg.split('_')
    is_encoder = (cfgs[1] == 'openclip' or cfgs[1] == 'clip')

    if reload_model:
        load_model(args, logger, model, test_loader, classifier_head, is_encoder)

    logger.info(f'len(train_loader): {len(train_loader)}')

    train_dataloader, u_train_dataloader = train_loader

    u_train_loader = iter(u_train_dataloader)

    logger.info(f"Start debiasPL Training ......")

    # ema_model = ModelEMA(args, model, args.ema_decay)
    # ema_cls = ModelEMA(args, classifier_head, args.ema_decay)
    ema_test_acc = -1

    model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    logit_scale_u = args.logit_scale_u

    optimizer = args.optimizer
    scheduler = args.scheduler

    val_loss = -1
    val_acc = -1
    test_acc = -1
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0

    for epoch in range(1, args.epochs+1):
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_r = AverageMeter()
        losses_u = AverageMeter()
        losses_cld = AverageMeter()
        mask_probs = AverageMeter()
        model.train()
        classifier_head.train()

        # for iterations in range(args.iterations):
        for inputs_x, targets_x, _, _ in train_dataloader:

            num_iter += 1

            # load a batch of unlabeled data
            try:
                u_inputs, _, _, _ = next(u_train_loader)
            except StopIteration:
                u_train_loader = iter(u_train_dataloader)
                u_inputs, _, _, _ = next(u_train_loader)

            inputs_u_w, inputs_u_s = u_inputs

            batch_size = inputs_x.shape[0]

            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)

            targets_x = targets_x.to(args.device)

            if is_encoder:
                image_features = model.encode_image(inputs)
                image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            else:
                image_feature = model(inputs)

            logits = classifier_head(image_feature)

            logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            # labeled loss
            logits_x = logits_x * logit_scale.exp()
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            #----------- unlabeled data loss using DebiasPL

            # producing debiased pseudo-labels, note the /args.T is critical to sharpen the logits so that the mask would be non-zero!!!
            pseudo_label = causal_inference(logits_u_w.detach()/args.T, qhat, exp_idx=0, tau=args.debiased_tau)
            max_probs, pseudo_targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            # print(max_probs)

            # update qhat
            qhat_mask = mask if args.masked_qhat else None
            qhat = update_qhat(torch.softmax(logits_u_w.detach()/args.T, dim=-1), qhat, momentum=args.qhat_m, qhat_mask=qhat_mask)

            per_cls_weights = None # we might be able to use this weights for each class from the true distribution!
            # CLD loss for unlabeled samples (optional)
            if args.CLDLoss:
                prob_s = torch.softmax(logits_u_s, dim=-1)
                prob_w = torch.softmax(logits_u_w.detach(), dim=-1)
                loss_cld = CLDLoss(prob_s, prob_w, mask=None, weights=per_cls_weights)
            else:
                loss_cld = torch.tensor(0.0).cuda()
                args.cld_lambda = 0.0

            # adaptive marginal loss
            delta_logits = torch.log(qhat)
            logits_u_s = logits_u_s * logit_scale_u.exp() # scale the logits for unlabeled data
            logits_u_s = logits_u_s + args.debiased_tau*delta_logits
            Lu = (F.cross_entropy(logits_u_s, pseudo_targets_u, reduction='none', weight=per_cls_weights) * mask).mean()


            # total loss
            total_loss = Lx + args.lambda_u * Lu + args.cld_lambda * loss_cld

            losses.update(total_loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            losses_cld.update(loss_cld.item())
            mask_probs.update(mask.mean().item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            # ema_model.update(model)
            # ema_cls.update(classifier_head)

            # break # for quick debugging

        if args.early_stop or epoch == args.epochs:
            val_acc, val_loss, confusion_matrix = validate(args,
                                                           data_loader=val_loader,
                                                        #    data_loader=test_loader, # note that here i used test loader for validation
                                                           model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        is_encoder=is_encoder,
                                                        )
            scores = calculate_scores(confusion_matrix)


        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        # if args.early_stop or epoch == args.epochs:
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                            loss=args.loss, logit_scale=logit_scale,
                            classifier_head=classifier_head,
                            dataset=args.dataset,
                            output_dir=args.output_dir, device=args.device,
                            is_encoder=is_encoder,
                            )

        # ema_test_acc, _, _ = validate(args,data_loader=test_loader, model=ema_model.ema, logger=logger,
        #                     loss=args.loss, logit_scale=logit_scale,
        #                     classifier_head=ema_cls.ema,
        #                     dataset=args.dataset,
        #                     output_dir=args.output_dir, device=args.device,
        #                     is_encoder=is_encoder,
        #                     )

        train_loss_avg = losses.avg
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()

        lr = scheduler.get_last_lr()
        # print(lr)
        lr_backbone, lr_classifier = lr[1], lr[0]
        qhat_mean = qhat.mean(dim=1).cpu().numpy()

        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, lr_bkb: {round(lr_backbone,8)}, lr_cls: {round(lr_classifier,8)}, "
                    f"Trn Loss: {round(train_loss_avg, 6)}, Lx: {round(losses_x.avg, 6)}, Lu: {round(losses_u.avg, 6)}, "
                    f"L_cld: {round(losses_cld.avg, 6)}, "
                    f"Mask: {round(mask_probs.avg, 6)}, qhat_mean: {qhat_mean}, "
                    f"logit_scale_x: {round(logit_scale.item(), 6)}, "
                    f"logit_scale_u: {round(logit_scale_u.item(), 6)}, "
                    f"Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, "
                    f"Test Acc: {round(test_acc, 3)}, EMA Test Acc: {round(ema_test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale



def train_fixmatch2(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader, reload_model):
    """
    Train the model with fixmatch SSL method.
    Part of the batch from labeled data, part is retrieved OOD data, part from unlabeled data
    """

    cfgs = args.model_cfg.split('_')
    is_encoder = (cfgs[1] == 'openclip' or cfgs[1] == 'clip')

    if reload_model:
        load_model(args, logger, model, test_loader, classifier_head, is_encoder)

    train_dataloader, train_loader_retrieve, u_train_dataloader = train_loader

    fewshot_loader = iter(train_dataloader)
    retrieve_loader = iter(train_loader_retrieve)
    u_train_loader = iter(u_train_dataloader)

    logger.info(f"Start FixMatch Training ......")

    # ema_model = ModelEMA(args, model, args.ema_decay)
    # ema_cls = ModelEMA(args, classifier_head, args.ema_decay)
    ema_test_acc = -1

    model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    logit_scale_retrieve = args.logit_scale_r
    logit_scale_u = args.logit_scale_u

    # loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    val_loss = -1
    val_acc = -1
    test_acc = -1
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0

    for epoch in range(1, args.epochs+1):
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_r = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()

        # for iterations in tqdm(range(args.iterations)):
        for iterations in range(args.iterations):

            num_iter += 1

            # load a batch of fewshot data
            try:
                inputs_x, targets_x, _, _ = next(fewshot_loader)
            except StopIteration:
                fewshot_loader = iter(train_dataloader)
                inputs_x, targets_x, _, _ = next(fewshot_loader)

            # load a batch of retrieved data
            try:
                inputs_r, targets_r, _, _ = next(retrieve_loader)
            except StopIteration:
                retrieve_loader = iter(train_loader_retrieve)
                inputs_r, targets_r, _, _ = next(retrieve_loader)

            # load a batch of unlabeled data
            try:
                u_inputs, _, _, _ = next(u_train_loader)
            except StopIteration:
                u_train_loader = iter(u_train_dataloader)
                u_inputs, _, _, _ = next(u_train_loader)

            inputs_u_w, inputs_u_s = u_inputs

            batch_size = inputs_x.shape[0]

            inputs = interleave(
                torch.cat((inputs_x, inputs_r, inputs_u_w, inputs_u_s)), 3*args.mu+1).to(args.device)

            targets_x = targets_x.to(args.device)
            targets_r = targets_r.to(args.device)


            if is_encoder:
                image_features = model.encode_image(inputs)
                image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            else:
                image_feature = model(inputs)

            logits = classifier_head(image_feature)

            logits = de_interleave(logits, 3*args.mu+1)
            logits_x = logits[:batch_size]
            logits_r = logits[batch_size:2*batch_size]
            logits_u_w, logits_u_s = logits[2*batch_size:].chunk(2)
            del logits

            # fewshot loss
            logits_x = logits_x * logit_scale.exp()
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            # retrieved data loss
            logits_r = logits_r * logit_scale_retrieve.exp()
            Lr = F.cross_entropy(logits_r, targets_r, reduction='mean')

            # unlabeled data loss
            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            logits_u_s = logits_u_s * logit_scale_u.exp()
            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()

            # add up the total loss
            total_loss = Lx + Lr + args.lambda_u * Lu

            losses.update(total_loss.item())
            losses_x.update(Lx.item())
            losses_r.update(Lr.item())
            losses_u.update(Lu.item())
            mask_probs.update(mask.mean().item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            # ema_model.update(model)
            # ema_cls.update(classifier_head)

        if args.early_stop or epoch == args.epochs:
            val_acc, val_loss, confusion_matrix = validate(args,
                                                           data_loader=val_loader,
                                                        #    data_loader=test_loader, # note that here i used test loader for validation
                                                           model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        is_encoder=is_encoder,
                                                        )
            scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        # if args.early_stop or epoch == args.epochs:
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                            loss=args.loss, logit_scale=logit_scale,
                            classifier_head=classifier_head,
                            dataset=args.dataset,
                            output_dir=args.output_dir, device=args.device,
                            is_encoder=is_encoder,
                            )

        # ema_test_acc, _, _ = validate(args,data_loader=test_loader, model=ema_model.ema, logger=logger,
        #                     loss=args.loss, logit_scale=logit_scale,
        #                     classifier_head=ema_cls.ema,
        #                     dataset=args.dataset,
        #                     output_dir=args.output_dir, device=args.device,
        #                     is_encoder=is_encoder,
        #                     )

        train_loss_avg = losses.avg
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()

        lr = scheduler.get_last_lr()
        print(lr)
        lr_backbone, lr_classifier = lr[1], lr[0]

        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, lr_bkb: {round(lr_backbone,8)}, lr_cls: {round(lr_classifier,6)}, "
                    f"Trn Loss: {round(train_loss_avg, 6)}, Lx: {round(losses_x.avg, 6)}, Lr: {round(losses_r.avg, 6)}, Lu: {round(losses_u.avg, 6)}, "
                    f"Mask: {round(mask_probs.avg, 6)}, logit_scale_x: {round(logit_scale.item(), 6)}, logit_scale_r: {round(logit_scale_retrieve.item(), 6)}, "
                    f"logit_scale_u: {round(logit_scale_u.item(), 6)}, "
                    f"Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, "
                    f"Test Acc: {round(test_acc, 3)}, EMA Test Acc: {round(ema_test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale



def train_fixmatch(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader, reload_model):
    """
    Train the model with fixmatch SSL method, fewshot + unlabeled data, no CutMix
    Part of the batch from labeled data, part from unlabeled data
    """

    cfgs = args.model_cfg.split('_')
    is_encoder = (cfgs[1] == 'openclip' or cfgs[1] == 'clip')

    if reload_model:
        load_model(args, logger, model, test_loader, classifier_head, is_encoder)
        # load_model(args, logger, model, None, classifier_head, is_encoder)

    train_dataloader, u_train_dataloader = train_loader
    u_train_loader = iter(u_train_dataloader)

    if args.check_logits:
        u_train_dataloader_copy = copy.deepcopy(u_train_dataloader)
        u_train_loader_check = iter(u_train_dataloader_copy)
        u_inputs_check, labels_check, _, _ = next(u_train_loader_check)  # fixed batch

    logger.info(f"Start Training FixMatch ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    logit_scale_u = args.logit_scale_u
    optimizer = args.optimizer
    scheduler = args.scheduler

    val_loss = -1
    val_acc = -1
    test_acc = -1
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    mask_all = AverageMeter()
    impurity_all = AverageMeter()

    for epoch in range(1, args.epochs+1):
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()

        model.train()
        classifier_head.train()

        for inputs_x, targets_x, _, _ in train_dataloader:

            num_iter += 1

            # load a batch of unlabeled data
            try:
                u_inputs, u_labels, _, _ = next(u_train_loader)
            except StopIteration:
                u_train_loader = iter(u_train_dataloader)
                u_inputs, u_labels, _, _ = next(u_train_loader)

            inputs_u_w, inputs_u_s = u_inputs
            u_labels = u_labels.to(args.device)

            batch_size = inputs_x.shape[0]

            # interleave the labeled and unlabeled data to stabilize training
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)

            targets_x = targets_x.to(args.device)

            if is_encoder:
                image_features = model.encode_image(inputs)
                image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            else:
                image_feature = model(inputs)

            logits = classifier_head(image_feature)

            logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            logits_x = logits_x * logit_scale.exp()
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)

            # use the temperature being learned. This is worse than using a fixed temperature Tuw!
            # T_learned = 1.0 / logit_scale_u.exp().detach()
            # logger.info(f'T_learned: {T_learned.item()}')
            # pseudo_label = torch.softmax(logits_u_w.detach()/T_learned, dim=-1)

            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            # calculate the number of incorrect targets_u that does not matches u_labels, with mask applied
            impurity = ((targets_u != u_labels) * mask).sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0.0

            logits_u_s = logits_u_s * logit_scale_u.exp() # scale the logits for unlabeled data
            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()

            total_loss = Lx + args.lambda_u * Lu

            losses.update(total_loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            mask_probs.update(mask.mean().item())
            mask_all.update(mask.mean().item())
            impurity_all.update(impurity)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # break

        if args.check_logits:
            if epoch == 1 or epoch % 5 == 0:

                # we calculate the pseudo-labels and logits for a same batch of unlabeled data
                inputs_u_w_check, inputs_u_s_check = u_inputs_check

                with torch.no_grad():
                    inputs_check = interleave(
                        torch.cat((inputs_u_w_check, inputs_u_s_check)), 2*args.mu).to(args.device)

                    if is_encoder:
                        image_features = model.encode_image(inputs_check)
                        image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
                    else:
                        image_feature = model(inputs_check)

                    logits_check = classifier_head(image_feature)

                    logits_check = de_interleave(logits_check, 2*args.mu)
                    logits_u_w_check, logits_u_s_check = logits_check.chunk(2)
                    del logits_check

                # get the pseudo-labels
                check_results = dict()
                for temp in ['1.0', '0.1', '0.01']:
                    pseudo_label = torch.softmax(logits_u_w_check.detach()/float(temp), dim=-1)
                    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                    mask = max_probs.ge(args.threshold).float()
                    check_results[temp] = dict()
                    check_results[temp]['labels'] = labels_check.tolist()
                    check_results[temp]['pseudo_label'] = pseudo_label.cpu().numpy()
                    check_results[temp]['max_probs'] = max_probs.cpu().numpy()
                    check_results[temp]['targets_u'] = targets_u.cpu().numpy()
                    check_results[temp]['mask'] = mask.cpu().numpy()

                # save the check_results to a pkl file
                if not os.path.exists(os.path.join(args.output_dir+'/check_results/')):
                    os.makedirs(os.path.join(args.output_dir+'/check_results/'))

                with open(os.path.join(args.output_dir+'/check_results/', f'check_results_epoch_{epoch}.pkl'), 'wb') as f:
                    pickle.dump(check_results, f)
                logger.info(f'check_results saved to {os.path.join(args.output_dir, f"check_results_epoch_{epoch}.pkl")}')


        # validate after 1 epoch
        if args.early_stop or epoch == args.epochs:
            val_acc, val_loss, confusion_matrix = validate(args,
                                                           data_loader=val_loader,
                                                        #    data_loader=test_loader, # note that here i used test loader for validation
                                                           model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        is_encoder=is_encoder,
                                                        )
            scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        # if args.early_stop or epoch == args.epochs:
        test_acc, _, _ = validate(args, data_loader=test_loader, model=model, logger=logger,
                            loss=args.loss, logit_scale=logit_scale,
                            classifier_head=classifier_head,
                            dataset=args.dataset,
                            output_dir=args.output_dir, device=args.device,
                            is_encoder=is_encoder,
                            )

        train_loss_avg = losses.avg
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)},{round(mask_probs.avg, 6)}\n')
        loss_logger.flush()

        lr = scheduler.get_last_lr()
        # print(lr)
        lr_backbone, lr_classifier = lr[1], lr[0]

        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, lr_bkb: {round(lr_backbone,8)}, lr_cls: {round(lr_classifier,8)}, "
                    f"Trn Loss: {round(train_loss_avg, 6)}, Lx: {round(losses_x.avg, 6)}, Lu: {round(losses_u.avg, 6)}, "
                    f"Mask: {round(mask_probs.avg, 6)}, Mask_All: {round(mask_all.avg, 4)}, Impurity_All: {round(impurity_all.avg, 4)}, "
                    f"logit_scale_x: {round(logit_scale.item(), 6)}, "
                    f"logit_scale_u: {round(logit_scale_u.item(), 6)}, "
                    f"Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, "
                    f"Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    mask_avg = mask_all.avg
    impurity_avg = impurity_all.avg
    logger.info(f'Mask avg: {round(mask_avg, 4)}, Impurity avg: {round(impurity_avg, 4)}')

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale, mask_avg, impurity_avg




def contrastive_loss(logits_u_w, logits_u_s, logit_scale):
    """
    Compute contrastive InfoNCE loss for weak and strong augmentations.

    Args:
        logits_u_w: Tensor of shape (batch_size, feature_dim) - logits for weak augmentations.
        logits_u_s: Tensor of shape (batch_size, feature_dim) - logits for strong augmentations.
        temperature: Softmax temperature parameter.

    Returns:
        loss: Contrastive loss value.
    """
    # Normalize logits (to get cosine similarity)
    logits_u_w = F.normalize(logits_u_w, dim=1)
    logits_u_s = F.normalize(logits_u_s, dim=1)

    # Compute similarity matrix (batch_size x batch_size)
    logits = torch.matmul(logits_u_w, logits_u_s.T) * logit_scale.exp()

    # Labels: each sample's positive pair is on the diagonal
    batch_size = logits.shape[0]
    labels = torch.arange(batch_size, device=logits.device)  # [0, 1, 2, ..., batch_size-1]

    # Compute cross-entropy loss
    loss = F.cross_entropy(logits, labels)

    return loss


def contrastive_loss_fea(fea_u_w, fea_u_s, logit_scale, normalize=True):
    """
    Compute contrastive InfoNCE loss for weak and strong augmentations.

    Args:
        fea_u_w: Tensor of shape (batch_size, feature_dim) - logits for weak augmentations.
        fea_u_s: Tensor of shape (batch_size, feature_dim) - logits for strong augmentations.
        logit_scale: Softmax temperature parameter.

    Returns:
        loss: Contrastive loss value.
    """

    # Normalize features if needed
    if normalize:
        fea_u_w = F.normalize(fea_u_w, dim=-1, p=2)  # Normalize along feature dimension
        fea_u_s = F.normalize(fea_u_s, dim=-1, p=2)  # Normalize along feature dimension

    # Compute cosine similarity between features from weak and strong augmentations
    logits = torch.matmul(fea_u_w, fea_u_s.T)  # Shape (batch_size, batch_size)

    # Scale the logits by the temperature parameter
    logits = logits * logit_scale.exp()

    # Compute the labels (diagonal is the positive pair)
    labels = torch.arange(fea_u_w.size(0)).to(fea_u_w.device)

    # Cross-entropy loss (InfoNCE)
    loss = F.cross_entropy(logits, labels)

    return loss


def train_fixmatch_contrastive(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader, reload_model):
    """
    Train the model with fixmatch SSL method, fewshot + unlabeled data, no CutMix
    Part of the batch from labeled data, part from unlabeled data
    Add contrastive loss using weak and strong augmentation of unlabeled data
    """

    cfgs = args.model_cfg.split('_')
    is_encoder = (cfgs[1] == 'openclip' or cfgs[1] == 'clip')

    if reload_model:
        load_model(args, logger, model, test_loader, classifier_head, is_encoder)

    train_dataloader, u_train_dataloader = train_loader
    u_train_loader = iter(u_train_dataloader)

    logger.info(f"Start Training FixMatch-Contrastive ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    logit_scale_cons = args.logit_scale_r # here I borrow the logit_scale_r for contrastive loss
    logit_scale_u = args.logit_scale_u
    # loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    val_loss = -1
    val_acc = -1
    test_acc = -1
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0

    # build a mlp projection layer with 1 hiddle layers, relu function
    # proj_layer = ProjectionMLP(512, 128)

    for epoch in range(1, args.epochs+1):
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        losses_c = AverageMeter()
        mask_probs = AverageMeter()

        for inputs_x, targets_x, _, _ in train_dataloader:
            model.train()
            classifier_head.train()

            num_iter += 1

            # load a batch of unlabeled data
            try:
                u_inputs, _, _, _ = next(u_train_loader)
            except StopIteration:
                u_train_loader = iter(u_train_dataloader)
                u_inputs, _, _, _ = next(u_train_loader)

            inputs_u_w, inputs_u_s = u_inputs
            batch_size = inputs_x.shape[0]

            # interleave the labeled and unlabeled data to stabilize training
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)

            targets_x = targets_x.to(args.device)

            if is_encoder:
                image_features = model.encode_image(inputs)
                image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            else:
                image_feature = model(inputs)

            logits = classifier_head(image_feature)

            logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            logits_x = logits_x * logit_scale.exp()
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1) # note the usage of args.T here to scale the logits such that some can go above the confidence threshold
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            logits_u_s = logits_u_s * logit_scale_u.exp() # scale the logits for unlabeled data
            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()

            #---------- add contrastive loss on all unlabeled data in the current batch

            # calculate the contrastive loss using logits_u_w and logits_u_s
            # Lc = contrastive_loss(logits_u_w, logits_u_s, logit_scale_cons)

            # logit_scale_cons = torch.zeros_like(logit_scale).to(args.device)

            # use unnormalized visual features
            image_fea = de_interleave(image_features, 2*args.mu+1)
            # image_fea_x = image_fea[:batch_size]  # weak augmentations' features
            image_fea_w, image_fea_s = image_fea[batch_size:].chunk(2)  # split into weak and strong augmentations for unlabeled data
            # Lc = contrastive_loss_fea(image_fea_w, image_fea_s, logit_scale_cons, True)

            # add a projection layer to the features before computing contrastive loss
            image_fea_w = args.proj_layer(image_fea_w)
            image_fea_s = args.proj_layer(image_fea_s)
            Lc = contrastive_loss_fea(image_fea_w, image_fea_s, logit_scale_cons, True)



            # if epoch <= 20:
            # # if epoch > 20:
            #     image_fea = de_interleave(image_features, 2*args.mu+1)
            #     image_fea_x = image_fea[:batch_size]  # weak augmentations' features
            #     image_fea_w, image_fea_s = image_fea[batch_size:].chunk(2)  # split into weak and strong augmentations for unlabeled data
            #     Lc = contrastive_loss_fea(image_fea_w, image_fea_s, logit_scale_cons, True)
            # else:
            #     Lc = torch.tensor(0.0).cuda()  # for the first 20 epochs, we will not apply contrastive loss, to stabilize training

            total_loss = Lx + args.lambda_u * Lu + args.lambda_contrastive * Lc

            losses.update(total_loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            losses_c.update(Lc.item())
            mask_probs.update(mask.mean().item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # break

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        if args.early_stop or epoch == args.epochs:
            val_acc, val_loss, confusion_matrix = validate(args,
                                                           data_loader=val_loader,
                                                        #    data_loader=test_loader, # note that here i used test loader for validation
                                                           model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        is_encoder=is_encoder,
                                                        )
            scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        # if args.early_stop or epoch == args.epochs:
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                            loss=args.loss, logit_scale=logit_scale,
                            classifier_head=classifier_head,
                            dataset=args.dataset,
                            output_dir=args.output_dir, device=args.device,
                            is_encoder=is_encoder,
                            )

        train_loss_avg = losses.avg
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()

        lr = scheduler.get_last_lr()
        # print(lr)
        lr_backbone, lr_classifier = lr[1], lr[0]

        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, lr_bkb: {round(lr_backbone,8)}, lr_cls: {round(lr_classifier,8)}, "
                    f"Trn Loss: {round(train_loss_avg, 6)}, Lx: {round(losses_x.avg, 6)}, Lu: {round(losses_u.avg, 6)}, Lc: {round(losses_c.avg, 6)}, "
                    f"Mask: {round(mask_probs.avg, 6)}, logit_scale_x: {round(logit_scale.item(), 6)}, "
                    f"logit_scale_u: {round(logit_scale_u.item(), 6)}, logit_scale_c: {round(logit_scale_cons.item(), 6)}, "
                    f"Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, "
                    f"Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale



def train_fixmatch_supervised_contrastive(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader, reload_model):
    """
    Train the model with fixmatch SSL method, fewshot + unlabeled data, no CutMix
    Part of the batch from labeled data, part from unlabeled data
    Add contrastive loss using weak and strong augmentation of unlabeled data
    """

    cfgs = args.model_cfg.split('_')
    is_encoder = (cfgs[1] == 'openclip' or cfgs[1] == 'clip')

    if reload_model:
        # load_model(args, logger, model, test_loader, classifier_head, is_encoder)
        load_model(args, logger, model, None, classifier_head, is_encoder)


    train_dataloader, u_train_dataloader = train_loader
    u_train_loader = iter(u_train_dataloader)

    logger.info(f"Start Training FixMatch + Supervised Contrastive ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    logit_scale_cons = args.logit_scale_r # here I borrow the logit_scale_r for contrastive loss
    logit_scale_u = args.logit_scale_u
    # loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    val_loss = -1
    val_acc = -1
    test_acc = -1
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0

    super_con_loss = SupConLoss()

    for epoch in range(1, args.epochs+1):
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        losses_c = AverageMeter()
        mask_probs = AverageMeter()
        mask_pos_pairs_avg = AverageMeter()

        for inputs_x, targets_x, _, _ in train_dataloader:
            model.train()
            classifier_head.train()

            num_iter += 1

            # load a batch of unlabeled data
            try:
                u_inputs, _, _, _ = next(u_train_loader)
            except StopIteration:
                u_train_loader = iter(u_train_dataloader)
                u_inputs, _, _, _ = next(u_train_loader)

            inputs_u_w, inputs_u_s = u_inputs
            batch_size = inputs_x.shape[0]

            # interleave the labeled and unlabeled data to stabilize training
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)

            targets_x = targets_x.to(args.device)

            if is_encoder:
                image_features = model.encode_image(inputs)
                image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            else:
                image_feature = model(inputs)

            logits = classifier_head(image_feature)

            logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            logits_x = logits_x * logit_scale.exp()
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            # note the usage of args.T here to scale the logits such that some can go above the confidence threshold
            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            logits_u_s = logits_u_s * logit_scale_u.exp() # scale the logits for unlabeled data
            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()

            #---------- add contrastive loss on all unlabeled data in the current batch

            # use normalized visual features
            image_fea = de_interleave(image_feature, 2*args.mu+1)
            # image_fea_x = image_fea[:batch_size]  # labeled images' features
            image_fea_w, image_fea_s = image_fea[batch_size:].chunk(2)  # split into weak and strong augmentations for unlabeled data

            # add a projection layer to the features before computing contrastive loss
            image_fea_w = args.proj_layer(image_fea_w)
            image_fea_s = args.proj_layer(image_fea_s)

            # self-supervised contrastive loss
            # Lc = contrastive_loss_fea(image_fea_w, image_fea_s, logit_scale_cons, True)

            # supervised contrastive loss, using the pseudo-labels from weak augmentations
            Lc, mask_pos_pairs = super_con_loss(image_fea_w, image_fea_s, logit_scale_cons, targets_u, mask)

            total_loss = Lx + args.lambda_u * Lu + args.lambda_contrastive * Lc

            losses.update(total_loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            losses_c.update(Lc.item())
            mask_probs.update(mask.mean().item())
            mask_pos_pairs_avg.update(mask_pos_pairs.mean().item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # stop
            # break

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        if args.early_stop or epoch == args.epochs:
            val_acc, val_loss, confusion_matrix = validate(args,
                                                           data_loader=val_loader,
                                                        #    data_loader=test_loader, # note that here i used test loader for validation
                                                           model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        is_encoder=is_encoder,
                                                        )
            scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        # if args.early_stop or epoch == args.epochs:
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                            loss=args.loss, logit_scale=logit_scale,
                            classifier_head=classifier_head,
                            dataset=args.dataset,
                            output_dir=args.output_dir, device=args.device,
                            is_encoder=is_encoder,
                            )

        train_loss_avg = losses.avg
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()

        lr = scheduler.get_last_lr()
        # print(lr)
        lr_backbone, lr_classifier = lr[1], lr[0]

        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, lr_bkb: {round(lr_backbone,8)}, lr_cls: {round(lr_classifier,8)}, "
                    f"Trn Loss: {round(train_loss_avg, 6)}, Lx: {round(losses_x.avg, 6)}, Lu: {round(losses_u.avg, 6)}, Lc: {round(losses_c.avg, 6)}, "
                    f"Mask: {round(mask_probs.avg, 6)}, logit_scale_x: {round(logit_scale.item(), 6)}, "
                    f"logit_scale_u: {round(logit_scale_u.item(), 6)}, logit_scale_c: {round(logit_scale_cons.item(), 6)}, "
                    f"Mask Pos Pairs: {round(mask_pos_pairs_avg.avg, 6)}, "
                    f"Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, "
                    f"Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale


def train_fixmatch_cutmix(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader, reload_model):
    """
    Train the model with fixmatch SSL method + CutMix
    Part of the batch from labeled data, part from unlabeled data
    """

    cfgs = args.model_cfg.split('_')
    is_encoder = (cfgs[1] == 'openclip' or cfgs[1] == 'clip')

    if reload_model:
        load_model(args, logger, model, test_loader, classifier_head, is_encoder)
        # load_model(args, logger, model, None, classifier_head, is_encoder)

    train_dataloader, u_train_dataloader = train_loader
    u_train_loader = iter(u_train_dataloader)

    logger.info(f"Start Training FixMatch-Joint ......")

    model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    logit_scale_u = args.logit_scale_u
    # loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    val_loss = -1
    val_acc = -1
    test_acc = -1
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0

    for epoch in range(1, args.epochs+1):
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        model.train()
        classifier_head.train()

        for inputs_x, targets_x, _, _ in train_dataloader:

            num_iter += 1

            # load a batch of unlabeled data
            try:
                u_inputs, _, _, _ = next(u_train_loader)
            except StopIteration:
                u_train_loader = iter(u_train_dataloader)
                u_inputs, _, _, _ = next(u_train_loader)

            inputs_u_w, inputs_u_s = u_inputs

            batch_size = inputs_x.shape[0]

            # apply CutMix before interleaving
            images = inputs_x
            labels = targets_x
            r = np.random.rand(1)
            if args.cutmix_beta > 0 and r < args.mix_prob:
                # generate mixed sample
                lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
                # rand_index = torch.randperm(images.size()[0]).cuda()
                rand_index = torch.randperm(images.size()[0])
                target_a = labels
                target_b = labels[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            else:
                target_a = labels
                target_b = labels
                lam = 1.0

            # assign the cutmixed images to inputs_x
            inputs_x = images

            # interleave the labeled and unlabeled data to stabilize training
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)

            # targets_x = targets_x.to(args.device)

            if is_encoder:
                image_features = model.encode_image(inputs)
                image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            else:
                image_feature = model(inputs)

            logits = classifier_head(image_feature)

            logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            logits_x = logits_x * logit_scale.exp()
            # Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
            target_a = target_a.to(args.device)
            target_b = target_b.to(args.device)

            Lx = mixup_criterion(F.cross_entropy, logits_x, target_a, target_b, lam) # use mixup loss

            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            logits_u_s = logits_u_s * logit_scale_u.exp() # scale the logits for unlabeled data
            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()

            total_loss = Lx + args.lambda_u * Lu

            losses.update(total_loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            mask_probs.update(mask.mean().item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # break # for fast debugging

        # validate after 1 epoch
        if args.early_stop or epoch == args.epochs:
            val_acc, val_loss, confusion_matrix = validate(args,
                                                           data_loader=val_loader,
                                                           model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        is_encoder=is_encoder,
                                                        )
            scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        # if args.early_stop or epoch == args.epochs:
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                            loss=args.loss, logit_scale=logit_scale,
                            classifier_head=classifier_head,
                            dataset=args.dataset,
                            output_dir=args.output_dir, device=args.device,
                            is_encoder=is_encoder,
                            )

        train_loss_avg = losses.avg
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()

        lr = scheduler.get_last_lr()
        # print(lr)
        lr_backbone, lr_classifier = lr[1], lr[0]

        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, lr_bkb: {round(lr_backbone,6)}, lr_cls: {round(lr_classifier,4)}, "
                    f"Trn Loss: {round(train_loss_avg, 6)}, Lx: {round(losses_x.avg, 6)}, Lu: {round(losses_u.avg, 6)}, "
                    f"Mask: {round(mask_probs.avg, 6)}, logit_scale_x: {round(logit_scale.item(), 6)}, "
                    f"logit_scale_u: {round(logit_scale_u.item(), 6)}, "
                    f"Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, "
                    f"Test Acc: {round(test_acc, 3)}")   

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale




def retrain_classifier(args, model, classifier_head):

    model.eval()
    classifier_head.train()

    # reextract the few-shot features using the updated visual encoder
    new_fewshot_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_fewshot_features_stage2.pth'
    new_test_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_test_features_stage2.pth'

    train_loader = extract_train_dataloader(args, model, args.fewshot_data, new_fewshot_fea_path,
                                            args.preprocess, None, args.bsz)
    test_loader = extract_dataloader(args, model, args.test_split, new_test_fea_path,
                                     args.preprocess, None)


    # Imporatnt! Need to reset the params, optimizer, scheduler, loss, logit_scale
    loss = set_loss(args)

    params_classifier = [{'params': classifier_head.parameters(), 'lr': args.lr_classifier}]
    for param in model.parameters():
        param.requires_grad = False
    params = params_classifier
    logit_scale = torch.tensor([4.60517]).to(device=args.device)

    optimizer, scheduler, total_iter = set_optimizer(args, params, train_loader)

    args.pre_extracted = True
    num_epoch = 10

    # retrain the classifier
    val_loss = -1
    val_acc = -1
    test_acc = -1
    num_iter = 0

    for epoch in range(1, num_epoch+1):

        train_loss_sum = 0
        for inputs, labels, _, _ in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()
            image_feature = images

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()
            total_loss = loss(logits, labels)
            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()


        # test after 1 epoch
        # if epoch == args.epochs:
        test_acc, _, _ = validate(args, data_loader=test_loader, model=model, logger=args.logger,
                            loss=args.loss, logit_scale=logit_scale,
                            classifier_head=classifier_head,
                            dataset=args.dataset,
                            output_dir=args.output_dir, device=args.device,
                            )

        train_loss_avg = train_loss_sum / len(train_loader)

        args.logger.info(f"      Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

    os.remove(new_fewshot_fea_path)
    os.remove(new_test_fea_path)

    # reset args.pre_extracted
    args.pre_extracted = False
    model.train()
    for param in model.parameters():
        param.requires_grad = True

    return classifier_head, test_acc




def train_fixmatch_retraincls(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader, reload_model):
    """
    Train the model with fixmatch SSL method.
    Part of the batch from labeled data, part from unlabeled data
    retrain the classifier using the balanced few-shot data after every epoch
    """

    cfgs = args.model_cfg.split('_')
    is_encoder = (cfgs[1] == 'openclip' or cfgs[1] == 'clip')

    if reload_model:
        load_model(args, logger, model, test_loader, classifier_head, is_encoder)

    train_dataloader, u_train_dataloader = train_loader
    # print(f'train_dataloader: {len(train_dataloader)}')
    # print(f'u_train_dataloader: {len(u_train_dataloader)}')
    # train_loader = iter(train_dataloader)
    u_train_loader = iter(u_train_dataloader)

    logger.info(f"Start FixMatch Training ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    logit_scale_u = args.logit_scale_u
    # loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    val_loss = -1
    val_acc = -1
    test_acc = -1
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0

    for epoch in range(1, args.epochs+1):
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()

        # for u_inputs, _, _, _ in u_train_dataloader:
        for inputs_x, targets_x, _, _ in train_dataloader:

            num_iter += 1

            # load a batch of labeled data
            # try:
            #     inputs_x, targets_x, _, _ = next(train_loader)
            # except StopIteration:
            #     train_loader = iter(train_dataloader)
            #     inputs_x, targets_x, _, _ = next(train_loader)

            # load a batch of unlabeled data
            try:
                u_inputs, _, _, _ = next(u_train_loader)
            except StopIteration:
                u_train_loader = iter(u_train_dataloader)
                u_inputs, _, _, _ = next(u_train_loader)


            inputs_u_w, inputs_u_s = u_inputs

            batch_size = inputs_x.shape[0]
            # print(batch_size)

            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)

            targets_x = targets_x.to(args.device)

            if is_encoder:
                image_features = model.encode_image(inputs)
                image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            else:
                image_feature = model(inputs)

            logits = classifier_head(image_feature)

            logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            logits_x = logits_x * logit_scale.exp()
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            logits_u_s = logits_u_s * logit_scale_u.exp() # scale the logits for unlabeled data
            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()

            total_loss = Lx + args.lambda_u * Lu

            losses.update(total_loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            mask_probs.update(mask.mean().item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            args.retrain_freq = 300
            if num_iter % args.retrain_freq == 0:
                # retrain the classifier after every epoch using the balanced few-shot data
                classifier_head, test_acc_new = retrain_classifier(args, model, classifier_head)

        # validate after 1 epoch
        if args.early_stop or epoch == args.epochs:
            val_acc, val_loss, confusion_matrix = validate(args,
                                                        #    data_loader=val_loader,
                                                           data_loader=test_loader, # note that here i used test loader for validation
                                                           model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        is_encoder=is_encoder,
                                                        )
            scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        # if args.early_stop or epoch == args.epochs:
        test_acc, _, _ = validate(args, data_loader=test_loader, model=model, logger=logger,
                            loss=args.loss, logit_scale=logit_scale,
                            classifier_head=classifier_head,
                            dataset=args.dataset,
                            output_dir=args.output_dir, device=args.device,
                            is_encoder=is_encoder,
                            )

        # retrain the classifier after every epoch using the balanced few-shot data
        # classifier_head, test_acc_new = retrain_classifier(args, model, classifier_head)


        train_loss_avg = losses.avg
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)},{round(test_acc_new, 6)}\n')
        loss_logger.flush()

        lr = scheduler.get_last_lr()
        # print(lr)
        lr_backbone, lr_classifier = lr[1], lr[0]

        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, lr_bkb: {round(lr_backbone,6)}, lr_cls: {round(lr_classifier,4)}, "
                    f"Trn Loss: {round(train_loss_avg, 6)}, Lx: {round(losses_x.avg, 6)}, Lu: {round(losses_u.avg, 6)}, "
                    f"Mask: {round(mask_probs.avg, 6)}, logit_scale_x: {round(logit_scale.item(), 6)}, "
                    f"logit_scale_u: {round(logit_scale_u.item(), 6)}, "
                    f"Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, "
                    f"Test Acc: {round(test_acc, 3)}, "
                    f"Test Acc retrain cls: {round(test_acc_new, 3)}, "
                    )

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale





def train_ce_multitask(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader, dataset_classifier_head):
    """ Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier"""

    logger.info(f"Start Training ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()
    dataset_classifier_head.train()

    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    dataset_weight = 1.0
    for epoch in range(1, args.epochs+1):
        # dataset_weight *= args.dataset_wd # decay the dataset weight for each epoch
        dataset_weight = args.dataset_wd

        train_loss_sum = 0
        dataset_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()
            source = source.to(args.device)

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()
            if args.loss_name == 'WeightedCE':
                total_loss = loss(logits, labels, source) # for WeightedCE, needs to input the source
            else:
                total_loss = loss(logits, labels)

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            # add the dataset classification loss
            dataset_logits = dataset_classifier_head(image_feature)
            # dataset_logits = dataset_logits * logit_scale.exp()
            dataset_loss = loss(dataset_logits, source)
            dataset_loss_sum += dataset_loss.item()

            multitask_loss = total_loss + dataset_loss * dataset_weight

            optimizer.zero_grad()
            multitask_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        val_acc, val_loss, confusion_matrix, dataset_val_acc, dataset_confusion_matrix = validate_multitask(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, dataset_classifier_head=dataset_classifier_head,
                                                        show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,

                                                        )
        scores = calculate_scores(confusion_matrix)
        # dataset_scores = calculate_scores(dataset_confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _, dataset_test_acc, _ = validate_multitask(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head, dataset_classifier_head=dataset_classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,

                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        dataset_loss_avg = dataset_loss_sum / len(train_loader)

        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Dataset Loss: {round(dataset_loss_avg, 6)}, weight: {round(dataset_weight, 3)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Dataset Val Acc: {round(dataset_val_acc, 3)}, Test Acc: {round(test_acc, 3)}, Dataset Test Acc: {round(dataset_test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale

def mixup_data(x, y, alpha=1.0, mix_prob=0.5):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    # samply a number uniformly from [0, 1],
    # this includes the clean/unmixed images to the training process
    flag = torch.rand(1).item()
    if flag <= mix_prob: # do mixup
        lam = lam
    else: # do not mixup
        lam = 1.0

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_data_fs(x_retr, y_retr, x_fs, y_fs, alpha=1.0, mix_prob=0.5):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    # samply a number uniformly from [0, 1],
    # this includes the clean/unmixed images to the training process
    flag = torch.rand(1).item()
    if flag <= mix_prob:
        lam = 0.0 # set to 0.0 to use few-shot data only
    else: # do not mixup
        lam = 1.0

    mixed_x = lam * x_retr + (1.0 - lam) * x_fs
    y_a, y_b = y_retr, y_fs

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)

def mixup_criterion_lam_list(criterion, pred, y_a, y_b, lam_list):
    # each value in lam_list is the lambda value for each image in the batch
    return sum([lam_list[i] * criterion(pred[i], y_a[i]) + (1.0 - lam_list[i]) * criterion(pred[i], y_b[i]) for i in range(len(lam_list))])

def train_mixup(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use Mixup method to augment the training data
    """

    logger.info(f"Start Training mixup ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()
            # source = source.to(args.device)

            # apply the mixup strategy
            images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=args.mixup_alpha, mix_prob=args.mix_prob)

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss
            total_loss = mixup_criterion(loss, logits, targets_a, targets_b, lam)

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,

                                                        )
        scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,

                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale


def train_mixup_fs(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use Mixup method to augment the training data
    """
    train_loader, train_dataloader_fs = train_loader
    train_loader_fs = iter(train_dataloader_fs)

    logger.info(f"Start Training ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader: # this is still the mixed data
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            # get a batch of few-shot data, handle the case when the few-shot data is exhausted, just loop back
            try:
                inputs_fs, labels_fs, text_fs, source_fs = next(train_loader_fs)
            except StopIteration:
                train_loader_fs = iter(train_dataloader_fs)
                inputs_fs, labels_fs, text_fs, source_fs = next(train_loader_fs)
            images_fs = inputs_fs.to(args.device)
            labels_fs = labels_fs.to(args.device).long()

            # apply the mixup strategy
            images, targets_a, targets_b, lam = mixup_data_fs(images, labels, images_fs, labels_fs,
                                                           alpha=args.mixup_alpha, mix_prob=args.mix_prob)

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss
            total_loss = mixup_criterion(loss, logits, targets_a, targets_b, lam)

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,

                                                        )
        scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,

                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def train_cutmix(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader, reload_model=False):
    """
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use CutMix method to augment the training data
    """

    if reload_model:
        load_model(args, logger, model, test_loader, classifier_head)

    logger.info(f"Start Training cutmix ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    val_loss = -1
    val_acc = -1
    test_acc = -1
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    # check if the model_cfg is openclip
    cfgs = args.model_cfg.split('_')
    is_encoder = (cfgs[1] == 'openclip' or cfgs[1] == 'clip')

    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True).long()

            # apply the cutmix strategy
            r = np.random.rand(1)
            if args.cutmix_beta > 0 and r < args.mix_prob:
                # generate mixed sample
                lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            else:
                target_a = labels
                target_b = labels
                lam = 1.0

            if is_encoder:
                image_features = model.encode_image(images)
                image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            else:
                image_feature = model(images)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss
            total_loss = mixup_criterion(loss, logits, target_a, target_b, lam)

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        if args.early_stop or epoch == args.epochs:
            val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        is_encoder=is_encoder
                                                        )
            scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        # if args.early_stop or epoch == args.epochs:
        # test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
        #                     loss=args.loss, logit_scale=logit_scale,
        #                     classifier_head=classifier_head,
        #                     dataset=args.dataset,
        #                     output_dir=args.output_dir, device=args.device,
        #                     is_encoder=is_encoder,
        #                     )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()

        lr = scheduler.get_last_lr()
        print(lr)
        lr_backbone, lr_classifier = lr[1], lr[0]
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, lr_bkb: {round(lr_backbone, 8)}, lr_cls: {round(lr_classifier, 8)}, "
                    f"Trn Loss: {round(train_loss_avg, 6)}, "
                    f"logit_scale: {round(logit_scale.item(), 6)}, "
                    f"Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, "
                    f"Test Acc: {round(test_acc, 3)}")

        # logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale


def add_moco_params(args, model):
    # linear scaling rule based on MoCo v3
    moco_lr = args.moco_lr * args.moco_bsz / 256

    # for param in model.parameters():
    #     param.requires_grad = False

    # turn on the visual encoder, projection head, and prediction head
    # for param in model.encoder_q.visual.parameters():
    #     param.requires_grad = True

    # make sure the free the patch embedding according to MoCo v3
    # model.encoder_q.visual.conv1.weight.requires_grad = False

    for param in model.encoder_q_projector.parameters():
        param.requires_grad = True

    for param in model.encoder_k_projector.parameters():
        param.requires_grad = True

    for param in model.predictor.parameters():
        param.requires_grad = True

    # params_visual = [{'params': model.encoder_q.visual.parameters(), 'lr': moco_lr}]
    params_q_projector = [{'params': model.encoder_q_projector.parameters(), 'lr': moco_lr}]
    params_k_projector = [{'params': model.encoder_k_projector.parameters(), 'lr': moco_lr}]
    params_predictor = [{'params': model.predictor.parameters(), 'lr': moco_lr}]

    params = args.params + params_q_projector + params_k_projector + params_predictor

    return params

def train_moco_cutmix(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader, reload_model=False):
    """
    Train the model with Cross-Entropy Loss + MoCo contrastive loss
    Finetuning visual encoder and classifier with CutMix method to augment the training data

    """

    if reload_model:
        load_model(args, logger, model, test_loader, classifier_head)

    logger.info(f"Start Training MoCo + Cutmix ......")

    model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    logit_scale_u = args.logit_scale_u
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    val_loss = -1
    val_acc = -1
    test_acc = -1
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    # check if the model_cfg is openclip
    cfgs = args.model_cfg.split('_')
    is_encoder = (cfgs[1] == 'openclip' or cfgs[1] == 'clip')

    # build moco model
    moco_model = MoCo_ViT(
        base_encoder=model,
        dim=256, # output dimension
        K=args.moco_k, # based on https://arxiv.org/abs/2104.00679
        T=1.0, # set to 1 as we will use logit_scale_u later
        # mlp=False, # we use the same head as in ViT.head for joint training
    )
    moco_model = moco_model.to(args.device)

    # reinitialize the parameters, optimizer for moco model
    params = add_moco_params(args, moco_model)
    optimizer, scheduler, _ = set_optimizer(args, params, train_loader, args.moco_wd)
    moco_criterion = nn.CrossEntropyLoss().cuda(args.device)

    train_dataloader, u_train_dataloader = train_loader
    u_train_loader = iter(u_train_dataloader)
    iters_per_epoch = len(train_dataloader)

    for epoch in range(1, args.epochs+1):

        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()

        for i, (inputs, labels, _, _) in enumerate(train_dataloader):
            num_iter += 1
            images = inputs.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True).long()

            # load a batch of unlabeled data
            try:
                u_inputs, _, _, _ = next(u_train_loader)
            except StopIteration:
                u_train_loader = iter(u_train_dataloader)
                u_inputs, _, _, _ = next(u_train_loader)
            query_images = u_inputs[0].to(args.device, non_blocking=True)
            key_images = u_inputs[1].to(args.device, non_blocking=True)

            # apply the cutmix strategy
            r = np.random.rand(1)
            if args.cutmix_beta > 0 and r < args.mix_prob:
                # generate mixed sample
                lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            else:
                target_a = labels
                target_b = labels
                lam = 1.0

            if is_encoder:
                image_features = model.encode_image(images)
                image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            else:
                image_feature = model(images)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss
            Lx = mixup_criterion(loss, logits, target_a, target_b, lam)

            #--------- MoCo loss
            moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args)
            moco_logit, moco_target = moco_model(im_q=query_images, im_k=key_images, m=moco_m)
            moco_logit = moco_logit * logit_scale_u.exp()
            Lmoco = moco_criterion(moco_logit, moco_target)

            # Lmoco = torch.tensor(0.0).cuda() # for sanity check

            #--------- total loss
            total_loss = Lx + args.moco_loss_weight * Lmoco

            losses.update(total_loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lmoco.item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

            # break # break the loop for fast debugging

        # validate after 1 epoch
        if args.early_stop or epoch == args.epochs:
            val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        is_encoder=is_encoder
                                                        )
            scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        # if args.early_stop or epoch == args.epochs:
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                            loss=args.loss, logit_scale=logit_scale,
                            classifier_head=classifier_head,
                            dataset=args.dataset,
                            output_dir=args.output_dir, device=args.device,
                            is_encoder=is_encoder,
                            )

        train_loss_avg = losses.avg
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()

        lr = scheduler.get_last_lr()
        # print(lr)
        lr_backbone, lr_classifier = lr[1], lr[0]
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, lr_bkb: {round(lr_backbone, 8)}, lr_cls: {round(lr_classifier, 8)}, "
                    f"Trn Loss: {round(train_loss_avg, 6)}, Lx: {round(losses_x.avg, 6)}, L_moco: {round(losses_u.avg, 6)}, "
                    f"logit_scale_x: {round(logit_scale.item(), 6)}, logit_scale_u_moco: {round(logit_scale_u.item(), 6)}, "
                    f"Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, "
                    f"Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale




def get_moco_params(args, model):
    # linear scaling rule based on MoCo v3
    moco_lr = args.moco_lr * args.moco_bsz / 256

    for param in model.parameters():
        param.requires_grad = False

    # turn on the visual encoder, projection head, and prediction head
    for param in model.encoder_q.visual.parameters():
        param.requires_grad = True
    # make sure the free the patch embedding according to MoCo v3
    model.encoder_q.visual.conv1.weight.requires_grad = False

    for param in model.encoder_q_projector.parameters():
        param.requires_grad = True

    for param in model.encoder_k_projector.parameters():
        param.requires_grad = True

    for param in model.predictor.parameters():
        param.requires_grad = True

    params_visual = [{'params': model.encoder_q.visual.parameters(), 'lr': moco_lr}]
    params_q_projector = [{'params': model.encoder_q_projector.parameters(), 'lr': moco_lr}]
    params_k_projector = [{'params': model.encoder_k_projector.parameters(), 'lr': moco_lr}]
    params_predictor = [{'params': model.predictor.parameters(), 'lr': moco_lr}]

    params = params_visual + params_q_projector + params_k_projector + params_predictor

    return params

def train_moco(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader, reload_model=False):
    """
    Train the model with MoCo contrastive loss
    """

    if reload_model:
        load_model(args, logger, model, test_loader, classifier_head)

    logger.info(f"Start Training MoCo ......")

    model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    logit_scale_u = args.logit_scale_u
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    val_loss = -1
    val_acc = -1
    test_acc = -1
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    # check if the model_cfg is openclip
    cfgs = args.model_cfg.split('_')
    is_encoder = (cfgs[1] == 'openclip' or cfgs[1] == 'clip')

    # build moco model
    moco_model = MoCo_ViT(
        base_encoder=model,
        dim=256, # output dimension
        K=args.moco_k, # based on https://arxiv.org/abs/2104.00679
        T=1.0, # set to 1 as we will use logit_scale_u later
        # mlp=False, # we use the same head as in ViT.head for joint training
    )
    moco_model = moco_model.to(args.device)

    # reinitialize the parameters, optimizer for moco model
    params = get_moco_params(args, moco_model)
    optimizer, scheduler, _ = set_optimizer(args, params, train_loader, args.moco_wd)

    moco_criterion = nn.CrossEntropyLoss().cuda(args.device)

    u_train_dataloader = train_loader
    iters_per_epoch = len(u_train_dataloader)

    for epoch in range(1, args.epochs+1):

        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()

        for i, (u_inputs, _, _, _) in enumerate(u_train_dataloader):
            num_iter += 1

            query_images = u_inputs[0].to(args.device, non_blocking=True)
            key_images = u_inputs[1].to(args.device, non_blocking=True)

            #--------- MoCo loss
            moco_m = adjust_moco_momentum((epoch-1) + i / iters_per_epoch, args)
            # logger.info(f'epoch: {epoch}, i: {i}, moco_m: {moco_m}')
            # moco_m= 0.999

            moco_logit, moco_target = moco_model(im_q=query_images, im_k=key_images, m=moco_m)
            moco_logit = moco_logit * logit_scale_u.exp()
            Lmoco = moco_criterion(moco_logit, moco_target)

            #--------- total loss
            total_loss = Lmoco

            losses.update(total_loss.item())
            # losses_x.update(Lx.item())
            losses_u.update(Lmoco.item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

            # break # break after 1 iteration for quick debugging

        # validate
        if args.early_stop or epoch == args.epochs:
            val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,
                                                        is_encoder=is_encoder
                                                        )
            scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        # if args.early_stop or epoch == args.epochs:
        # test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
        #                     loss=args.loss, logit_scale=logit_scale,
        #                     classifier_head=classifier_head,
        #                     dataset=args.dataset,
        #                     output_dir=args.output_dir, device=args.device,
        #                     is_encoder=is_encoder,
        #                     )


        train_loss_avg = losses.avg
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()

        lr = scheduler.get_last_lr()
        # print(lr)
        lr_backbone, lr_projector = lr[0], lr[1]
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, lr_bkb: {round(lr_backbone, 8)}, lr_cls: {round(lr_projector, 8)}, "
                    f"Trn Loss: {round(train_loss_avg, 6)}, moco_m: {round(moco_m, 6)}, L_moco: {round(losses_u.avg, 6)}, "
                    f"logit_scale_x: {round(logit_scale.item(), 6)}, logit_scale_u_moco: {round(logit_scale_u.item(), 6)}, "
                    f"Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, "
                    f"Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale


def train_cutmix_fs(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use CutMix method to augment the training data
    two dataloader, one from the mixed, one from the few-shot only
    """

    train_loader, train_dataloader_fs = train_loader
    train_loader_fs = iter(train_dataloader_fs)

    logger.info(f"Start Training cutmix-fs ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            # get a batch of few-shot data, when the few-shot data is exhausted, just loop back
            try:
                inputs_fs, labels_fs, text_fs, source_fs = next(train_loader_fs)
            except StopIteration:
                train_loader_fs = iter(train_dataloader_fs)
                inputs_fs, labels_fs, text_fs, source_fs = next(train_loader_fs)
            images_fs = inputs_fs.to(args.device)
            labels_fs = labels_fs.to(args.device).long()

            # apply the cutmix strategy
            r = np.random.rand(1)
            if args.cutmix_beta > 0 and r < args.mix_prob:
                # generate mixed sample
                lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
                target_a = labels
                target_b = labels_fs
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images_fs[:, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            else:
                target_a = labels
                target_b = labels_fs
                lam = 1.0

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss
            total_loss = mixup_criterion(loss, logits, target_a, target_b, lam)

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,

                                                        )
        scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,

                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale


def train_CMO(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use CMO method to augment the training data
    """
    train_loader, weighted_train_loader = train_loader
    inverse_iter = iter(weighted_train_loader)

    logger.info(f"Start Training CMO ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:

            try:
                inputs2, targets2, text2, source2 = next(inverse_iter)
            except:
                inverse_iter = iter(weighted_train_loader)
                inputs2, targets2, text2, source2 = next(inverse_iter)

            inputs2 = inputs2.to(args.device)
            targets2 = targets2.to(args.device).long()

            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            # apply the cutmix strategy
            r = np.random.rand(1)
            if args.cutmix_beta > 0 and r < args.mix_prob:
                # generate mixed sample
                lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = inputs2[:, :, bbx1:bbx2, bby1:bby2]

                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                target_a = labels
                target_b = targets2

            else:
                target_a = labels
                target_b = labels
                lam = 1.0

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss
            total_loss = mixup_criterion(loss, logits, target_a, target_b, lam)

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate
        if args.early_stop or epoch == args.epochs:
            val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,

                                                        )
            scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        if args.early_stop or epoch == args.epochs:
            test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,

                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale

def saliency_bbox(img, lam):
    size = img.size()
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # initialize OpenCV's static fine grained saliency detector and compute the saliency map
    img_arr = img.cpu().numpy().transpose(1, 2, 0)
    img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())
    img_arr = (img_arr * 255).astype(np.uint8)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(img_arr)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
    x = maximum_indices[0]
    y = maximum_indices[1]

    # centered around the peak saliency
    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def train_saliencymix(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use SaliencyMix method to augment the training data
    """

    logger.info(f"Start Training ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            # apply the saliencymix strategy
            r = np.random.rand(1)
            if args.cutmix_beta > 0 and r < args.mix_prob:
                # generate mixed sample
                lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]

                # old implementation
                bbx1, bby1, bbx2, bby2 = saliency_bbox(images[rand_index[0]], lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]

                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            else:
                target_a = labels
                target_b = labels
                lam = 1.0

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss
            total_loss = mixup_criterion(loss, logits, target_a, target_b, lam)

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,

                                                        )
        scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,

                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale

def saliency_bbox_list(imgs, lam):
    corners_list = []
    for img in imgs:
        bbx1, bby1, bbx2, bby2 = saliency_bbox(img, lam)
        corners_list.append([bbx1, bby1, bbx2, bby2])
    return corners_list

def train_saliencymix2(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use SaliencyMix method to augment the training data
    +++++ here we compute the saliency for each image and apply the saliency to the image +++++
    """

    logger.info(f"Start Training saliencymix2 ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            # apply the saliencymix strategy
            r = np.random.rand(1)
            if args.cutmix_beta > 0 and r < args.mix_prob:
                # generate mixed sample
                lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]
                shuffled_images = images[rand_index]

                corners_list = saliency_bbox_list(shuffled_images, lam)
                lam_list = []

                for i in range(images.size(0)):
                    bbx1, bby1, bbx2, bby2 = corners_list[i]
                    images[i, :, bbx1:bbx2, bby1:bby2] = shuffled_images[i, :, bbx1:bbx2, bby1:bby2]
                    # adjust lambda to exactly match pixel ratio
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                    lam_list.append(lam)

            else:
                target_a = labels
                target_b = labels
                lam_list = [1.0] * images.size(0)

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss
            total_loss = mixup_criterion_lam_list(loss, logits, target_a, target_b, lam_list)

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        if args.early_stop or epoch == args.epochs:
            val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,

                                                        )
            scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,

                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale


def train_resizemix(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use ResizeMix method to augment the training data
    """

    logger.info(f"Start Training resizemix ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            # apply the resizemix strategy
            r = np.random.rand(1)
            if r < args.mix_prob:
                # generate mixed sample
                # uniform sampling the lambda from [0.1, 0.8]
                tau = np.random.uniform(0.1, 0.8)

                rand_index = torch.randperm(images.size()[0]).cuda()
                shuffled_images = images[rand_index]
                # print('shuffled_images.size()', shuffled_images.size())
                # resize the shuffled_images to a smaller size of the original images, with scale tau
                resized_images = F.interpolate(shuffled_images, scale_factor=tau, mode='bilinear', align_corners=False)
                # print('resized_images.size()', resized_images.size())

                # get the size of the resized_images
                resized_w = resized_images.size()[-1]
                resized_h = resized_images.size()[-2]

                # get the random position to paste the resized_images
                pos_x = np.random.randint(0, images.size()[-1] - resized_w)
                pos_y = np.random.randint(0, images.size()[-2] - resized_h)

                # paste the resized_images to the original images
                images[:, :, pos_x:pos_x+resized_w, pos_y:pos_y+resized_h] = resized_images

                # adjust lambda to exactly match pixel ratio
                lam = 1.0 - (resized_w * resized_h / (images.size()[-1] * images.size()[-2]))

                # labels
                target_a = labels
                target_b = labels[rand_index]
            else:
                target_a = labels
                target_b = labels
                lam = 1.0

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss, lam belongs to target_a
            total_loss = mixup_criterion(loss, logits, target_a, target_b, lam)

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,

                                                        )
        scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,

                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale


def get_GEM_heatmap(gem_model, images, texts):
    heatmap_list = gem_model.batched_forward(images, texts)
    return heatmap_list


def saliency_bbox_gem(heatmap, lam):
    # convert heatmap from size [1, W, H] to [W, H]
    # detach heatmap to cpu first
    # print('heatmap.size()', heatmap.size())
    heatmap= heatmap.squeeze(0).cpu()

    # print('heatmap.size()', heatmap.size())
    # convert heatmap from torch tensor to numpy array
    heatmap = heatmap.detach().numpy()

    size = heatmap.shape
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # use heatmap asthe saliencymap
    maximum_indices = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
    x = maximum_indices[0]
    y = maximum_indices[1]

    # centered around the peak saliency
    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def saliency_bbox_list_gem(shuffled_images, lam, gem_model, texts):

    corners_list = []
    heatmap_list = get_GEM_heatmap(gem_model, shuffled_images, texts)
    # print(shuffled_images[0].size())
    # print(heatmap_list[0].size())
    for heatmap in heatmap_list:
        bbx1, bby1, bbx2, bby2 = saliency_bbox_gem(heatmap, lam)
        corners_list.append([bbx1, bby1, bbx2, bby2])
    return corners_list

def train_attentivemix(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use AttentiveMix method to augment the training data
    """

    # initialize GEM model
    model_name = 'ViT-B/32'
    pretrained = 'laion400m_e32'
    gem_model = create_gem_model(model_name=model_name, pretrained=pretrained, device=args.device)
    gem_model.eval()
    logger.info(f'GEM model loaded from {model_name} {pretrained}')
    threshold = args.attentive_threshold

    # get the label names dict
    metric_fn = f'{args.dataset_root}/id_scname_dict.json'
    with open(metric_fn, 'r') as f:
        metrics = json.load(f)

    logger.info(f"Start Training ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            # apply the attentivemix strategy
            r = np.random.rand(1)
            if r < args.mix_prob:

                # generate mixed sample
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]
                shuffled_images = images[rand_index]

                # use general domain name here
                if args.attentive_name == 'general':
                    texts = [['bird'] for i in range(images.size()[0])]
                elif args.attentive_name == 'c-name':
                    # use c-name to get the heatmap
                    texts = []
                    for label in target_b:
                        texts.append([metrics[str(label.item())][1]])
                elif args.attentive_name == 's-name':
                    texts = []
                    for label in target_b:
                        texts.append([metrics[str(label.item())][0]])
                # print(target_b)
                # print(texts)

                heatmap_list = get_GEM_heatmap(gem_model, shuffled_images, texts)

                # get the binary_mask_list using the threshold
                binary_mask_list = []
                for heatmap in heatmap_list:
                    binary_mask = (heatmap > threshold).int()
                    binary_mask_list.append(binary_mask)

                # build the new image by attentively mixing the images usingt he binary_mask
                lam_list = []
                for i, binary_mask in enumerate(binary_mask_list):
                    images[i] = images[i] * (1 - binary_mask) + shuffled_images[i] * binary_mask
                    lam = 1.0 - binary_mask.sum() / (images.size()[-1] * images.size()[-2])
                    lam_list.append(lam)
            else:
                target_a = labels
                target_b = labels
                lam_list = [1.0] * images.size(0)

            # print('lam_list', lam_list)

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss, lam belongs to target_a
            total_loss = mixup_criterion_lam_list(loss, logits, target_a, target_b, lam_list)

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,

                                                        )
        scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,

                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale


def train_attentivemix2(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """
    Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier
    Use AttentiveMix method to augment the training data
    Use rectangular patches
    """

    # initialize GEM model
    model_name = 'ViT-B/32'
    pretrained = 'laion400m_e32'
    gem_model = create_gem_model(model_name=model_name, pretrained=pretrained, device=args.device)
    gem_model.eval()
    logger.info(f'GEM model loaded from {model_name} {pretrained}')
    threshold = args.attentive_threshold

    # get the label names dict
    metric_fn = f'{args.dataset_root}/id_scname_dict.json'
    with open(metric_fn, 'r') as f:
        metrics = json.load(f)

    logger.info(f"Start Training ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0
    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            # apply the attentivemix strategy
            r = np.random.rand(1)
            if args.cutmix_beta > 0 and r < args.mix_prob:
                # generate mixed sample
                lam = np.random.beta(args.cutmix_beta, args.cutmix_beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]
                shuffled_images = images[rand_index]

                # build texts for GEM
                if args.attentive_name == 'general':
                    texts = [['bird'] for i in range(images.size()[0])]
                elif args.attentive_name == 'c-name':
                    texts = []
                    for label in target_b:
                        texts.append([metrics[str(label.item())][1]])
                elif args.attentive_name == 's-name':
                    texts = []
                    for label in target_b:
                        texts.append([metrics[str(label.item())][0]])
                else:
                    raise NotImplementedError

                corners_list = saliency_bbox_list_gem(shuffled_images, lam, gem_model, texts)
                lam_list = []

                for i in range(images.size(0)):
                    bbx1, bby1, bbx2, bby2 = corners_list[i]
                    images[i, :, bbx1:bbx2, bby1:bby2] = shuffled_images[i, :, bbx1:bbx2, bby1:bby2]
                    # adjust lambda to exactly match pixel ratio
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                    lam_list.append(lam)

            else:
                target_a = labels
                target_b = labels
                lam_list = [1.0] * images.size(0)

            # print('lam_list', lam_list)

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()

            # mixup loss, lam belongs to target_a
            total_loss = mixup_criterion_lam_list(loss, logits, target_a, target_b, lam_list)

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        # validate after 1 epoch
        # if (args.early_stop and num_iter >= args.start_validation) or \
        #     epoch == args.epochs:
        val_acc, val_loss, confusion_matrix = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                        loss=args.loss, logit_scale=logit_scale,
                                                        classifier_head=classifier_head, show_confusion_matrix=True,
                                                        dataset=args.dataset,
                                                        output_dir=args.output_dir, device=args.device,

                                                        )
        scores = calculate_scores(confusion_matrix)

        # check if val acc has improved, here i use the val_acc rather than val_loss,
        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, _ = validate(args,data_loader=test_loader, model=model, logger=logger,
                                loss=args.loss, logit_scale=logit_scale,
                                classifier_head=classifier_head,
                                dataset=args.dataset,
                                output_dir=args.output_dir, device=args.device,

                                )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},{round(test_acc, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (epoch % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                        model, classifier_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

        if epoch == args.stop_epochs:
            break

    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale