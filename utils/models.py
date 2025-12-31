import torch
from torch import nn
import json
import pickle
from utils.features import prompt_sampler, get_text_features
from utils.extras import get_engine#, cal_hard_avg_acc, cal_easy_avg_acc
from utils.datasets.dataset_utils import NUM_CLASSES_DICT
from utils import features
from copy import deepcopy



class ProjectionMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(ProjectionMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def set_model(args, logger):

    model, preprocess, tokenizer = get_engine(model_cfg=args.model_cfg, device=args.device)
    logger.info(f'Loaded model: {args.model_cfg}')
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    return model, preprocess, tokenizer



def set_classifier(args, prompt_tensors, logger):

    if args.method == "dataset-cls":
        num_class = 2 # binary classification, retrieved 0 or fewshot 1
        classifier_head = MyLinear(inp_dim=512, num_classes=num_class, bias=False)
        logger.info(f'Initialized classifier head with random weights. Num of classes: {num_class}')

    elif args.cls_init == 'REAL-Prompt' or args.cls_init == 'REAL-Linear' or args.cls_init == 'text':
        weights = features.prompt_sampler(prompt_tensors, sample_by='mean')

        if args.scale_text_embedding:
            weights *= args.model_logit_scale.exp()
            logger.info(f'Applied logit scale: {args.model_logit_scale.item()} to the text embedding weights.')

        classifier_head = MyLinear(weights=weights)
        logger.info(f'Initialized classifier head with text embedding. weights.shape: {weights.shape}')

    elif args.cls_init == 'random':
        num_class = NUM_CLASSES_DICT[args.dataset]
        classifier_head = MyLinear(inp_dim=512, num_classes=num_class, bias=False)
        logger.info(f'Initialized classifier head with random weights. Num of classes: {num_class}')
    else:
        raise NotImplementedError(f'Classifier head {args.cls_init} not implemented.')

    classifier_head.to(args.device)

    return classifier_head



class DinoVisionTransformer(nn.Module):
    def __init__(self, model_cfg):
        super(DinoVisionTransformer, self).__init__()
        dinov2_model = torch.hub.load('facebookresearch/dinov2', model_cfg)
        # for name, module in dinov2_model.named_modules():
            # print(name)
        # print(f'model_ft.linear_head.in_features: {dinov2_model.linear_head.in_features}')
        # print(f'model_ft.linear_head.out_features: {dinov2_model.linear_head.out_features}')
        # exit()

        self.transformer = deepcopy(dinov2_model)
        # self.classifier = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        return x


class ImageEncoder(nn.Module):
    def __init__(self, model, model_cfg):
        super(ImageEncoder, self).__init__()
        self.model = model
        self.model_cfg = model_cfg

    def forward(self, x):
        cfgs = self.model_cfg.split('_')
        # arch = cfgs[0]
        model_name = cfgs[1]
        # pretraining_dataset = cfgs[2] if len(cfgs) == 3 else None

        if model_name == 'openclip':
            x = self.model.encode_image(x)
            x = x / x.norm(dim=-1, keepdim=True)
        else:
            x = self.model(x)

        return x



class MyLinear(nn.Module):
    def __init__(self, weights=None, inp_dim=512, num_classes=810, bias = False):
        super(MyLinear, self).__init__()

        if torch.is_tensor(weights):
            self.linear = nn.Linear(weights.shape[1], weights.shape[0], bias=bias) # Set bias = False, so that we simply do Zero Shot.
            with torch.no_grad():
                self.linear.weight.copy_(weights)
            self.num_classes = weights.shape[0]
        else:
            self.linear = nn.Linear(inp_dim, num_classes, bias=bias)
            self.num_classes = num_classes

    def forward(self, x):
        x = self.linear(x)
        return x

    def update_weights(self, weights): # Cosine Similarity Validation during CLIP fine-tuning.
        with torch.no_grad():
            self.linear.weight.copy_(weights)



def build_classifier_head(args, model, text_prompts, tokenizer):
    # build new classifier head using the updated model
    updated_prompt_tensors = get_text_features(model, text_prompts, tokenizer, 'encode')
    weights = prompt_sampler(updated_prompt_tensors, sample_by='mean')
    new_head = MyLinear(weights=weights, bias=False)
    new_head.to(args.device)

    return new_head


def save_model_ckpt(args, best_records, model, classifier_head, optimizer, scheduler, logit_scale,
                    val_acc=-1, epoch=-1, num_iter=-1):

    model_path = f'{args.ckpt_path}/ckpt_epoch_{epoch}.pth'

    state = {}
    # state['best_val_acc'] = best_records['best_val_acc']
    # state['best_epoch'] = best_records['best_epoch']
    # state['best_iter'] = best_records['best_iter']
    # state['best_scores'] = best_scores

    state['val_acc'] = val_acc
    state['epoch'] = epoch
    state['num_iter'] = num_iter
    if not args.freeze_visual:
        state['clip'] = model.state_dict()
    state['head'] = classifier_head.state_dict()
    state['optimizer'] = optimizer.state_dict()
    state['scheduler'] = scheduler.state_dict()
    state['logit_scale'] = logit_scale

    torch.save(state, model_path)

    return model_path


def save_best_model(args, best_records, best_model, best_head, best_logit_scale,
                    test_acc, best_tau, best_tau_test_acc, wsft_test_acc,
                    best_tau_head, wsft_backbone, wsft_head, stage=1):

    best_epoch = best_records['best_epoch']
    best_iter = best_records['best_iter']
    # model_path = f'{args.output_dir}/stage{stage}_model_best-epoch_{best_epoch}_best.pth'
    model_path = f'{args.output_dir}/stage{stage}_model_best.pth'


    # save scores of the best model to a json file
    save_path = f'{args.output_dir}/stage{stage}_val_scores_best.json'
    with open(save_path, 'w') as f:
        json.dump(best_records['best_scores'], f, indent=4)

    save_path = f'{args.output_dir}/stage{stage}_val_confusion_matrix_best.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(best_records['best_confusion_matrix'], f)

    state = {}
    state['best_val_acc'] = best_records['best_val_acc']
    state['best_epoch'] = best_records['best_epoch']
    state['best_iter'] = best_records['best_iter']
    # state['best_scores'] = best_scores
    if not args.freeze_visual:
        state['clip'] = best_model.state_dict()
    state['head'] = best_head.state_dict()
    state['logit_scale'] = best_logit_scale
    state['test_acc'] = round(test_acc, 3)
    state['best_tau'] = best_tau
    state['best_tau_test_acc'] = round(best_tau_test_acc, 3)
    state['wsft_test_acc'] = round(wsft_test_acc,3)
    state['best_tau_head'] = best_tau_head.state_dict() if best_tau_head is not None else None
    state['wsft_head'] = wsft_head.state_dict() if wsft_head is not None else None
    state['wsft_backbone'] = wsft_backbone.state_dict() if wsft_backbone is not None else None

    torch.save(state, model_path)

    return model_path


def save_test_scores(scores, confusion_matrix, output_dir, tag, stage=1):

    # save scores to a json file
    save_path = f'{output_dir}/stage{stage}_{tag}_scores.json'
    with open(save_path, 'w') as f:
        json.dump(scores, f, indent=4)

    # save the confusion matrix
    save_path = f'{output_dir}/stage{stage}_{tag}_confusion_matrix.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(confusion_matrix, f)

def save_head_weights(classifier_head, output_dir, tag):
    save_path = f'{output_dir}/{tag}_head_weights.pth'
    torch.save(classifier_head.state_dict(), save_path)

