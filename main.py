import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        dest='config',
                        help='settings of Cluster-Adapter in yaml format')
    args = parser.parse_args()

    return args


class ImageAdapter(nn.Module):

    def __init__(self, cache_keys, drop_rate=0):
        super().__init__()
        self.num_cluster = 16
        self.num_cls = cache_keys.shape[1] // self.num_cluster
        print('num_cls: ', self.num_cls)
        print('cache_keys.shape: ', cache_keys.shape)

        self.center = nn.Parameter(cache_keys, requires_grad=False)

        self.center_bias = nn.Parameter(torch.zeros(cache_keys.shape),
                                        requires_grad=True)

        self.weight = nn.Parameter(torch.ones(cache_keys.shape[0]),
                                   requires_grad=True)

        self.num = cache_keys.shape[1]

        self.scale = nn.Parameter(torch.ones(cache_keys.shape[1]),
                                  requires_grad=True)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, beta=0.0, cache_values=None, pow_weight=20):
        x = x * torch.pow(F.relu(self.weight), pow_weight)  # 10
        x = x / x.norm(dim=-1, keepdim=True)

        center = self.center + self.center_bias
        center = center / center.norm(dim=0, keepdim=True)

        x = x @ center
        cls_center = center @ cache_values
        cls_center = cls_center.t()

        scale = torch.pow(F.relu(self.scale), 3)  # 3
        scale = self.dropout(scale)
        x = x * scale
        x = (beta * x - beta).exp()

        x = x @ cache_values
        if self.training:
            return x, center.t(), cls_center
        return x


class TextAdapter(nn.Module):

    def __init__(self, clip_weights, text_alpha):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(clip_weights.shape[1:]),
                                 requires_grad=True)

        self.scale = nn.Parameter(torch.ones(clip_weights.shape[0], 1, 1),
                                  requires_grad=True)

        self.scale_2 = nn.Parameter(torch.ones(1, clip_weights.shape[2]),
                                    requires_grad=True)

        self.llm_prompt = nn.Parameter(clip_weights[-1], requires_grad=False)

        self.alpha = text_alpha

    def forward(self, x_text):
        x_text = (x_text[:-1]).mean(dim=0)
        x_text_temp = x_text * self.alpha + self.llm_prompt * (1 - self.alpha)

        x_text = x_text_temp + self.bias
        x_text = x_text / x_text.norm(dim=-1, keepdim=True)

        return x_text.transpose(1, 0)


def run_cluster_adapter(cfg, cache_keys, cache_values, val_features,
                        val_labels, test_features, test_labels, clip_weights,
                        clip_model, train_loader_F):
    image_adapter = ImageAdapter(cache_keys, cfg.get('dw', 0)).cuda()
    text_adapter = TextAdapter(clip_weights, cfg['ta']).cuda()

    optimizer = torch.optim.AdamW([
        {
            'params': image_adapter.parameters(),
            'lr': cfg['lr'] * cfg['lr1']
        },
        {
            'params': text_adapter.parameters(),
            'lr': cfg['lr'] * cfg['lr2']
        },
    ],
                                  eps=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg['train_epoch'] * len(train_loader_F))

    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0

    nce_loss_1 = InfoNCE(temperature=cfg['t1'])
    nce_loss_incls = InfoNCE(temperature=cfg['t2'])
    nce_loss_2 = InfoNCE(temperature=cfg['t3'])

    num_shots = cfg['shots']
    num_cls = clip_weights.shape[1]
    mask = torch.ones([num_shots * num_cls, num_shots * num_cls])

    _mask = torch.zeros([num_shots * num_cls, num_shots * num_cls])

    for i in range(num_cls):
        for j in range(num_shots):
            for k in range(num_shots):
                r = i * num_shots + j
                c = i * num_shots + k
                if r == c:
                    pass
                else:
                    mask[r, c] = 0

    for i in range(num_cls):
        for j in range(num_shots):
            for k in range(num_shots):
                r = i * num_shots + j
                c = i * num_shots + k
                _mask[r, c] = 1

    mask = mask.cuda()
    _mask = _mask.cuda()

    for train_idx in range(cfg['train_epoch']):
        # Train
        cache_values = cache_values.float()
        image_adapter.train()
        text_adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            image_features = image_features.float()
            cache_logits, image_centers, _ = image_adapter(
                image_features,
                beta=beta,
                cache_values=cache_values,
                pow_weight=cfg['iw'])

            texts = text_adapter(clip_weights)

            clip_logits = 100. * image_features @ texts
            clip_logits = clip_logits.float()
            cache_logits = cache_logits.float()
            cluster_logits = clip_logits + cache_logits * alpha

            # image_centers
            n, c = image_centers.size()

            nce_losses_1 = nce_loss_1(
                image_centers.detach(), image_centers, mask=mask) * cfg['ls1']

            margin = 0.0
            nce_losses_in = nce_loss_incls(image_centers.detach(),
                                           image_centers,
                                           mask=_mask,
                                           margin=margin) * cfg['ls2']

            nce_losses_2 = nce_loss_2(texts.t().detach(),
                                      texts.t()) * cfg['ls3']

            loss = F.cross_entropy(
                cluster_logits,
                target) + nce_losses_1 + nce_losses_in + nce_losses_2

            acc = cls_acc(cluster_logits, target)
            correct_samples += acc / 100 * len(cluster_logits)
            all_samples += len(cluster_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        torch.cuda.empty_cache()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(
            current_lr, correct_samples / all_samples, correct_samples,
            all_samples,
            sum(loss_list) / len(loss_list)))

        # Eval
        image_adapter.eval()
        text_adapter.eval()
        cache_logits = image_adapter(test_features,
                                     beta=beta,
                                     cache_values=cache_values,
                                     pow_weight=cfg['iw'])

        clip_logits = 100. * test_features @ text_adapter(clip_weights)

        cluster_logits = clip_logits + cache_logits * alpha
        acc = cls_acc(cluster_logits, test_labels)

        print(
            "**** Cluster-Adapter's test accuracy: {:.2f}. ****\n".format(acc))
        if acc >= best_acc:
            best_acc = acc
            best_epoch = train_idx

    print(
        f"**** After fine-tuning, Cluster-Adapter's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n"
    )

    print("\n-------- Searching hyperparameters on the val set. --------")
    # Search Hyperparameters
    _, best_beta, best_alpha = search_hp_text(cfg,
                                              cache_values,
                                              val_features,
                                              val_labels,
                                              clip_weights,
                                              image_adapter=image_adapter,
                                              text_adapter=text_adapter)

    print("\n-------- Evaluating on the test set. --------")

    cache_logits = image_adapter(test_features,
                                 beta=best_beta,
                                 cache_values=cache_values,
                                 pow_weight=cfg['iw'])
    # image_adapter.vis()
    cluster_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(cluster_logits, test_labels)
    print("**** Cluster-Adapter's test accuracy: {:.2f}. ****\n".format(
        round(max(best_acc, acc), 3)))


def main(i):

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # Prepare dataset
    random.seed(i)
    torch.manual_seed(i)

    print("Preparing dataset.")
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])

    val_loader = build_data_loader(data_source=dataset.val,
                                   batch_size=64,
                                   is_train=False,
                                   tfm=preprocess,
                                   shuffle=False)
    test_loader = build_data_loader(data_source=dataset.test,
                                    batch_size=64,
                                    is_train=False,
                                    tfm=preprocess,
                                    shuffle=False)

    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(
            size=224,
            scale=(0.5, 1),
            interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711))
    ])

    train_loader_cache = build_data_loader(data_source=dataset.train_x,
                                           batch_size=256,
                                           tfm=train_tranform,
                                           is_train=True,
                                           shuffle=False)
    train_loader_F = build_data_loader(data_source=dataset.train_x,
                                       batch_size=256,
                                       tfm=train_tranform,
                                       is_train=True,
                                       shuffle=True)

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(dataset.classnames, dataset.template,
                                   clip_model, cfg['dataset'])
    print(clip_weights.size())
    clip_weights = clip_weights.float()
    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(cfg, clip_model,
                                                 train_loader_cache)

    cache_keys = cache_keys.float()
    cache_values = cache_values.float()

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(cfg, "val", clip_model,
                                                 val_loader)
    val_features = val_features.float()

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model,
                                                   test_loader)
    test_features = test_features.float()

    # ------------------------------------------ Cluster-Adapter ------------------------------------------
    run_cluster_adapter(cfg, cache_keys, cache_values, val_features,
                        val_labels, test_features, test_labels, clip_weights,
                        clip_model, train_loader_F)


if __name__ == '__main__':
    main(1)
