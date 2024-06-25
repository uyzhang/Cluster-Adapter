from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import json
import os

if os.environ.get('use_openclip') == '1':
    import open_clip
    use_open_clip = True
else:
    use_open_clip = False
    import clip


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[:topk].reshape(-1).float().sum(
        0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier_back(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def clip_classifier_back(classnames, template, clip_model):
    # f = open('gpt3_prompts/CuPL_prompts_stanfordcars.json')
    # f = open('gpt3_prompts/CuPL_prompts_fgvcaircraft.json')
    f = open('gpt3_prompts/CuPL_prompts_flowers102.json')

    prompts = json.load(f)
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')

            template_texts = [t.format(classname) for t in template]
            cupl_texts = prompts[classname]
            texts = template_texts + cupl_texts

            texts_token = clip.tokenize(texts, truncate=True).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts_token)

            class_embeddings[:len(template_texts)] = class_embeddings[:len(
                template_texts)] / class_embeddings[:len(template_texts)].norm(
                    dim=-1, keepdim=True)
            class_embeddings[:len(template_texts)] /= class_embeddings[:len(
                template_texts)].norm()
            learnable_weights = class_embeddings[len(template_texts):].mean(
                dim=0).unsqueeze(
                    0) / class_embeddings[len(template_texts):].mean(
                        dim=0).unsqueeze(0).norm(dim=-1, keepdim=True)
            class_embeddings = torch.concat(
                (class_embeddings[:len(template_texts)], learnable_weights),
                dim=0)
            clip_weights.append(class_embeddings)
        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def clip_classifier_xxx(classnames,
                        template,
                        clip_model,
                        dataset='stanford_cars'):
    dataset2prompt = {
        'stanford_cars': 'CuPL_prompts_stanfordcars.json',
        'fgvc': 'CuPL_prompts_fgvcaircraft.json',
        'oxford_flowers': 'CuPL_prompts_flowers102.json',
        'oxford_pets': 'CuPL_prompts_oxfordpets.json',
        'food101': 'CuPL_prompts_food101.json',
        'sun397': 'CuPL_prompts_sun397.json',
        'eurosat': 'CuPL_prompts_eurosat.json',
        'caltech101': 'CuPL_prompts_caltech101.json',
        'dtd': 'CuPL_prompts_dtd.json',
        'ucf101': 'CuPL_prompts_ucf101.json',
        'imagenet': 'CuPL_prompts_imagenet.json'
    }

    f = open('gpt3_prompts/' + dataset2prompt[dataset])

    prompts = json.load(f)
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')

            template_texts = [t.format(classname) for t in template]
            cupl_texts = prompts[classname]
            texts = template_texts + cupl_texts

            texts_token = clip.tokenize(texts, truncate=True).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts_token)

            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

            # class_embeddings[:len(template_texts)] = class_embeddings[:len(template_texts)] / class_embeddings[:len(template_texts)].norm(dim=-1, keepdim=True)
            # class_embeddings[:len(template_texts)] /= class_embeddings[:len(template_texts)].norm()
            L = len(template_texts)
            gpt_text = class_embeddings[L:].mean(dim=0).unsqueeze(0)
            # gpt_text = gpt_text / gpt_text.norm()
            # learnable_weights = class_embeddings[len(template_texts):].mean(dim=0).unsqueeze(0) / class_embeddings[len(template_texts):].mean(dim=0).unsqueeze(0).norm(dim=-1, keepdim=True)
            class_embeddings = torch.concat((class_embeddings[:L], gpt_text),
                                            dim=0)
            clip_weights.append(class_embeddings)
        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def clip_classifier(classnames, template, clip_model, dataset='stanford_cars'):
    dataset2prompt = {
        'stanford_cars': 'CuPL_prompts_stanfordcars.json',
        'fgvc': 'CuPL_prompts_fgvcaircraft.json',
        'oxford_flowers': 'CuPL_prompts_flowers102.json',
        'oxford_pets': 'CuPL_prompts_oxfordpets.json',
        'food101': 'CuPL_prompts_food101.json',
        'sun397': 'CuPL_prompts_sun397.json',
        'eurosat': 'CuPL_prompts_eurosat.json',
        'caltech101': 'CuPL_prompts_caltech101.json',
        'dtd': 'CuPL_prompts_dtd.json',
        'ucf101': 'CuPL_prompts_ucf101.json',
        'imagenet': 'CuPL_prompts_imagenet.json'
    }

    f = open('gpt3_prompts/' + dataset2prompt[dataset])

    prompts = json.load(f)
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')

            template_texts = [t.format(classname) for t in template]
            cupl_texts = prompts[classname]
            texts = template_texts + cupl_texts

            if use_open_clip:
                tokenizer = open_clip.get_tokenizer('EVA02-L-14')
                texts_token = tokenizer(texts).cuda()
            else:
                texts_token = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts_token)

            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

            # class_embeddings[:len(template_texts)] = class_embeddings[:len(template_texts)] / class_embeddings[:len(template_texts)].norm(dim=-1, keepdim=True)
            # class_embeddings[:len(template_texts)] /= class_embeddings[:len(template_texts)].norm()
            L = len(template_texts)

            total_len = 16
            embedding_len = class_embeddings.shape[0]
            embedding_len - total_len
            # gpt_text = class_embeddings[L:].mean(dim=0).unsqueeze(0)
            # gpt_text = gpt_text / gpt_text.norm()
            # learnable_weights = class_embeddings[len(template_texts):].mean(dim=0).unsqueeze(0) / class_embeddings[len(template_texts):].mean(dim=0).unsqueeze(0).norm(dim=-1, keepdim=True)
            # class_embeddings = torch.concat((class_embeddings[:L], gpt_text), dim=0)
            # for i in range (total_len):
            #     if i <= 7:
            #         class_embeddings[i,:] = class_embeddings[i,:] * 0.8

            # gpt_text = class_embeddings[L:].mean(dim=0).unsqueeze(0)
            # class_embeddings = torch.concat((class_embeddings[:L], gpt_text), dim=0)
            # clip_weights.append(class_embeddings)

            # clip_weights.append(class_embeddings[:L])
            # concat_weight = class_embeddings[:L]
            distance = (class_embeddings.shape[0] - L) // (total_len - L)
            for i in range(total_len - L):
                left = L + i * distance
                right = L + (i + 1) * distance
                if i == total_len - L - 1:
                    right = class_embeddings.shape[0]
                embeddings = class_embeddings[left:right, :].mean(
                    dim=0).unsqueeze(0)
                embeddings /= embeddings.norm(dim=-1, keepdim=True)
                class_embeddings[L + i] = embeddings * 1.0

            clip_weights.append(class_embeddings[:total_len])

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def build_cache_model(cfg, clip_model, train_loader_cache):

    if cfg['load_cache'] == False:
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(
                    augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(
                    torch.cat(train_features, dim=0).unsqueeze(0))

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(
            cache_keys,
            cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(
            cache_values,
            cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' +
                                str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' +
                                  str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values


def pre_load_features(cfg, split, clip_model, loader):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")

    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")

    return features, labels


def search_hp(cfg,
              cache_keys,
              cache_values,
              features,
              labels,
              clip_weights,
              adapter=None,
              text_adapter=None):

    if cfg['search_hp'] == True:

        beta_list = [
            i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1
            for i in range(cfg['search_step'][0])
        ]
        alpha_list = [
            i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1
            for i in range(cfg['search_step'][1])
        ]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) *
                                (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)

                if acc > best_acc:
                    print(
                        "New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}"
                        .format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print(
            "\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha


def search_hp_text(cfg,
                   cache_values,
                   image_features,
                   labels,
                   clip_weights,
                   image_adapter=None,
                   text_adapter=None):

    if cfg['search_hp'] == True:

        beta_list = [
            i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1
            for i in range(cfg['search_step'][0])
        ]
        alpha_list = [
            i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1
            for i in range(cfg['search_step'][1])
        ]

        best_acc = 0
        best_beta, best_alpha = 0, 0
        for beta in beta_list:
            for alpha in alpha_list:

                cache_logits = image_adapter(image_features,
                                             beta=beta,
                                             cache_values=cache_values,
                                             pow_weight=cfg['iw'])

                texts = text_adapter(clip_weights)
                clip_logits = 100. * image_features @ texts

                clip_logits = clip_logits.float()
                cache_logits = cache_logits.float()
                tip_logits = clip_logits + cache_logits * alpha

                acc = cls_acc(tip_logits, labels)

                if acc > best_acc:
                    print(
                        "New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}"
                        .format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print(
            "\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_acc, best_beta, best_alpha


def soft_ce_loss(pred, target, smoothing=True):
    if smoothing:
        eps = 0.03

        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, target, reduction='mean')

    return loss


class LargeMarginInSoftmaxLoss(nn.CrossEntropyLoss):
    r"""
    This combines the Softmax Cross-Entropy Loss (nn.CrossEntropyLoss) and the large-margin inducing
    regularization proposed in
       T. Kobayashi, "Large-Margin In Softmax Cross-Entropy Loss." In BMVC2019.
    
    This loss function inherits the parameters from nn.CrossEntropyLoss except for `reg_lambda` and `deg_logit`.

    Args:
         reg_lambda (float, optional): a regularization parameter. (default: 0.3)
         deg_logit (bool, optional): underestimate (degrade) the target logit by -1 or not. (default: False)
                                     If True, it realizes the method that incorporates the modified loss into ours
                                     as described in the above paper (Table 4).
    """

    def __init__(self,
                 reg_lambda=0.3,
                 deg_logit=None,
                 weight=None,
                 size_average=None,
                 ignore_index=-100,
                 reduce=None,
                 reduction='mean'):
        super(LargeMarginInSoftmaxLoss,
              self).__init__(weight=weight,
                             size_average=size_average,
                             ignore_index=ignore_index,
                             reduce=reduce,
                             reduction=reduction)
        self.reg_lambda = reg_lambda
        self.deg_logit = deg_logit

    def forward(self, input, target):
        N = input.size(0)  # number of samples
        C = input.size(1)  # number of classes
        Mask = torch.zeros_like(input, requires_grad=False)
        Mask[range(N), target] = 1

        if self.deg_logit is not None:
            input = input - self.deg_logit * Mask

        loss = F.cross_entropy(input,
                               target,
                               weight=self.weight,
                               ignore_index=self.ignore_index,
                               reduction=self.reduction)

        X = input - 1.e6 * Mask  # [N x C], excluding the target class
        reg = 0.5 * ((F.softmax(X, dim=1) - 1.0 /
                      (C - 1)) * F.log_softmax(X, dim=1) *
                     (1.0 - Mask)).sum(dim=1)
        if self.reduction == 'sum':
            reg = reg.sum()
        elif self.reduction == 'mean':
            reg = reg.mean()
        elif self.reduction == 'none':
            reg = reg

        return loss + self.reg_lambda * reg


loss_func = LargeMarginInSoftmaxLoss(
    reg_lambda=0.25)  # 0.9 fgvc / 0.22 standford car /


def large_loss(input, target):
    global loss_func
    return loss_func(input, target)


class AdMSoftmaxLoss(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m=0.4):
        '''
        AM Softmax Loss
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = F.normalize(W, dim=1)

        x = F.normalize(x, dim=1)

        wf = self.fc(x)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) -
                              self.m)
        excl = torch.cat([
            torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0)
            for i, y in enumerate(labels)
        ],
                         dim=0)
        denominator = torch.exp(numerator) + torch.sum(
            torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self,
                 temperature=0.1,
                 reduction='mean',
                 negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self,
                query,
                positive_key,
                negative_keys=None,
                mask=None,
                margin=None):
        return info_nce(query,
                        positive_key,
                        negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode,
                        mask=mask,
                        margin=margin)


def info_nce(query,
             positive_key,
             negative_keys=None,
             temperature=0.1,
             reduction='mean',
             negative_mode='unpaired',
             mask=None,
             margin=None):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError(
                "<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'."
            )
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError(
                "<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'."
            )

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError(
            '<query> and <positive_key> must must have the same number of samples.'
        )
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError(
                "If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>."
            )

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError(
            'Vectors of <query> and <positive_key> should have the same number of components.'
        )
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError(
                'Vectors of <query> and <negative_keys> should have the same number of components.'
            )

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key,
                                                   negative_keys)

    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits),
                             dtype=torch.long,
                             device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    if margin is not None:
        mask[logits < margin] = 0.0

    if mask is not None:
        logits = logits * mask

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)

    # labels = F.one_hot(labels, num_classes=len(query)).float()
    # logits = F.softmax(logits / temperature, dim=1)
    # logits = torch.log(logits)
    # loss = -(logits * labels).sum(dim=1).mean()

    # print (logits)
    # breakpoint()
    # loss = -(logits @ labels).sum(dim=1).mean()

    # return Loss
    # if margin is not None:
    #     # print (torch.abs(logits) < margin)
    #     # breakpoint()
    #     logits[torch.abs(logits) < margin] = 0.0
    # # logits[torch.abs(logits) < 1e-6] = -999999.9
    # # print (logits.max(), logits.min())
    # # return large_loss( logits / temperature, labels)
    # labels
    # return F.cross_entropy(logits / temperature, labels, reduction=reduction)
    # loss_1 = F.cross_entropy(logits / temperature, labels, reduction=reduction)

    # amsoftmax_loss = AdMSoftmaxLoss(len(query), len(query), m=0.1, s=0.5).cuda()

    # return amsoftmax_loss(logits / temperature, labels) * 0.2 + loss_1 * 0.8


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


# def contrastive_loss()
