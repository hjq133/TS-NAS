import torch
import torch.nn as nn
import os
import shutil
import logging


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = m.div_(1 - dropout).detach()
        mask = mask.expand_as(x)
        return mask * x


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
            embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1
    X = nn.functional.embedding(words, masked_embed_weight,
                                padding_idx, embed.max_norm, embed.norm_type,
                                embed.scale_grad_by_freq, embed.sparse
                                )
    return X


def mask2d(B, D, keep_prob):
    m = torch.floor(torch.rand(B, D) + keep_prob) / keep_prob
    m = m.requires_grad_(False).cuda()
    return m


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len].detach() if evaluation else source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len]
    return data, target


def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    print(data.size())
    data = data.cuda()
    return data


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def make_dirs(path):
    if not os.path.exists(path):
        logging.info("[*] Make directories : {}".format(path))
        os.makedirs(path)


def save_warm_up_checkpoint(model, optimizer, bandit, path):
    torch.save(model.state_dict(), os.path.join(path, 'warm_up_model.pt'))
    torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))
    torch.save({'bandit': bandit}, os.path.join(path, 'bandit.pt'))


def load_warm_up_checkpoint(model, optimizer, path, device):
    print('load from warm up model:', path)
    model_dict = torch.load(os.path.join(path, 'warm_up_model.pt'), map_location=device)
    model.load_state_dict(model_dict.state_dict())
    optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer.pt')))
    bandit = torch.load(os.path.join(path, 'bandit.pt'))['bandit']
    return model, optimizer, bandit


def save(model, model_path):
    print('saved to model:', model_path)
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    print('load from model:', model_path)
    model.load_state_dict(torch.load(model_path))
    return model
