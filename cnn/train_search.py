import os, sys, time, glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
from torch import optim
from bandit_search import BanditTS
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from model_search import Network

parser = argparse.ArgumentParser("cifar")
# thompson sampling parameters
parser.add_argument('--reward_c', type=int, default=3)  # reward 基数  # TODO 每条边所获得的reward需要与训练次数t作比较
parser.add_argument('--mu0', type=int, default=1)  # 高斯分布的均值  TODO 初值是不是很重要
parser.add_argument('--sigma0', type=int, default=1)  # 高斯分布的方差  TODO 初值是不是很重要
parser.add_argument('--sigma_tilde', type=int, default=1)  # 方差 assume为1
parser.add_argument('--top_k', type=int, default=2)  # 保留top k的边
# CIFAR neural network parameters
parser.add_argument('--data', type=str, default='../data/cifar10', help='location of the data corpus')
parser.add_argument('--batchsz', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.025, help='init learning rate')
parser.add_argument('--lr_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--wd', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
parser.add_argument('--warm_up_epochs', type=int, default=80, help='num of warm up training epochs')
parser.add_argument('--train_epoch', type=int, default=400, help='train the final architecture')
parser.add_argument('--init_ch', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--exp_path', type=str, default='search', help='experiment name')
parser.add_argument('--seed', type=int, default=3, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping range')
parser.add_argument('--train_portion', type=float, default=0.75, help='portion of training/val splitting')  # 区分训练集和测试集合
args = parser.parse_args()

args.exp_path += str(args.gpu)
utils.create_exp_dir(args.exp_path, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.exp_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device('cuda:0')


def main():
    np.random.seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.manual_seed(args.seed)

    logging.info('GPU device = %d' % args.gpu)
    logging.info("args = %s", args)
    criterion = nn.CrossEntropyLoss().to(device)
    model = Network(args.init_ch, 10, args.layers, criterion).to(device)
    logging.info("Total param size = %f MB", utils.count_parameters_in_MB(model))
    # this is the optimizer to optimize
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)
    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    num_train = len(train_data)  # 50000
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))  # 25000

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batchsz,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split - 37000]),
        pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batchsz,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[12000 + split:]),
        pin_memory=True, num_workers=2)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.lr_min)

    bandit = BanditTS(args)

    logging.info("warm up start")
    for epoch in range(args.warm_up_epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('\nwarm up Epoch: %d lr: %e', epoch, lr)
        n_prev, n_act, r_prev, r_act = bandit.warm_up_sample()  # 优先选择没有训练的
        genotype = bandit.construct_genotype(n_prev, n_act, r_prev, r_act)
        train(train_queue, model, criterion, optimizer, bandit, genotype)
    logging.info("warm up end")

    logging.info("search start")
    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('\nsearch Epoch: %d lr: %e', epoch, lr)

        n_prev, n_act, r_prev, r_act = bandit.pick_action()
        genotype = bandit.construct_genotype(n_prev, n_act, r_prev, r_act)

        # training
        train_acc, train_obj = train(train_queue, model, criterion, optimizer, bandit, genotype)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion, genotype)
        reward = valid_acc  # TODO how to set reward value, reward should be less and less as T go up
        bandit.update_observation(n_prev, n_act, r_prev, r_act, reward)
        logging.info('valid acc: %f', valid_acc)
        bandit.prune(epoch)
        # utils.save(model, os.path.join(args.exp_path, 'search.pt'))
    logging.info('search end')
    logging.info('-' * 89)
    n_prev, n_act, r_prev, r_act = bandit.pick_action()
    genotype = bandit.construct_genotype(n_prev, n_act, r_prev, r_act)
    logging.info(genotype)
    logging.info('-' * 89)
    logging.info('train final start')
    for epoch in range(args.train_epoch):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('\ntrain final Epoch: %d lr: %e', epoch, lr)
        # training
        train_acc, train_obj = train(train_queue, model, criterion, optimizer, bandit, genotype)
        logging.info('train acc: %f', train_acc)
        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion, genotype)
        logging.info('valid acc: %f', valid_acc)

    logging.info('train final end')
    logging.info('-' * 89)
    logging.info('-' * 89)
    logging.info('test start')
    logging.info('test end')


def train(train_queue, model, criterion, optimizer, bandit, genotype):
    """
    :param train_queue: train loader
    :param model: network
    :param criterion
    :param optimizer
    :param bandit
    :param genotype
    :return:
    """
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    for step, (x, target) in enumerate(train_queue):  # max step > 500
        batchsz = x.size(0)
        model.train()
        # [b, 3, 32, 32], [40]

        x, target = x.to(device), target.cuda(non_blocking=True)
        logits = model(x, genotype)
        loss = criterion(logits, target)
        # update weight
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        losses.update(loss.item(), batchsz)
        top1.update(prec1.item(), batchsz)
        top5.update(prec5.item(), batchsz)

        if step % args.report_freq == 0:
            logging.info('Step:%03d loss:%f acc1:%f acc5:%f', step, losses.avg, top1.avg, top5.avg)

    return top1.avg, losses.avg


def infer(valid_queue, model, criterion, genotype):
    """
    :param valid_queue:
    :param model:
    :param criterion:
    :param genotype
    :return:
    """
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model.eval()
    with torch.no_grad():
        for step, (x, target) in enumerate(valid_queue):

            x, target = x.to(device), target.cuda(non_blocking=True)
            batchsz = x.size(0)

            logits = model(x, genotype)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            losses.update(loss.item(), batchsz)
            top1.update(prec1.item(), batchsz)
            top5.update(prec5.item(), batchsz)

            if step % args.report_freq == 0:
                logging.info('>> Validation: %3d %e %f %f', step, losses.avg, top1.avg, top5.avg)

    return top1.avg, losses.avg


if __name__ == '__main__':
    main()
