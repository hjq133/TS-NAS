import numpy as np
import sys, os, argparse, data, time, logging, gc, math, glob
import torch.nn as nn
import torch
import utils
import torch.backends.cudnn as cudnn
from bandit_search import BanditTS
from model_search import RNNModel, DARTSCell
from utils import batchify, get_batch, repackage_hidden, create_exp_dir, save_warm_up_checkpoint
from genotypes import DARTS_V2

embed_size, n_hid, n_hid_last = (850, 850, 850)
dropout, dropout_h, dropout_x, dropout_i, dropout_e = (0.75, 0.25, 0.75, 0.2, 0.1)
seed = 1001
train_batch_size = 64
eval_batch_size = 10
test_batch_size = 1
data_path = '../data/ptb/'

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank/WikiText2 Language Model')
parser.add_argument('--bptt', type=int, default=35, help='sequence length')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--log-interval', type=int, default=200, metavar='N', help='report interval')
parser.add_argument('--lr', type=float, default=7, help='initial learning rate')
parser.add_argument('--w-decay', type=float, default=8e-7, help='weight decay applied to all weights')
parser.add_argument('--epochs', type=int, default=300, help='upper epoch limit')
parser.add_argument('--save', type=str, default='EXP', help='path to save the final model')
parser.add_argument('--gpu', type=int, default=1, help='gpu')
parser.add_argument('--warm_up_epoch', type=int, default=300, help='warm up the network')
parser.add_argument('--load_warm_up', type=bool, default=True)
parser.add_argument('--load_epoch', type=int, default=200)
parser.add_argument('--load_path', type=str, default='warm_up')
parser.add_argument('--test_network', type=bool, default=True)

parser.add_argument('--reward', type=int, default=80)
parser.add_argument('--mu0', type=int, default=1)  # 高斯分布的均值
parser.add_argument('--sigma0', type=int, default=1)  # 高斯分布的方差
parser.add_argument('--sigma_tilde', type=int, default=1)
args = parser.parse_args()

np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True  # cuDNN使用的非确定性算法就会自动寻找最适合当前配置的高效算法，来达到优化运行效率
    cudnn.enabled = True

corpus = data.Corpus(data_path)
train_data = batchify(corpus.train, train_batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, test_batch_size)

n_tokens = len(corpus.dictionary)
model = RNNModel(n_tokens, embed_size, n_hid, n_hid_last, dropout, dropout_h, dropout_x, dropout_i, dropout_e,
                 cell_cls=DARTSCell)
parallel_model = model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
bandit = BanditTS(args)
device = torch.device("cuda:{}".format(args.gpu))
if args.load_warm_up:
    parallel_model, optimizer, bandit = utils.load_warm_up_checkpoint(parallel_model, optimizer, args.load_path, device,
                                                                      args.load_epoch)

total_params = sum(x.data.nelement() for x in model.parameters())


def init_logging():
    log_format = '%(asctime)s %(message)s'
    if args.load_warm_up:
        args.save = 'search-{}'.format(args.load_epoch)
        # args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    else:
        args.save = 'warm_up'
    # create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    create_exp_dir(args.save)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    return logging


def summary(total_loss, start_time, batch):
    cur_loss = total_loss / args.log_interval
    elapsed = time.time() - start_time
    logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                 'loss {:5.2f} | ppl {:8.2f}'.format(epoch, batch, len(train_data) // args.bptt,
                                                     optimizer.param_groups[0]['lr'],
                                                     elapsed * 1000 / args.log_interval, cur_loss,
                                                     math.exp(cur_loss)))


def evaluate(genotype, data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(batch_size)
    logging.info('Genotype: {}'.format(genotype))

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args, evaluation=True)
            targets = targets.view(-1)
            log_prob, hidden = parallel_model(data, hidden, genotype)
            loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data
            total_loss += loss * len(data)
            hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)


def train(genotype):
    sum_loss = 0
    total_loss = 0
    start_time = time.time()
    hidden = model.init_hidden(train_batch_size)
    batch, i = 0, 0

    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt
        seq_len = bptt
        model.train()
        datas, targets = get_batch(train_data, i, args, seq_len=seq_len)
        targets = targets.view(-1)
        optimizer.zero_grad()
        hidden = repackage_hidden(hidden)
        log_prob, hidden = parallel_model(datas, hidden, genotype, return_h=False)
        raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets)
        loss = raw_loss
        total_loss += loss
        loss.backward()
        gc.collect()  # 清内存

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        if batch % args.log_interval == 0 and batch > 0:
            sum_loss += total_loss
            summary(total_loss, start_time, batch)
            total_loss = 0
            start_time = time.time()
        batch += 1
        i += seq_len
    return sum_loss / batch


init_logging()
if not args.load_warm_up:
    logging.info('-' * 89)
    logging.info('now warm up start')
    for epoch in range(1, 1 + args.warm_up_epoch):
        epoch_start_time = time.time()
        prev, act = bandit.warm_up_sample()
        genotype = bandit.construct_genotype(prev, act)
        train(genotype)
        # val_loss = evaluate(genotype, val_data, eval_batch_size)
        # logging.info('-' * 89)
        # logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} '
        #              '| valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), val_loss,
        #                                           math.exp(val_loss)))
        # logging.info('-' * 89)
        if epoch > 0 and epoch % 40 == 0:
            save_warm_up_checkpoint(model, optimizer, bandit, args.save, epoch)
            bandit.cell.save_table(os.path.join(args.save, 'table'))
            logging.info('saved to checkpoint: warm_up_mode, bandit, optimizer')
    logging.info('now warm up end !')
    logging.info('-' * 89)
    exit()

if args.test_network:
    architecture = []
    ppl = []
    ppl_op = [[[] for _ in range(i * bandit.num_op)] for i in range(1, bandit.num_node + 1)]
    train_time = bandit.cell.sap_time
    for i in range(100):
        prev, act = bandit.random_sample()
        genotype = bandit.construct_genotype(prev, act)
        val_loss = evaluate(genotype, val_data, eval_batch_size)

        architecture.append(genotype)
        ppl.append(math.exp(val_loss))
        for i in range(bandit.num_node):
            cur_action = prev[i] * bandit.num_op + act[i]
            ppl_op[i][cur_action].append(math.exp(val_loss))

        logging.info('random sample | valid loss {:5.2f} | valid ppl {:8.2f}'.format(val_loss, math.exp(val_loss)))
        logging.info('-' * 89)
    param = {'arch': architecture, 'ppl': ppl, 'ppl_op': ppl_op, 'train_time': train_time}
    torch.save(param, os.path.join(args.save, "para.pth"))

exit()
# stored_loss = 100000000
for epoch in range(1 + args.warm_up_epoch, args.epochs + 1):
    epoch_start_time = time.time()

    prev, act = bandit.pick_action()
    genotype = bandit.construct_genotype(prev, act)
    loss = train(genotype)
    bandit.update_observation(prev, act, np.log(args.reward) / loss.item())

    if epoch > 0 and epoch % 5 == 0:  # 每5个epoch, derive and evaluate一次
        genotype = bandit.derive_sample()
        val_loss = evaluate(genotype, val_data, eval_batch_size)  # derive and evaluate
        logging.info('-' * 89)
        logging.info('derive sample | valid loss {:5.2f} | valid ppl {:8.2f}'.format(val_loss, math.exp(val_loss)))
        logging.info('-' * 89)
        # utils.save(model, os.path.join(args.save, 'trained.pt'))
        # print('saved to: trained.pt')
        # bandit.cell.save_table(os.path.join(args.save, 'table'))
        #
        # if val_loss < stored_loss:
        #     logging.info('better valid loss!')
        #     stored_loss = val_loss
