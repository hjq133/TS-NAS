import numpy as np
import sys, os, argparse, data, time, logging, gc, math, glob
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from bandit_search import BanditTS
from model_search import RNNModel, DARTSCell
from utils import batchify, get_batch, repackage_hidden, create_exp_dir  # , save_checkpoint
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
parser.add_argument('--lr', type=float, default=10, help='initial learning rate')
parser.add_argument('--w-decay', type=float, default=8e-7, help='weight decay applied to all weights')
parser.add_argument('--epochs', type=int, default=300, help='upper epoch limit')
parser.add_argument('--save', type=str, default='EXP', help='path to save the final model')
parser.add_argument('--gpu', type=int, default=3, help='gpu')

parser.add_argument('--reward', type=int, default=80)
parser.add_argument('--mu0', type=int, default=0)  # 高斯分布的均值
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
total_params = sum(x.data.nelement() for x in model.parameters())


def init_logging():
    log_format = '%(asctime)s %(message)s'
    args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
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


def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(batch_size)
    prev, act = bandit.derive_sample()
    genotype = bandit.construct_genotype(prev, act)
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


def train():
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

        prev, act = bandit.pick_action()
        genotype = bandit.construct_genotype(prev, act)

        log_prob, hidden = parallel_model(datas, hidden, genotype, return_h=False)
        raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets)
        loss = raw_loss
        bandit.update_observation(prev, act, np.log(args.reward) / loss.item())

        total_loss += loss
        loss.backward()
        gc.collect()  # 清内存

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        if batch % args.log_interval == 0 and batch > 0:
            summary(total_loss, start_time, batch)
            total_loss = 0
            start_time = time.time()
        batch += 1
        i += seq_len


bandit = BanditTS(args)
logging = init_logging()
stored_loss = 100000000
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    train()

    val_loss = evaluate(val_data, eval_batch_size)
    logging.info('-' * 89)
    logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} '
                 '| valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
    logging.info('-' * 89)

    if epoch > 0 and epoch % 5 == 0:
        bandit.internal_env.save_table(os.path.join(args.save, 'table'))

    if val_loss < stored_loss:
        logging.info('better loss!')
        stored_loss = val_loss
