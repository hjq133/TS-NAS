import torch
import torch.nn as nn

from genotypes import STEPS
from utils import mask2d
from utils import LockedDropout
from utils import embedded_dropout

INITRANGE = 0.04


class DARTSCell(nn.Module):

    def __init__(self, n_inp, n_hid, dropout_h, dropout_x):
        super().__init__()  # python3 下 == super().__init(),初始化nn.Module类
        self.n_hid = n_hid
        self.dropout_h = dropout_h
        self.dropout_x = dropout_x
        # self.bn = nn.BatchNorm1d(n_hid, affine=False)

        # genotype is None when doing arch search
        steps = STEPS
        self._W0 = nn.Parameter(torch.Tensor(n_inp + n_hid, 2 * n_hid).uniform_(-INITRANGE, INITRANGE))
        self._Ws = nn.ParameterList([
            nn.Parameter(torch.Tensor(n_hid, 2 * n_hid).uniform_(-INITRANGE, INITRANGE)) for _ in range(steps)
        ])

    def forward(self, inputs, hidden, genotype):
        T, B = inputs.size(0), inputs.size(1)  # time of step , batch size

        if self.training:
            x_mask = mask2d(B, inputs.size(2), keep_prob=1. - self.dropout_x)
            h_mask = mask2d(B, hidden.size(2), keep_prob=1. - self.dropout_h)
        else:
            x_mask = h_mask = None

        hidden = hidden[0]
        hiddens = []
        for t in range(T):  # 进入darts cell search的cell函数
            hidden = self.cell(inputs[t], hidden, x_mask, h_mask, genotype)
            hiddens.append(hidden)
        hiddens = torch.stack(hiddens)
        return hiddens, hiddens[-1].unsqueeze(0)

    def _compute_init_state(self, x, h_prev, x_mask, h_mask):
        if self.training:
            xh_prev = torch.cat([x * x_mask, h_prev * h_mask], dim=-1)
        else:
            xh_prev = torch.cat([x, h_prev], dim=-1)
        c0, h0 = torch.split(xh_prev.mm(self._W0), self.n_hid, dim=-1)
        c0 = c0.sigmoid()
        h0 = h0.tanh()
        s0 = h_prev + c0 * (h0 - h_prev)
        return s0

    def _get_activation(self, name):
        if name == 'tanh':
            f = torch.tanh
        elif name == 'relu':
            f = torch.relu
        elif name == 'sigmoid':
            f = torch.sigmoid
        elif name == 'identity':
            f = lambda x: x
        else:
            raise NotImplementedError
        return f

    def cell(self, x, h_prev, x_mask, h_mask, genotype):
        s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)

        states = [s0]
        for i, (name, pred) in enumerate(genotype.recurrent):
            s_prev = states[pred]
            if self.training:
                ch = (s_prev * h_mask).mm(self._Ws[i])  # h_mask估计是drop out
            else:
                ch = s_prev.mm(self._Ws[i])
            c, h = torch.split(ch, self.n_hid, dim=-1)
            c = c.sigmoid()
            fn = self._get_activation(name)
            h = fn(h)
            s = s_prev + c * (h - s_prev)
            states += [s]

        output = torch.mean(torch.stack([states[i] for i in genotype.concat], -1), -1)
        return output


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, n_token, n_inp, n_hid, n_hid_last,
                 dropout=0.5, dropout_h=0.5, dropout_x=0.5, dropout_i=0.5, dropout_e=0.1,
                 cell_cls=DARTSCell):
        super().__init__()
        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(n_token, n_inp)  # n_inp 词向量长度 embedding的作用是把输入的index转化为向量

        self.rnn = cell_cls(n_inp, n_hid, dropout_h, dropout_x)

        self.decoder = nn.Linear(n_inp, n_token)
        self.decoder.weight = self.encoder.weight
        self.init_weights()

        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_hid_last = n_hid_last
        self.dropout = dropout
        self.dropout_i = dropout_i
        self.dropout_e = dropout_e
        self.n_token = n_token
        self.cell_cls = cell_cls

    def init_weights(self):
        self.encoder.weight.data.uniform_(-INITRANGE, INITRANGE)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-INITRANGE, INITRANGE)

    def forward(self, input, hidden, genotype, return_h=False):  # 传入的hidden注意下
        batch_size = input.size(1)

        emb = embedded_dropout(self.encoder, input, dropout=self.dropout_e if self.training else 0)
        emb = self.lockdrop(emb, self.dropout_i)

        raw_output = emb

        raw_output, new_h = self.rnn(raw_output, hidden, genotype)  # hidden states， last hidden states
        hidden = new_h
        output = self.lockdrop(raw_output, self.dropout)  # raw output是rnn每个t时刻的hidden

        # 下面这层，实际上就是hidden->output的那个线性层
        logit = self.decoder(output.view(-1, self.n_inp))  # 计算这个batch的logit值，view一下一起计算
        log_prob = nn.functional.log_softmax(logit, dim=-1)
        model_output = log_prob
        model_output = model_output.view(-1, batch_size, self.n_token)

        if return_h:
            return model_output, hidden, raw_output, output
        return model_output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return weight.new(1, bsz, self.n_hid).zero_()
