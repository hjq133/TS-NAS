import torch
from torch import nn
import torch.nn.functional as F
from operations import OPS, FactorizedReduce, ReLUConvBN
from genotypes import PRIMITIVES, Genotype, NAME2ID


class Cell(nn.Module):
    def __init__(self, steps, multiplier, cpp, cp, c, reduction, reduction_prev):
        """

        :param steps: 4, number of layers inside a cell
        :param multiplier: 4
        :param cpp: 48
        :param cp: 48
        :param c: 16
        :param reduction: indicates whether to reduce the output maps width
        :param reduction_prev: when previous cell reduced width, s1_d = s0_d//2
        in order to keep same shape between s1 and s0, we adopt prep0 layer to
        reduce the s0 width by half.
        """
        super().__init__()

        # indicating current cell is reduction or not
        self.reduction = reduction
        self.reduction_prev = reduction_prev

        # preprocess0 deal with output from prev_prev cell
        if reduction_prev:
            # if prev cell has reduced channel/double width,
            # it will reduce width by half
            self.preprocess0 = FactorizedReduce(cpp, c, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(cpp, c, 1, 1, 0, affine=False)
        # preprocess1 deal with output from prev cell
        self.preprocess1 = ReLUConvBN(cp, c, 1, 1, 0, affine=False)

        # steps inside a cell
        self.steps = steps  # 4
        self.multiplier = multiplier  # 4
        self.layers = nn.ModuleList()

        for i in range(self.steps):
            for j in range(2 + i):
                # for reduction cell, it will reduce the heading 2 inputs only
                stride = 2 if reduction and j < 2 else 1
                self.layers.append([OPS[primitive](c, stride, False) for primitive in PRIMITIVES])

    def forward(self, s0, s1, genotype):
        """
        :param s0:
        :param s1:
        :param genotype: [14, 8]
        :return:
        """
        s0 = self.preprocess0(s0)  # [40, 48, 32, 32], [40, 16, 32, 32]
        s1 = self.preprocess1(s1)  # [40, 48, 32, 32], [40, 16, 32, 32]

        states = [s0, s1]
        offset = 0

        if self.reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat

        # for each node, receive input from all previous intermediate nodes and s0, s1
        for i in range(self.steps):
            op1_id = NAME2ID[op_names[2 * i]]
            op2_id = NAME2ID[op_names[2 * i + 1]]
            h1 = states[indices[2 * i]]
            h2 = states[indices[2 * i + 1]]
            op1 = self.layers[offset + indices[2 * i]][op1_id]
            op2 = self.layers[offset + indices[2 * i + 1]][op2_id]
            h1 = op1(h1)
            h2 = op2(h2)

            s = h1 + h2
            states += [s]
            offset += len(states)
        # concat along dim=channel
        return torch.cat([states[i] for i in concat], dim=1)  # 6 of [40, 16, 32, 32]


class Network(nn.Module):
    """
    stack number:layer of cells and then flatten to fed a linear layer
    """

    def __init__(self, c, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        """

        :param c: 16
        :param num_classes: 10
        :param layers: number of cells of current network
        :param criterion:
        :param steps: nodes num inside cell
        :param multiplier: output channel of cell = multiplier * ch
        :param stem_multiplier: output channel of stem net = stem_multiplier * ch
        """
        super().__init__()

        self.c = c
        self.num_classes = num_classes
        self.layers = layers
        self.criterion = criterion
        self.steps = steps
        self.multiplier = multiplier

        # stem_multiplier is for stem network,
        # and multiplier is for general cell
        c_curr = stem_multiplier * c  # 3*16
        # stem network, convert 3 channel to c_curr
        self.stem = nn.Sequential(  # 3 => 48
            nn.Conv2d(3, c_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_curr)
        )

        # c_curr means a factor of the output channels of current cell
        # output channels = multiplier * c_curr
        cpp, cp, c_curr = c_curr, c_curr, c  # 48, 48, 16
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):

            # for layer in the middle [1/3, 2/3], reduce via stride=2
            if i in [layers // 3, 2 * layers // 3]:
                c_curr *= 2
                reduction = True
            else:
                reduction = False

            # [cp, h, h] => [multiplier*c_curr, h/h//2, h/h//2]
            # the output channels = multiplier * c_curr
            cell = Cell(steps, multiplier, cpp, cp, c_curr, reduction, reduction_prev)
            # update reduction_prev
            reduction_prev = reduction
            self.cells += [cell]
            cpp, cp = cp, multiplier * c_curr
        # adaptive pooling output size to 1x1
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # since cp records last cell's output channels
        # it indicates the input channel number
        self.classifier = nn.Linear(cp, num_classes)

    def forward(self, x, genotype):
        """
        in: torch.Size([3, 3, 32, 32])
        stem: torch.Size([3, 48, 32, 32])
        cell: 0 torch.Size([3, 64, 32, 32]) False
        cell: 1 torch.Size([3, 64, 32, 32]) False
        cell: 2 torch.Size([3, 128, 16, 16]) True
        cell: 3 torch.Size([3, 128, 16, 16]) False
        cell: 4 torch.Size([3, 128, 16, 16]) False
        cell: 5 torch.Size([3, 256, 8, 8]) True
        cell: 6 torch.Size([3, 256, 8, 8]) False
        cell: 7 torch.Size([3, 256, 8, 8]) False
        pool:   torch.Size([16, 256, 1, 1])
        linear: [b, 10]
        :param x:
        :param genotype:
        :return:
        """
        # print('in:', x.shape)
        # s0 & s1 means the last cells' output
        s0 = s1 = self.stem(x)  # [b, 3, 32, 32] => [b, 48, 32, 32]
        # print('stem:', s0.shape)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, genotype)  # [40, 64, 32, 32]
        # s1 is the last cell's output
        out = self.global_pooling(s1)
        # print('pool', out.shape)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def loss(self, x, target):
        """
        :param x:
        :param target:
        :return:
        """
        logits = self(x)  # TODO how many params should be there ? genotype ?
        return self.criterion(logits, target)

    # def genotype(self):
    #     """
    #     :return:
    #     """
    #
    #     def _parse(weights):
    #         """
    #         :param weights: [14, 8]
    #         :return:
    #         """
    #         gene = []
    #         n = 2
    #         start = 0
    #         for i in range(self.steps):  # for each node
    #             end = start + n
    #             W = weights[start:end].copy()  # [2, 8], [3, 8], ...
    #             edges = sorted(range(i + 2),  # i+2 is the number of connection for node i
    #                            key=lambda x: -max(W[x][k]  # by descending order
    #                                               for k in range(len(W[x]))  # get strongest ops
    #                                               if k != PRIMITIVES.index('none'))
    #                            )[:2]  # only has two inputs
    #             for j in edges:  # for every input nodes j of current node i
    #                 k_best = None
    #                 for k in range(len(W[j])):  # get strongest ops for current input j->i
    #                     if k != PRIMITIVES.index('none'):
    #                         if k_best is None or W[j][k] > W[j][k_best]:
    #                             k_best = k
    #                 gene.append((PRIMITIVES[k_best], j))  # save ops and input node
    #             start = end
    #             n += 1
    #         return gene
    #
    #     gene_normal = _parse(F.softmax(self.alpha_normal, dim=-1).data.cpu().numpy())
    #     gene_reduce = _parse(F.softmax(self.alpha_reduce, dim=-1).data.cpu().numpy())
    #
    #     concat = range(2 + self.steps - self.multiplier, self.steps + 2)
    #     genotype = Genotype(
    #         normal=gene_normal, normal_concat=concat,
    #         reduce=gene_reduce, reduce_concat=concat
    #     )
    #
    #     return genotype
