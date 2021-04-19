""" CNN for architecture search """
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax_bisect

from epe_darts import genotypes as gt, ops, utils
from epe_darts.architect import Architect


class SearchCell(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """
    def __init__(self, n_nodes, C_pp, C_p, C, reduction_p, reduction):
        """
        Args:
            n_nodes: # of intermediate n_nodes
            C_pp: C_out[k-2]
            C_p : C_out[k-1]
            C   : C_in[k] (current)
            reduction_p: flag for whether the previous cell is reduction cell or not
            reduction: flag for whether the current cell is reduction cell or not
        """
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = ops.FactorizedReduce(C_pp, C, affine=False)
        else:
            self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0, affine=False)
        self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0, affine=False)

        # generate dag
        self.dag = nn.ModuleList()
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            for j in range(2 + i):  # include 2 input nodes
                # reduction should be used only for input node
                stride = 2 if reduction and j < 2 else 1
                op = ops.MixedOp(C, stride)
                self.dag[i].append(op)

    def forward(self, s0, s1, w_dag):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        for edges, w_list in zip(self.dag, w_dag):
            s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(states, w_list)))
            states.append(s_cur)

        s_out = torch.cat(states[2:], dim=1)
        return s_out


class SearchCNN(nn.Module):
    """ Search CNN model """
    def __init__(self, C_in, C, n_classes, n_layers, n_nodes=4, stem_multiplier=3):
        """
        Args:
            C_in: # of input channels
            C: # of starting model channels
            n_classes: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell
            stem_multiplier
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers

        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )

        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.
        C_pp, C_p, C_cur = C_cur, C_cur, C

        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers//3, 2*n_layers//3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False

            cell = SearchCell(n_nodes, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)

    def forward(self, x, weights_normal, weights_reduce):
        s0 = s1 = self.stem(x)

        for cell in self.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, weights)

        out = self.gap(s1)
        out = out.view(out.size(0), -1) # flatten
        logits = self.linear(out)
        return logits


class SearchCNNController(pl.LightningModule):
    """ SearchCNN controller supporting multi-gpu """
    def __init__(self, input_channels, init_channels, n_classes, n_layers, n_nodes=4, stem_multiplier=3,
                 w_lr=0.025, w_momentum=0.9, w_weight_decay: float = 3e-4, w_lr_min: float = 0.001, w_grad_clip=5.,
                 alpha_lr=3e-4, alpha_weight_decay=1e-3,
                 max_epochs: int = 50,
                 sparsity=8, alpha_normal=None, alpha_reduce=None):
        super().__init__()
        self.automatic_optimization = False
        self.n_nodes: int = n_nodes
        self.criterion = nn.CrossEntropyLoss()
        self.w_lr: float = w_lr
        self.w_momentum: float = w_momentum
        self.w_weight_decay: float = w_weight_decay
        self.w_lr_min = w_lr_min
        self.w_grad_clip = w_grad_clip
        self.alpha_lr: float = alpha_lr
        self.alpha_weight_decay: float = alpha_weight_decay
        self.max_epochs: int = max_epochs
        self.sparsity = sparsity

        # initialize architect parameters: alphas
        n_ops = len(gt.PRIMITIVES)

        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()

        if alpha_normal is not None and alpha_reduce is not None:
            print('Using provided alphas...')
            for normal, reduce in zip(alpha_normal, alpha_reduce):
                self.alpha_normal.append(nn.Parameter(normal))
                self.alpha_reduce.append(nn.Parameter(reduce))
        else:
            for i in range(n_nodes):
                self.alpha_normal.append(nn.Parameter(torch.randn(i + 2, n_ops)))  # * 1e-3 *
                self.alpha_reduce.append(nn.Parameter(torch.randn(i + 2, n_ops)))  # * 1e-3 *

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))

        self.net = SearchCNN(input_channels, init_channels, n_classes, n_layers, n_nodes, stem_multiplier)
        self.architect: Optional[Architect] = None

    def forward(self, x):
        if self.sparsity == 1:
            weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
            weights_reduce = [F.softmax(alpha, dim=-1) for alpha in self.alpha_reduce]
        else:
            weights_normal = [entmax_bisect(alpha, dim=-1, alpha=self.sparsity) for alpha in self.alpha_normal]
            weights_reduce = [entmax_bisect(alpha, dim=-1, alpha=self.sparsity) for alpha in self.alpha_reduce]

        return self.net(x, weights_normal, weights_reduce)

    def training_step(self, batch, batch_idx, optimizer_idx):
        (trn_X, trn_y), (val_X, val_y) = batch
        w_optim, alpha_optim = self.optimizers()

        if optimizer_idx != 0:
            return

        # phase 2. architect step (alpha)
        alpha_optim.zero_grad()
        w_lr = self.w_scheduler['scheduler'].get_last_lr()[-1]
        self.architect.unrolled_backward(trn_X, trn_y, val_X, val_y, w_lr, w_optim)
        alpha_optim.step()

        # phase 1. child network step (w)
        w_optim.zero_grad()
        logits = self(trn_X)
        loss = self.criterion(logits, trn_y)
        self.manual_backward(loss)

        # gradient clipping
        nn.utils.clip_grad_norm_(self.weights(), self.w_grad_clip)
        w_optim.step()

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        self.log('train_loss', loss)
        self.log('train_top1', prec1)
        self.log('train_top5', prec5)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            logits = self(x)
            loss = self.criterion(logits, y)
            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))

        self.log('valid_loss', loss)
        self.log('valid_top1', prec1)
        self.log('valid_top5', prec5)

    def on_train_epoch_start(self):
        self.print_alphas()

    def on_validation_epoch_end(self):
        # log genotype
        genotype = self.genotype()
        print(genotype)
        # TODO: log genotype with wandb (visualization of alpha connections)
        # self.log('genotype', genotype)

    def loss(self, x, y):
        logits = self.forward(x)
        return self.criterion(logits, y)

    def configure_optimizers(self):
        w_optim = torch.optim.SGD(self.weights(), self.w_lr, momentum=self.w_momentum, weight_decay=self.w_weight_decay)
        self.w_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(w_optim, self.max_epochs, eta_min=self.w_lr_min),
            'interval': 'epoch',
        }

        alpha_optim = torch.optim.Adam(self.alphas(), self.alpha_lr, betas=(0.5, 0.999), weight_decay=self.alpha_weight_decay)
        self.alpha_scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(alpha_optim, lr_lambda=lambda x: self.alpha_lr),
            'interval': 'epoch',
        }
        return [w_optim, alpha_optim], [self.w_scheduler, self.alpha_scheduler]

    def print_alphas(self):
        # remove formats
        print(f'Sparsity: {self.sparsity}')
        print("####### ALPHA #######")
        print("\n# Alpha - normal")
        for alpha in self.alpha_normal:
            print(F.softmax(alpha, dim=-1) if self.sparsity == 1 else
                  entmax_bisect(alpha, dim=-1, alpha=self.sparsity))

        print("\n# Alpha - reduce")
        for alpha in self.alpha_reduce:
            print(F.softmax(alpha, dim=-1) if self.sparsity == 1 else
                  entmax_bisect(alpha, dim=-1, alpha=self.sparsity))
        print("#####################")

    def genotype(self):
        gene_normal = gt.parse(self.alpha_normal, k=2)
        gene_reduce = gt.parse(self.alpha_reduce, k=2)
        concat = range(2, 2 + self.n_nodes)  # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat,
                           reduce=gene_reduce, reduce_concat=concat)

    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p
