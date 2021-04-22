import copy
from typing import Dict

import numpy as np
import plotly.graph_objects as go
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from plotly.subplots import make_subplots

from epe_darts import genotypes as gt, utils
from epe_darts.architect import Architect


class SearchController(pl.LightningModule):
    def __init__(self, net: nn.Module,
                 w_lr=0.025, w_momentum=0.9, w_weight_decay: float = 3e-4, w_lr_min: float = 0.001, w_grad_clip=5.,
                 nesterov=False,
                 alpha_lr=3e-4, alpha_weight_decay=1e-3,
                 max_epochs: int = 50):
        super().__init__()
        self.save_hyperparameters('w_lr', 'w_momentum', 'w_weight_decay', 'w_lr_min', 'w_grad_clip',
                                  'alpha_lr', 'alpha_weight_decay', 'max_epochs')
        self.automatic_optimization = False

        self.w_lr: float = w_lr
        self.w_momentum: float = w_momentum
        self.w_weight_decay: float = w_weight_decay
        self.w_lr_min: float = w_lr_min
        self.w_grad_clip: float = w_grad_clip
        self.nesterov: bool = nesterov
        self.alpha_lr: float = alpha_lr
        self.alpha_weight_decay: float = alpha_weight_decay
        self.max_epochs: int = max_epochs

        self.epoch2normal_alphas: Dict = {}
        self.epoch2reduce_alphas: Dict = {}

        self.net: nn.Module = net
        self.net_copy: nn.Module = copy.deepcopy(net)
        self.architect = Architect(self.net, self.net_copy, self.w_momentum, self.w_weight_decay)

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx != 0:
            return

        (trn_X, trn_y), (val_X, val_y) = batch
        w_optim, alpha_optim = self.optimizers()

        # phase 2. architect step (alpha)
        alpha_optim.zero_grad()
        w_lr = self.w_scheduler['scheduler'].get_last_lr()[-1]
        self.architect.unrolled_backward(trn_X, trn_y, val_X, val_y, w_lr, w_optim)
        alpha_optim.step()

        # phase 1. child network step (w)
        w_optim.zero_grad()
        logits = self.net(trn_X)
        loss = self.net.criterion(logits, trn_y)
        self.manual_backward(loss)

        # gradient clipping
        nn.utils.clip_grad_norm_(self.net.weights(), self.w_grad_clip)
        w_optim.step()

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        self.log('train_loss', loss)
        self.log('train_top1', prec1)
        self.log('train_top5', prec5)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            logits = self.net(x)
            loss = self.net.criterion(logits, y)
            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))

        self.log('valid_loss', loss)
        self.log('valid_top1', prec1)
        self.log('valid_top5', prec5)

    def configure_optimizers(self):
        w_optim = torch.optim.SGD(self.net.weights(), self.w_lr,
                                  momentum=self.w_momentum, weight_decay=self.w_weight_decay, nesterov=self.nesterov)
        self.w_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(w_optim, self.max_epochs, eta_min=self.w_lr_min),
            'interval': 'epoch',
        }

        alpha_optim = torch.optim.Adam(self.net.alphas(), self.alpha_lr,
                                       betas=(0.5, 0.999), weight_decay=self.alpha_weight_decay)
        self.alpha_scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(alpha_optim, lr_lambda=lambda x: 1),
            'interval': 'epoch',
        }
        return [w_optim, alpha_optim], [self.w_scheduler, self.alpha_scheduler]

    def on_validation_epoch_end(self):
        # log genotype
        epoch = self.trainer.current_epoch

        genotype = self.net.genotype(algorithm='top-k')
        gt.plot(genotype.normal, f'normal-top2-{epoch}')
        gt.plot(genotype.reduce, f'reduction-top2-{epoch}')
        wandb.log({'normal-top2-cell': wandb.Image(f'normal-top2-{epoch}.png')})
        wandb.log({'reduction-top2-cell': wandb.Image(f'reduction-top2-{epoch}.png')})
        print('Genotype with top-2 connections:', genotype)

        genotype = self.net.genotype(algorithm='best')
        gt.plot(genotype.normal, f'normal-best-{epoch}')
        gt.plot(genotype.reduce, f'reduction-best-{epoch}')
        wandb.log({'normal-best-cell': wandb.Image(f'normal-best-{epoch}.png')})
        wandb.log({'reduction-best-cell': wandb.Image(f'reduction-best-{epoch}.png')})
        print('Genotype with nonzero connections:', genotype)

        alpha_normal, alpha_reduce = self.net.alpha_weights()

        self.epoch2normal_alphas[epoch] = [alpha.detach().cpu().numpy() for alpha in alpha_normal]
        self.epoch2reduce_alphas[epoch] = [alpha.detach().cpu().numpy() for alpha in alpha_reduce]

        normal_fig = self.plot_alphas(self.epoch2normal_alphas)
        reduce_fig = self.plot_alphas(self.epoch2reduce_alphas)

        wandb.log({'Normal cell alpha change throughout epochs': normal_fig})
        wandb.log({'Reduce cell alpha change throughout epochs': reduce_fig})
        if epoch != 0:
            normal_diff = np.sum([np.sum(np.abs(cur - prev)) for cur, prev in zip(self.epoch2normal_alphas[epoch],
                                                                                  self.epoch2normal_alphas[epoch - 1])])
            reduce_diff = np.sum([np.sum(np.abs(cur - prev)) for cur, prev in zip(self.epoch2reduce_alphas[epoch],
                                                                                  self.epoch2reduce_alphas[epoch - 1])])
            self.log('normal_diff', normal_diff)
            self.log('reduce_diff', reduce_diff)

        print(f'Sparsity: {self.net.sparsity}')
        print("####### ALPHA #######")
        print("\n# Alpha - normal")
        for alpha in alpha_normal:
            print(alpha)
        print("\n# Alpha - reduce")
        for alpha in alpha_reduce:
            print(alpha)

    def plot_alphas(self, epoch2alphas: Dict):
        epochs = len(epoch2alphas)

        fig = make_subplots(rows=self.net.n_nodes, cols=self.net.n_nodes + 1,
                            subplot_titles=[f'{node1} ➜ {node2}' if node1 < node2 else ''
                                            for node2 in range(2, self.net.n_nodes + 2)
                                            for node1 in range(self.net.n_nodes + 1)])

        for node1 in range(2, self.net.n_nodes + 2):
            for node2 in range(node1):
                for connection_id, connection_name in enumerate(self.net.primitives):
                    fig.add_trace(
                        go.Scatter(x=list(range(epochs)),
                                   y=[epoch2alphas[epoch][node1 - 2][node2][connection_id] for epoch in range(epochs)],
                                   name=connection_name),
                        row=node1 - 1,
                        col=node2 + 1,
                    )

        fig.update_layout(height=1000, width=1000)
        return fig
