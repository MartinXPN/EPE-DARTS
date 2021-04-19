""" Search cell """
from typing import Union, List, Optional

import fire
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger

from epe_darts.architect import Architect
from epe_darts.data import DataModule
from epe_darts.search_cnn import SearchCNNController
from epe_darts.utils import fix_random_seed, ExperimentSetup

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


def main(name: str, dataset: str, batch_size: int = 64, data_path: str = 'data/',
         w_lr: float = 0.025, w_lr_min: float = 0.001, w_momentum: float = 0.9, w_weight_decay: float = 3e-4,
         w_grad_clip: float = 5.,
         print_freq: int = 50, gpus: Union[int, List[int]] = -1, workers: int = 4, epochs: int = 50,
         init_channels: int = 16, layers: int = 8, seed: int = 42,
         sparsity: float = 4,
         alpha_lr: float = 3e-4, alpha_weight_decay: float = 1e-3, alphas_path: Optional[str] = None):
    """
    :param name: Experiment name
    :param dataset: CIFAR10 / CIFAR100 / ImageNet / MNIST / FashionMNIST
    :param data_path: Path to the dataset (download in that location if not present)
    :param batch_size: Batch size
    :param w_lr: Learning rate for network weights
    :param w_lr_min: Minimum learning rate for network weights
    :param w_momentum: Momentum for network weights
    :param w_weight_decay: Weight decay for network weights
    :param w_grad_clip: Gradient clipping threshold for network weights
    :param print_freq: Logging frequency
    :param gpus: Lis of GPUs to use or a single GPU (will be ignored if no GPU is available)
    :param epochs: # of training epochs
    :param init_channels: Initial channels
    :param layers: # of layers in the network (number of cells)
    :param seed: Random seed
    :param workers: # of workers for data loading
    :param sparsity: Entmax(sparisty) for alphas [1 is equivalent to Softmax]
    :param alpha_lr: Learning rate for alphas
    :param alpha_weight_decay: Weight decay for alphas
    :param alphas_path: Optional path for initial alphas (will be loaded as a torch file)
    """
    hyperparams = locals()
    # set seed
    fix_random_seed(seed, fix_cudnn=True)
    experiment = ExperimentSetup(name=name, create_latest=True, long_description="""
        Trying out Pytorch Lightning
    """)

    data = DataModule(dataset=dataset, data_dir=data_path, split_train=True,
                      cutout_length=0, batch_size=batch_size, workers=workers)
    data.setup()

    alpha_normal, alpha_reduce = torch.load(alphas_path) if alphas_path else (None, None)
    model = SearchCNNController(data.input_channels, init_channels, data.n_classes, layers,
                                sparsity=sparsity, alpha_normal=alpha_normal, alpha_reduce=alpha_reduce).to(device)
    model.architect = Architect(model, model.w_momentum, model.w_weight_decay)

    # callbacks = [
    #     RankingChangeEarlyStopping(monitor_param=param, patience=10)
    #     for name, param in model.named_parameters()
    #     if 'alpha_normal' in name
    # ]

    loggers = [
        CSVLogger(experiment.log_dir, name='history'),
        TensorBoardLogger(experiment.log_dir, name=experiment.name, default_hp_metric=False),
        WandbLogger(name=experiment.name, save_dir=experiment.log_dir, project='epe-darts', save_code=True, notes=experiment.long_description),
        # AimLogger(experiment=experiment.name),
    ]
    for logger in loggers:
        logger.log_hyperparams(hyperparams)

    trainer = Trainer(logger=loggers, log_every_n_steps=print_freq,
                      gpus=-1 if torch.cuda.is_available() else None,
                      max_epochs=epochs, terminate_on_nan=True,
                      callbacks=[
                          # EarlyStopping(monitor='valid_top1', patience=5, verbose=True, mode='max'),
                          ModelCheckpoint(dirpath=experiment.model_save_path, filename='model-{epoch:02d}-{valid_top1:.2f}', monitor='valid_top1', save_top_k=5, verbose=True, mode='max'),
                          LearningRateMonitor(logging_interval='epoch'),
                      ])

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    fire.Fire(main)
