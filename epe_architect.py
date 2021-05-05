import itertools
from pathlib import Path
from pprint import pprint
from typing import Dict

import fire
import numpy as np
import torch
from pytorch_lightning.core.saving import load_hparams_from_yaml
from torch import nn

from epe_darts.controller import SearchController
from epe_darts.data import DataModule
from epe_darts.epe_nas import get_batch_jacobian, eval_score_per_class
from epe_darts.search_cnn import SearchCNNController
from epe_darts.utils import fix_random_seed, PathLike

fix_random_seed(42, fix_cudnn=True)
# torch.autograd.set_detect_anomaly(True)


def extract_architecture(darts_model_path: PathLike,
                         hparams_file: PathLike,
                         dataset: str,
                         nb_architectures: int = 500,
                         batch_size: int = 32,
                         workers: int = 4,
                         data_path: Path = Path('datasets'),
                         save_path: Path = Path('epe_architecture')):
    data = DataModule(dataset=dataset, data_dir=data_path, split_train=False, cutout_length=0,
                      batch_size=batch_size, workers=workers)
    data.setup()
    data_iterator = itertools.cycle(data.val_dataloader())

    hparams = load_hparams_from_yaml(hparams_file)
    print('Hyper-params from', hparams_file, ':', hparams)

    # Setup and load networks
    net = SearchCNNController(input_channels=data.input_channels, n_classes=data.n_classes, n_layers=8, **hparams)
    SearchController.load_from_checkpoint(darts_model_path, net=net, image_log_path=Path('alphas'))

    n_nodes = len(net.alpha_normal)
    n_ops = net.alpha_normal[0].shape[-1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'#Nodes: {n_nodes}, #Ops: {n_ops}, device: {device}')

    scores: Dict[str, float] = {}
    for architecture in range(nb_architectures):
        for i in range(n_nodes):
            p = 3 / ((i + 2) * n_ops)
            normal = np.random.choice([0., 1.], size=(i + 2, n_ops), p=[1 - p, p])
            reduce = np.random.choice([0., 1.], size=(i + 2, n_ops), p=[1 - p, p])
            normal[:, -1] = 0.5
            reduce[:, -1] = 0.5
            net.alpha_normal[i] = nn.Parameter(torch.from_numpy(normal))
            net.alpha_reduce[i] = nn.Parameter(torch.from_numpy(reduce))

        net = net.to(device)
        x, target = next(data_iterator)
        x = x.to(device)
        jacobs_batch = get_batch_jacobian(net, x)
        jacobs = jacobs_batch.reshape(jacobs_batch.size(0), -1).detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        s = eval_score_per_class(jacobs, target, n_classes=data.n_classes)

        genotype = net.genotype(algorithm='best')
        print(s, '\t', genotype)
        scores[f'{genotype}'] = s

    top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5]
    print('---- TOP 5: ----')
    pprint(top)


if __name__ == '__main__':
    fire.Fire(extract_architecture)
