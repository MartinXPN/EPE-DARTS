""" Training augmented model """
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger

from epe_darts.augment_cnn import AugmentCNN
from epe_darts.config import AugmentConfig
from epe_darts.data import AugmentDataModule
from epe_darts.utils import fix_random_seed, ExperimentSetup

config = AugmentConfig()
experiment = ExperimentSetup(name='vanilla-darts', create_latest=True, long_description="""
Trying out Pytorch Lightning
""")
print(config)


def main():
    fix_random_seed(config.seed, fix_cudnn=True)
    data = AugmentDataModule(config.dataset, config.data_path, split_train=False,
                             cutout_length=config.cutout_length, batch_size=config.batch_size, workers=4)
    data.setup()

    model = AugmentCNN(data.input_size, data.input_channels, config.init_channels, data.n_classes, config.layers,
                       config.aux_weight, config.genotype, stem_multiplier=3,
                       lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay,
                       max_epochs=config.epochs)

    loggers = [
        CSVLogger(experiment.log_dir, name='history'),
        TensorBoardLogger(experiment.log_dir, name=experiment.name, default_hp_metric=False),
        WandbLogger(name=experiment.name, save_dir=experiment.log_dir, project='epe-darts', save_code=True, notes=experiment.long_description),
        # AimLogger(experiment=experiment.name),
    ]

    trainer = Trainer(logger=loggers, log_every_n_steps=config.print_freq,
                      gpus=config.gpus if torch.cuda.is_available() else None, auto_select_gpus=True,
                      max_epochs=config.epochs, terminate_on_nan=True,
                      gradient_clip_val=config.grad_clip,
                      callbacks=[
                          # EarlyStopping(monitor='valid_top1', patience=5, verbose=True, mode='max'),
                          ModelCheckpoint(dirpath=experiment.model_save_path, filename='model-{epoch:02d}-{valid_top1:.2f}', monitor='valid_top1', save_top_k=5, verbose=True, mode='max'),
                          LearningRateMonitor(logging_interval='epoch'),
                      ])

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
