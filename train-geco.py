import argparse
from argparse import ArgumentParser
import importlib.util
import os
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from geco.backbones.shared import BackboneRegistry
from geco.data_module import SpecsDataModule
from geco.sdes import SDERegistry
from geco.model import ScoreModel


def get_argparse_groups(parser):
     groups = {}
     for group in parser._action_groups:
          group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
          groups[group.title] = argparse.Namespace(**group_dict)
     return groups


class CleanTQDMProgressBar(TQDMProgressBar):
     def init_train_tqdm(self):
          bar = super().init_train_tqdm()
          bar.leave = False
          return bar

     def init_validation_tqdm(self):
          bar = super().init_validation_tqdm()
          bar.leave = False
          return bar

     def init_test_tqdm(self):
          bar = super().init_test_tqdm()
          bar.leave = False
          return bar


if __name__ == '__main__':
     # throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
     base_parser = ArgumentParser(add_help=False)
     parser = ArgumentParser()
     for parser_ in (base_parser, parser):
          parser_.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp")
          parser_.add_argument("--sde", type=str, choices=SDERegistry.get_all_names(), default="bbed")    
          parser_.add_argument("--nolog", action='store_true', help="Turn off logging (for development purposes)")
          parser_.add_argument("--wandb", action='store_true', help="Enable Weights & Biases logging (default: local TensorBoard)")
          parser_.add_argument("--checkpoint_dir", type=str, default="", help="Directory for local checkpoints. Defaults to a logger folder, or ./logs/geco-local when --nolog is used.")
          parser_.add_argument("--seed", type=int, default=1337, help="Random seed for reproducible training.")
     temp_args, _ = base_parser.parse_known_args()

     # Add specific args for ScoreModel, pl.Trainer, the SDE class and backbone DNN class
     backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
     sde_class = SDERegistry.get_by_name(temp_args.sde)
     parser = pl.Trainer.add_argparse_args(parser)
     parser.set_defaults(max_epochs=30)
     ScoreModel.add_argparse_args(
          parser.add_argument_group("ScoreModel", description=ScoreModel.__name__))
     sde_class.add_argparse_args(
          parser.add_argument_group("SDE", description=sde_class.__name__))
     backbone_cls.add_argparse_args(
          parser.add_argument_group("Backbone", description=backbone_cls.__name__))
     # Add data module args
     data_module_cls = SpecsDataModule
     data_module_cls.add_argparse_args(
          parser.add_argument_group("DataModule", description=data_module_cls.__name__))
     # Parse args and separate into groups
     args = parser.parse_args()
     pl.seed_everything(args.seed, workers=True)
     arg_groups = get_argparse_groups(parser)

     # Initialize logger, trainer, model, datamodule
     model = ScoreModel(
          backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
          **{
               **vars(arg_groups['ScoreModel']),
               **vars(arg_groups['SDE']),
               **vars(arg_groups['Backbone']),
               **vars(arg_groups['DataModule'])
          }
     )
 
     if args.nolog:
          logger = False
          savedir_ck = args.checkpoint_dir or './logs/geco-local'
     elif args.wandb:
          if importlib.util.find_spec("wandb") is None:
               parser.error("--wandb requires the optional 'wandb' package; install it with 'pip install wandb'.")
          import wandb
          if not wandb.login():
               parser.error("W&B authentication failed. Run 'wandb login' or set WANDB_API_KEY, then retry.")
          logger = WandbLogger(project="geco", log_model=True, save_dir="logs")
          logger.log_hyperparams(vars(args))
          savedir_ck = args.checkpoint_dir or os.path.join('./logs', logger.version)
     else:
          logger = TensorBoardLogger(save_dir="logs", name="geco")
          logger.log_hyperparams(vars(args))
          savedir_ck = args.checkpoint_dir or logger.log_dir
     os.makedirs(savedir_ck, exist_ok=True)




     # Set up callbacks for logger
     callbacks = [
          CleanTQDMProgressBar(refresh_rate=0),
          ModelCheckpoint(dirpath=savedir_ck, save_last=True, filename='{epoch}-last'),
     ]
     if args.num_eval_files:
          callbacks.extend([ 
               ModelCheckpoint(dirpath=savedir_ck, save_top_k=1, monitor="estoi", mode="max", filename='{epoch}-estoi-best-{estoi:.3f}'),
               ModelCheckpoint(dirpath=savedir_ck, save_top_k=1, monitor="si_sdr", mode="max", filename='{epoch}-sisdr-best-{si_sdr:.2f}'),
               ModelCheckpoint(dirpath=savedir_ck, save_top_k=1, monitor="pesq", mode="max", filename='{epoch}-pesq-best-{pesq:.2f}'),
          ])
     # Initialize the Trainer and the DataModule
     trainer = pl.Trainer.from_argparse_args(
          arg_groups['pl.Trainer'],
          strategy=DDPPlugin(find_unused_parameters=False), logger=logger,
          log_every_n_steps=10, num_sanity_val_steps=0,
          callbacks=callbacks,
     )

     # Train model
     trainer.fit(model)
