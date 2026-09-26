import argparse
from argparse import ArgumentParser
import importlib.util
import math
import os
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from fastgeco.backbones.shared import BackboneRegistry
from geco.data_module import SpecsDataModule
from geco.sdes import SDERegistry
from fastgeco.model import ScoreModel


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


class ExactBestMetricCheckpoints(pl.Callback):
     """Keep one descriptively named best checkpoint per validation metric."""

     FILES = {
          "pesq": "best_pesq",
          "estoi": "best_estoi",
          "si_sdr": "best_sisdr",
     }

     def __init__(self, dirpath):
          self.dirpath = Path(dirpath)
          self.best = {metric: float("-inf") for metric in self.FILES}
          self.best_epochs = {metric: None for metric in self.FILES}
          self.best_paths = {metric: None for metric in self.FILES}

     @staticmethod
     def _legacy_value(item):
          if isinstance(item, dict):
               return item.get("value", float("-inf"))
          return item

     def on_validation_epoch_end(self, trainer, pl_module):
          if trainer.sanity_checking or not trainer.is_global_zero:
               return
          self.dirpath.mkdir(parents=True, exist_ok=True)
          for metric, prefix in self.FILES.items():
               value = trainer.callback_metrics.get(metric)
               if value is None:
                    continue
               value = float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
               if not math.isfinite(value) or value <= self.best[metric]:
                    continue
               epoch = int(trainer.current_epoch) + 1
               filename = f"{prefix}.ckpt"
               new_path = self.dirpath / filename
               old_value = self.best[metric]
               old_epoch = self.best_epochs[metric]
               old_path = self.best_paths[metric]
               self.best[metric] = value
               self.best_epochs[metric] = epoch
               self.best_paths[metric] = str(new_path)
               try:
                    trainer.save_checkpoint(str(new_path))
               except Exception:
                    self.best[metric] = old_value
                    self.best_epochs[metric] = old_epoch
                    self.best_paths[metric] = old_path
                    raise

     def state_dict(self):
          return {
               "best": dict(self.best),
               "best_epochs": dict(self.best_epochs),
               "best_paths": dict(self.best_paths),
          }

     def load_state_dict(self, state_dict):
          raw_best = state_dict.get("best", {})
          self.best = {
               metric: float(self._legacy_value(raw_best.get(metric, float("-inf"))))
               for metric in self.FILES
          }
          self.best_epochs = dict(state_dict.get("best_epochs", {}))
          self.best_paths = dict(state_dict.get("best_paths", {}))
          for metric in self.FILES:
               self.best_epochs.setdefault(metric, None)
               self.best_paths.setdefault(metric, None)


if __name__ == '__main__':
     parser = ArgumentParser()
     parser.add_argument("--batch_size", type=int, default=16,  help="Training batch size")
     parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Number of micro-batches accumulated per optimizer step.")
     parser.add_argument("--t_rsp_min", type=float, default=0.5,  help="Minimum reverse starting point during training")
     parser.add_argument("--t_rsp_max", type=float, default=0.5,  help="Maximum reverse starting point during training")
     parser.add_argument("--pre_ckpt", type=str, default='./logs/u0kwl5bj/epoch=13-si_sdr=8.10.ckpt',  help="Load ckpt")
     parser.add_argument("--resume_ckpt", type=str, default="", help="Strictly resume a Lightning training run from this checkpoint, including optimizer, scheduler, epoch, and global_step.")
     parser.add_argument("--train_dir", type=str, default="", help="Override train data directory from the loaded checkpoint.")
     parser.add_argument("--val_dir", type=str, default="", help="Override validation data directory from the loaded checkpoint.")
     parser.add_argument("--test_dir", type=str, default="", help="Override test data directory from the loaded checkpoint.")
     parser.add_argument("--nolog", action='store_true', help="Turn off logging (for development purposes)")
     parser.add_argument("--wandb", action='store_true', help="Enable Weights & Biases logging (default: local TensorBoard)")
     parser.add_argument("--lr", type=float, default=1e-5, help="The learning rate (1e-4 by default)")
     parser.add_argument("--loss_type", type=str, default="default", help="The type of loss function to use.")
     parser.add_argument("--num_eval_files", type=int, default=20, help="Number of validation files used for PESQ/SI-SDR checkpoint metrics. Use 0 to save only last checkpoint.")
     parser.add_argument("--inference_start", type=float, default=0.5, help="inference start")
     parser.add_argument("--max_epochs", type=int, default=30, help="Number of training epochs")
     parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many optimizer steps; -1 disables the limit.")
     parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Ratio of total training steps used for linear LR warmup.")
     parser.add_argument("--checkpoint_dir", type=str, default="", help="Directory for local checkpoints. Defaults to a logger folder, or ./logs/fastgeco-local when --nolog is used.")
     parser.add_argument("--gpus", type=int, default=1, help="Compatibility flag. Use 0 for CPU, otherwise GPU if available.")
     parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducible training.")
     parser.add_argument("--role_loss_weight", type=float, default=0.0, help="Weight for E2 syllable-role loss. 0 disables role loss.")
     parser.add_argument("--role_loss_warmup_steps", type=int, default=5000, help="Disable role loss for the first N optimizer steps.")
     parser.add_argument("--role_mask_dir", type=str, default="", help="Directory containing E1 *_role_mask.npz files, e.g. zang_data/train/role_masks.")
     parser.add_argument("--role_onset_weight", type=float, default=3.0, help="Onset-like subloss weight inside the role loss.")
     parser.add_argument("--role_nucleus_weight", type=float, default=2.0, help="Nucleus-like subloss weight inside the role loss.")
     parser.add_argument("--role_transition_weight", type=float, default=0.0, help="Transition-like subloss weight inside the role loss.")
     parser.add_argument("--role_highband_start_hz", type=float, default=1800.0, help="High-band cutoff for onset transient consistency.")
     args = parser.parse_args()
     if args.accumulate_grad_batches < 1:
          parser.error("--accumulate_grad_batches must be at least 1")
     if args.role_loss_weight > 0 and not os.path.isdir(args.role_mask_dir):
          parser.error("--role_mask_dir must exist when --role_loss_weight is positive")
     pl.seed_everything(args.seed, workers=True)
     checkpoint_file = args.resume_ckpt if args.resume_ckpt else args.pre_ckpt

    # Load score model
     model = ScoreModel.load_from_checkpoint(
        checkpoint_file,
        strict=True,
        batch_size=16, num_workers=0, kwargs=dict(gpu=False)
     )
     model.add_para(args.t_rsp_min, args.t_rsp_max, 
                    args.batch_size, args.loss_type, args.lr,
                    args.inference_start, args.warmup_ratio,
                    role_loss_weight=args.role_loss_weight,
                    role_loss_warmup_steps=args.role_loss_warmup_steps,
                    role_mask_dir=args.role_mask_dir,
                    role_onset_weight=args.role_onset_weight,
                    role_nucleus_weight=args.role_nucleus_weight,
                    role_transition_weight=args.role_transition_weight,
                    role_highband_start_hz=args.role_highband_start_hz)
     if args.train_dir:
          model.data_module.train_dir = args.train_dir
     if args.val_dir:
          model.data_module.val_dir = args.val_dir
     if args.test_dir:
          model.data_module.test_dir = args.test_dir
     model.num_eval_files = args.num_eval_files
     model.data_module.gpu = args.gpus > 0
     if args.gpus:
          model.cuda()
          model.to('cuda:0')
     
     

     if args.nolog:
          logger = False
          savedir_ck = args.checkpoint_dir or './logs/fastgeco-local'
     elif args.wandb:
          if importlib.util.find_spec("wandb") is None:
               parser.error("--wandb requires the optional 'wandb' package; install it with 'pip install wandb'.")
          import wandb
          if not wandb.login():
               parser.error("W&B authentication failed. Run 'wandb login' or set WANDB_API_KEY, then retry.")
          logger = WandbLogger(project="fastgeco", log_model=True, save_dir="logs")
          logger.log_hyperparams(vars(args))
          savedir_ck = args.checkpoint_dir or os.path.join('./logs', logger.version)
     else:
          logger = TensorBoardLogger(save_dir="logs", name="fastgeco")
          logger.log_hyperparams(vars(args))
          savedir_ck = args.checkpoint_dir or logger.log_dir
     os.makedirs(savedir_ck, exist_ok=True)

     # Set up callbacks for logger
     callbacks = [CleanTQDMProgressBar(refresh_rate=0)]
     if model.num_eval_files:
          # Run before Lightning's last-checkpoint callback so last.ckpt gets
          # the same best-metric metadata as the metric-specific checkpoints.
          callbacks.append(ExactBestMetricCheckpoints(savedir_ck))
     callbacks.append(ModelCheckpoint(
               dirpath=savedir_ck,
               save_top_k=0,
               save_last=True,
               filename="last",
               auto_insert_metric_name=False,
          ))
     # Initialize the Trainer and the DataModule
     trainer_kwargs = dict(
          logger=logger,
          log_every_n_steps=10,
          num_sanity_val_steps=0,
          accelerator="gpu" if args.gpus else "cpu",
          devices=args.gpus if args.gpus > 0 else 1,
          max_epochs=args.max_epochs,
          max_steps=args.max_steps,
          accumulate_grad_batches=args.accumulate_grad_batches,
          callbacks=callbacks,
     )
     if args.gpus > 1:
          trainer_kwargs["strategy"] = DDPPlugin(find_unused_parameters=False)
     trainer = pl.Trainer(**trainer_kwargs)


     # Train model
     if args.resume_ckpt:
          trainer.fit(model, ckpt_path=args.resume_ckpt)
     else:
          trainer.fit(model)
