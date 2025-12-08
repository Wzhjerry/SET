import os

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies.ddp import DDPStrategy
from torch.utils.data import DataLoader

from datasets import load_dataset
from model import load_model
from utils.losses import load_criteria

from sklearn.ensemble import AdaBoostRegressor


# Train worker: worker for training segmentation task
def train_worker(args):
    kfold = 5 if args.kfold else 1

    for fold in range(kfold):
        if args.resume != -1:
            fold = args.resume
        train_dataset, val_dataset = load_dataset(args, fold=fold, train=True)
        criteria = load_criteria(args)
        model = load_model(criteria, args)
        # model = torch.compile(model)
        epoch = args.epoch
        save_path = os.path.join(args.save_name, str(args.seed))

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            persistent_workers=False,
            num_workers=args.workers,
            pin_memory=False,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            persistent_workers=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )

        # optionally resume from a checkpoint
        model_path = None
        if args.resume == fold:
            model_path = "{}{}/model_{}_segmentation_checkpoint_{}.ckpt".format(args.save_name, str(args.seed), str(args.dataset_name), str(fold))
            if os.path.exists(model_path):
                print("=> loading checkpoint '{}'".format(model_path))
            else:
                print("=> no checkpoint found at '{}'".format(model_path))
                model_path = None

        checkpoint_best = ModelCheckpoint(
            dirpath=save_path,
            monitor="Val/Jac" if not args.inpainting else "Val/InpLoss",
            mode="max" if not args.inpainting else "min",
            filename="model_{}_{}_best_{}".format(
                str(args.dataset_name),
                "segmentation" if not args.inpainting else "inpainting",
                str(fold)
            )
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=save_path,
            filename="model_{}_{}_checkpoint_{}".format(
                str(args.dataset_name),
                "segmentation" if not args.inpainting else "inpainting",
                str(fold)
            )
        )

        trainer = Trainer(
            accelerator='gpu',
            devices=args.gpu,
            strategy=DDPStrategy(find_unused_parameters=False),
            logger=args.logger,
            callbacks=[checkpoint_best, checkpoint_callback],
            max_epochs=epoch,
            log_every_n_steps=10
        )

        trainer.fit(model, train_loader, val_loader, ckpt_path=model_path)
