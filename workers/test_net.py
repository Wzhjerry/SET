import os

from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from datasets import load_dataset
from model import load_model


def test_worker(args):
    kfold = 5 if args.kfold else 1
    print(args.seed)
    save_path = os.path.join(args.save_name, str(args.seed))

    for fold in range(kfold):
        args.fold = fold
        test_dataset = load_dataset(args, fold=fold, train=False)
        model = load_model(criteria=[None], args=args)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )

        # load corresponding model
        if not args.inpainting:
            best_model_path = "{}/model_{}_segmentation_best_{}.ckpt".format(
                save_path, args.dataset_name, fold
            )
        else:
            best_model_path = "{}/model_{}_inpainting_best_{}.ckpt".format(
                save_path, args.dataset_name, fold
            )
        # model.load_weights(best_model_path)
        model.init_weights(best_model_path)

        trainer = Trainer(accelerator="gpu", devices=[args.gpu[0]], logger=args.logger)
        trainer.test(model, dataloaders=test_loader)
