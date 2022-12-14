import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
from dataset import MaskBaseDataset
from utils import RAdam, AdamW, PlainRAdam
from loss import create_criterion
import wandb
from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt_s = gts[0] + gts[1]
        pred_s = preds[0] + preds[1]

        gt = gt_s[choice].item()
        pred = pred_s[choice].item()
        
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    mask_num_classes = dataset.mask_num_classes  # 3
    age_gender_num_classes = dataset.age_gender_num_classes  # 6

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    # dataset.set_transform(transform)
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()
    # val_set.dataset.set_transform(transform.transformations['val'])
    # train_set.dataset.set_transform(transform.transformations['train'])

    train_loader = DataLoader(
        # CutMix(train_set, num_class=9,beta=1.0,prob=0.5,num_mix=2),
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )
    # wandb
    # wandb.init(config=args)

    # -- model
    model = Ensemble(
        mask_num_classes = mask_num_classes,
        age_gender_num_classes = age_gender_num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # wandb.watch(model, log_freq=100)

    # -- loss & metric
    criterion_mask = create_criterion(args.ag_criterion)  # default: cross_entropy
    criterion_age_gender = create_criterion(args.ag_criterion)  # default: focal
    # criterion_mask = CutMixCrossEntropyLoss(True)  # default: cross_entropy
    # criterion_age_gender = CutMixCrossEntropyLoss(True)  # default: focal

    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    # optimizer = opt_module(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=args.lr,
    #     weight_decay=5e-4,
    #     betas=(0.9,0.99)
    # )

    # -- optimizer는 utils.py에서 불러옴
    optimizer = RAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )

    # scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min')

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        # train loop
        dataset.set_transform(transform.transformations['train'])
        model.train()

        loss_value = 0
        matches = 0

        for idx, train_batch in enumerate(train_loader):
            inputs, (mask_labels, age_gender_labels) = train_batch
            inputs = inputs.to(device)
            mask_labels = mask_labels.to(device)
            age_gender_labels = age_gender_labels.to(device)

            optimizer.zero_grad()

            mask_outs, age_gender_outs = model(inputs)

            mask_preds = torch.argmax(mask_outs, dim=-1)
            age_gender_preds = torch.argmax(age_gender_outs, dim=-1)
            # print("mask:", mask_preds.shape, mask_preds, mask_labels)
            # print("age_gender:", age_gender_preds.shape, age_gender_preds, age_gender_labels)

            mask_loss = criterion_mask(mask_outs, mask_labels)
            age_gender_loss = criterion_age_gender(age_gender_outs, age_gender_labels)

            loss = mask_loss + 1.5*age_gender_loss

            loss.backward()
            optimizer.step()
            loss_value += loss.item()

            preds = mask_preds*6 + age_gender_preds
            labels = mask_labels*6 + age_gender_labels
            
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0

        # scheduler.step()

        # val loop
        dataset.set_transform(transform.transformations['val'])
        with torch.no_grad():
            print("Calculating validation results...")
            
            model.eval()
            
            val_loss_items = []
            val_acc_items = []
            figure = None
            for val_batch in val_loader:
                inputs, (mask_labels, age_gender_labels) = val_batch
                inputs = inputs.to(device)

                mask_labels = mask_labels.to(device)
                age_gender_labels = age_gender_labels.to(device)

                mask_outs, age_gender_outs = model(inputs)
                
                mask_preds = torch.argmax(mask_outs, dim=-1)
                age_gender_preds = torch.argmax(age_gender_outs, dim=-1)

                mask_loss_item = criterion_mask(mask_outs, mask_labels).item()
                age_gender_loss_item = criterion_age_gender(age_gender_outs, age_gender_labels).item()

                # mask_acc_item = (mask_labels == mask_preds).sum().item()
                # age_gender_acc_item = (age_gender_labels == age_gender_preds).sum().item()
                # print("mask_acc_item:", mask_acc_item, "age_gender_acc_item:", age_gender_acc_item)

                loss_item = mask_loss_item + age_gender_loss_item
                ## acc_item = mask_acc_item + age_gender_acc_item

                acc_item = 0
                for idx in range(args.valid_batch_size):
                    if mask_labels[idx] == mask_preds[idx] and age_gender_labels[idx] == age_gender_preds[idx]:
                        acc_item += 1

                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, (mask_labels, age_gender_labels), (mask_preds, age_gender_preds), n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)

            scheduler.step(val_loss)

            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_figure("results", figure, epoch)
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[224, 224], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=64, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--m_criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--ag_criterion', type=str, default='focal', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
