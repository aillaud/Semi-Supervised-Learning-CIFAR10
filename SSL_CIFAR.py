#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy
import logging
import math
import os
import random
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import Dataset, Subset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from wideresnet import WideResNet

# Inspiration : https://github.com/kekmodel/FixMatch-pytorch

# Variables gloables, reprises des paramètres optimaux donnés dans l'article
nb_labels = 25
nb_classes = 10
lr = 0.03
bs = 64
mu = 7
weight_decay = 5e-4
ema_decay = 0.999
epochs = 1024
lambda_u = 1
temperature = 1
threshold = 0.95
random.seed(5)
np.random.seed(5)
torch.manual_seed(5)
device = torch.device("cuda")
writer = SummaryWriter("./logs")


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    """ Fonction pour enregistrer le meilleur modèle """
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def calc_images_moments(dataset, batch_size=None):
    """ Calcule la moyenne et l'écart-type par canal pour la normalisation initiale des images """
    batch_size = batch_size if batch_size is not None else len(dataset)
    count = len(dataset) * (dataset[0][0].shape[1] * dataset[0][0].shape[2])
    tmp_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    mean, var = torch.zeros(3), torch.zeros(3)
    for images, _ in tqdm(tmp_loader, leave=False):
        mean += images.double().sum(axis=[0, 2, 3])
        var += (images.double() ** 2).sum(axis=[0, 2, 3])
    mean = mean / count
    var = (var / count) - mean**2
    std = torch.sqrt(var)

    return mean.tolist(), std.tolist()


class UnlabeledDataset(Dataset):
    """ Datasets that returns a tuple of weakly_augmented and strongly_augmented image
        for each id
    """
    def __init__(self, weakly_augmented, strongly_augmented):
        self.weakly_augmented = weakly_augmented
        self.strongly_augmented = strongly_augmented

    def __len__(self):
        assert len(self.weakly_augmented) == len(self.strongly_augmented)
        return len(self.weakly_augmented)

    def __getitem__(self, idx):
        return self.weakly_augmented[idx], self.strongly_augmented[idx]


def prepare_datasets():
    """ Génère les 3 datasets nécessaires à partir du jeu de données CIFAR10
            - labeled_trainset : contient les 250 images labellisées (choisies aléatoirement)
            - unlabeled_trainset : contient un tuple de datasets contenant les images faiblement augmentées (resp. fortement augmentées )
            - testset : contient les images de test
    """
    raw_trainset = CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
    cifar_mean, cifar_std = calc_images_moments(raw_trainset)

    # Transform pour normalisation des images
    normalize = [transforms.ToTensor(), transforms.Normalize(mean=cifar_mean, std=cifar_std)]
    # Transforms pour data augmentation weak and strong (avec RandAugment)
    weak_augment = [transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect')]
    strong_augment = weak_augment + [transforms.RandAugment(num_ops=2, magnitude=10,
                                                            num_magnitude_bins=11, interpolation=InterpolationMode.BILINEAR)]
    # Sélectionne aléatoirement nb_labels labels pour chaque classe
    label_indices = []
    labels = np.array(raw_trainset.targets)
    for i in range(10):
        idx = np.random.choice(np.where(labels == i)[0], nb_labels, replace=False)
        label_indices.extend(idx)
    label_indices = np.array(label_indices)
    np.random.shuffle(label_indices)
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10) -> pas besoin de liste spécifique

    # Les images labellisées conservées doivent être faiblement augmentées d'après l'article (section 2.2)
    labeled_dataset = CIFAR10(root="./data_cifar", train=True, download=True, transform=transforms.Compose(weak_augment + normalize))
    # On extrait uniquement les indices random selectionnées de ce dataset
    labeled_trainset = Subset(labeled_dataset, label_indices)

    # Les données non labellisées doivent être faiblement augmentées ET fortement augmentées
    # Puis subset sur les indices non inclus dans label_indices pour chacun des 2 datasets, i.e. tous les indices
    strongly_augm_dataset = CIFAR10(root="./data_cifar", train=True, download=True, transform=transforms.Compose(strong_augment + normalize))
    weakly_augm_dataset = CIFAR10(root="./data_cifar", train=True, download=True, transform=transforms.Compose(weak_augment + normalize))
    unlabeled_trainset = UnlabeledDataset(weakly_augm_dataset, strongly_augm_dataset)

    # Les données de test sont juste renormalisées avec les moments calculés sur le jeu d'entraînement
    testset = CIFAR10(root="./data", train=False, download=False, transform=transforms.Compose(normalize))

    return labeled_trainset, unlabeled_trainset, testset


def get_cosine_schedule_with_warmup(optimizer, num_training_steps):
    """ Implémentation spécifique à Fixmatch, différente de l'implémentation proposée dans l'article SGDR
    https://arxiv.org/pdf/1608.03983.pdf"""
    def cos_schedule(current_step):
        no_progress = float(current_step) / float(max(1, num_training_steps))
        return max(0., math.cos(math.pi * 7. / 16. * no_progress))

    return LambdaLR(optimizer, cos_schedule, -1)


class EMA(object):
    """ Modèle utilisé pour l'évaluation
        Ses paramètres sont mis à jour à chaque epoch avec un momentum (Exponential moving average)
    """
    def __init__(self, model, decay=ema_decay):
        self.ema = deepcopy(model)
        self.ema.to(device)
        self.ema.eval()
        self.decay = decay
        self.param_keys = [p for p, _ in self.ema.named_parameters()]
        self.buffer_keys = [b for b, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """ Fonction pour la mise à jour des paramètres (et buffers)"""
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                model_v = msd[k].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)
            # No decay for buffers
            for k in self.buffer_keys:
                esd[k].copy_(msd[k])


def accuracy(output, target, topk=(1,)):
    """Calcule la top k précision"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def test(testloader, model):
    """ Fonction pour calculer la loss et la précision sur le jeu de test à chaque epoch """
    testloader = tqdm(testloader)
    with torch.no_grad():
        losses, top1, top5 = [], [], []
        count = 0
        for batch_id, (X_test, y_test) in enumerate(testloader):
            count += X_test.shape[0]
            model.eval()
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            y_pred = model(X_test)
            loss = F.cross_entropy(y_pred, y_test, reduction='mean')

            prec1, prec5 = accuracy(y_pred, y_test, topk=(1, 5))
            losses.append(loss.item() * X_test.shape[0])
            top1.append(prec1.item() * X_test.shape[0])
            top5.append(prec5.item() * X_test.shape[0])
            testloader.set_description(f"Test Iter: {batch_id+1}/{len(testloader)} - Loss: {np.sum(losses)/count:.4f} - top1: {np.sum(top1)/count:.2f} - top5: {np.sum(top5)/count:.2f}.")

    logging.info(f"top-1 acc: {np.sum(top1)/count:.2f}")
    logging.info(f"top-5 acc: {np.sum(top5)/count:.2f}")
    return np.sum(losses) / count, np.sum(top1) / count


def train(labeled_trainloader, unlabeled_trainloader, testloader, model, optimizer, scheduler, ema):
    """ Boucle d'entraînement"""
    end = time.time()
    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    test_accuracies = []
    best_accuracy = 0
    model.train()
    for e in range(epochs):
        losses, losses_s, losses_u, masks, lrates = [], [], [], [], []
        data_time, batch_time = [], []
        p_bar = tqdm(range(1024))
        for batch_id in range(1024):
            optimizer.zero_grad()
            model.zero_grad()
            try:
                X, y = next(labeled_iter)
            except:
                labeled_iter = iter(labeled_trainloader)
                X, y = next(labeled_iter)

            try:
                (weak_X, _), (strong_X, _) = next(unlabeled_iter)
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                (weak_X, _), (strong_X, _) = next(unlabeled_iter)

            data_time.append(time.time() - end)
            predictions = model(torch.cat((X, weak_X, strong_X)).to(device))
            y = y.to(device)
            y_pred = predictions[:bs]
            weak_pred, strong_pred = predictions[bs:].chunk(2)
            del predictions
            probas = torch.softmax(weak_pred / temperature, dim=-1).to(device)
            max_proba, pseudo_label = torch.max(probas, dim=-1)
            mask = max_proba.ge(threshold).float()

            Ls = F.cross_entropy(y_pred, y, reduction='mean')
            Lu = (F.cross_entropy(strong_pred, pseudo_label, reduction='none') * mask).mean()
            loss = Ls + lambda_u * Lu
            losses.append(loss.item())
            losses_s.append(Ls.item())
            losses_u.append(Lu.item())
            masks.append(mask.mean().item())
            loss.backward()
            optimizer.step()
            scheduler.step()
            ema.update(model)
            batch_time.append(time.time() - end)
            end = time.time()
            lrates.append(scheduler.get_last_lr()[0])
            p_bar.set_description(f"Train Epoch: {e}/{epochs} - Iter: {batch_id+1}/1024 - Data : {np.mean(data_time):.3f} - Batch {np.mean(batch_time):.3f} LR: {lrates[-1]:.4f} -  Loss: {np.mean(losses):.4f} - Loss sup: {np.mean(losses_s):.4f} - Loss unsup: {np.mean(losses_u):.4f} - Mask: {np.mean(masks):.4f}")
            p_bar.update()

        test_loss, test_accuracy = test(testloader, ema.ema)
        writer.add_scalar('train/1.train_loss', np.mean(losses), e)
        writer.add_scalar('train/2.train_loss_s', np.mean(losses_s), e)
        writer.add_scalar('train/3.train_loss_u', np.mean(losses_u), e)
        writer.add_scalar('train/4.mask', np.mean(masks), e)
        writer.add_scalar('test/1.test_acc', test_accuracy, e)
        writer.add_scalar('test/2.test_loss', test_loss, e)
        writer.add_scalars('learning/learning_curves', {'train': np.mean(losses), 'test': test_loss}, e)
        writer.add_scalar('learning/learning_rate', lrates[-1], e)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
        save_checkpoint({
            'epoch': e + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema.ema.state_dict(),
            'acc': test_accuracy,
            'best_acc': best_accuracy,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, test_accuracy > best_accuracy, "./results")
        test_accuracies.append(test_accuracy)
        logging.info(f"Best top-1 acc : {best_accuracy:.2f}")
        logging.info(f"Mean top-1 acc : {np.mean(test_accuracies[-20:]):.2f}")
    writer.close()


def main():
    logging.basicConfig(style='{', format='{asctime} : {message}', datefmt="%c", level=logging.INFO)
    labeled_trainset, unlabeled_trainset, testset = prepare_datasets()
    labeled_trainloader = DataLoader(labeled_trainset, sampler=RandomSampler(labeled_trainset),
                                     batch_size=bs, num_workers=4, drop_last=True, pin_memory=True)
    unlabeled_trainloader = DataLoader(unlabeled_trainset, sampler=RandomSampler(unlabeled_trainset),
                                       batch_size=bs * mu, num_workers=4, drop_last=True, pin_memory=True)
    testloader = DataLoader(testset, sampler=SequentialSampler(testset), num_workers=4, batch_size=bs, pin_memory=True)

    # Wide ResNet-28-2 comme utilisé par kekmodel
    model = WideResNet(depth=28, widen_factor=2, drop_rate=0, num_classes=nb_classes)
    model.to(device)

    # Applique un weight decay pour les couches autres que les biais et batch norm
    decay = [param for name, param in model.named_parameters() if not any(word in name for word in ["bias", "bn"])]
    no_decay = [param for name, param in model.named_parameters() if any(word in name for word in ["bias", "bn"])]
    parameters = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    optim = torch.optim.SGD(parameters, lr=lr, momentum=0.9, nesterov=True)
    # optim = torch.optim.Adam(parameters, lr=0.001)

    # Configuration du scheduler et exponential moving average
    scheduler = get_cosine_schedule_with_warmup(optim, epochs**2)
    # scheduler = ReduceLROnPlateau(optim)
    ema = EMA(model, ema_decay)

    train(labeled_trainloader, unlabeled_trainloader, testloader, model, optim, scheduler, ema)


if __name__ == "__main__":
    main()
