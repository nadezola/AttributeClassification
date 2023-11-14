"""
Trains model to classify Human Attributes
"""

from pathlib import Path

import numpy as np
import torch
import torchsummary
from torch.utils.data import DataLoader
from torchvision import models

from lib import data_processing, vis_lib
from lib.training import training
from lib.models import get_model
from lib.loss import Loss
import config.opt as opt


def load_last_state(expID, model, optimizer):
    state = torch.load(f'outputs/exps/exp_{expID}/last_state.pth')
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    return model, optimizer, state


def main():
    model = get_model()
    model = model.to(opt.DEVICE)
    print(model)
    model_info = str(torchsummary.summary(model, torch.zeros(1, 3, opt.resize[0], opt.resize[1])))

    train_dataset, val_dataset = data_processing.trainval_dataset()
    if opt.vis_train_samples:
        vis_lib.plot_train_data(train_dataset, 'train')
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)


    num_epochs = opt.num_epochs
    state = None

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    if opt.resume > 0:
        model, optimizer, state = load_last_state(opt.resume, model, optimizer)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=0.1)

    training_results = training(model=model,
                                train_dataloader=train_dataloader,
                                val_dataloader=val_dataloader,
                                optimizer=optimizer,
                                loss_function=loss_function,
                                scheduler=scheduler,
                                state=state)
    vis_lib.plot_train_results_data(training_results, num_epochs, batch_size=opt.batch_size)

    model, train_loss_array, train_acc_array, val_loss_array, val_acc_array = training_results
    min_loss = min(val_loss_array)
    min_loss_epoch = val_loss_array.index(min_loss)
    min_loss_accuracy = val_acc_array[min_loss_epoch]

    with open(f'outputs/exps/exp_{opt.expID}/training_results.txt', 'w') as f:
        print("Training parameters:", file=f)
        print(f"\tTrainVal dataset: {opt.dataset_root}", file=f)
        print(f"\tAttributes: {opt.attributes}", file=f)
        print(f"\tBackbone: {opt.model}", file=f)
        print(f"\tEpochs: {opt.num_epochs}", file=f)
        print(f"\tFreeze: {opt.freeze}", file=f)
        print(f"\tWeights: {opt.weights}", file=f)
        print(f"\tBatch size: {opt.batch_size} | Learning rate: {opt.lr} | Step size: {opt.step_size} | Resize: {opt.resize}", file=f)

        print("\nModel info (raw):", file=f)
        print(model, file=f)
        print("\nModel info:", file=f)
        print(model_info, file=f)

        print("\nTraining results:", file=f)
        print(f"\tMin val loss {min_loss:.4f} was achieved during epoch #{min_loss_epoch + 1}", file=f)
        print(f"\tVal accuracy during min val loss is {min_loss_accuracy:.4f}", file=f)
        print(f"\nTrain acc:\n{train_acc_array}", file=f)
        print(f"\nVal acc:\n{val_acc_array}", file=f)

        print('\nMain Results:', file=f)
        print('\tAvgValAcc\tLastEAcc\tMaxValAcc/NumE', file=f)
        print(f'\t{np.average(val_acc_array):.4f}\t{val_acc_array[-1]:.4f}\t{np.max(val_acc_array):.4f} / {np.argmax(val_acc_array) + 1}', file=f)


if __name__ == '__main__':
    if torch.cuda.is_available():
        print('CUDA is available. Working on GPU')
        opt.DEVICE = torch.device('cuda')
    else:
        print('CUDA is not available. Working on CPU')
        opt.DEVICE = torch.device('cpu')

    results_dir = Path('outputs') / 'exps'
    filelist = list(results_dir.glob('exp_*'))
    if len(filelist) > 0:
        filelist = sorted([int(x.stem.split('_')[-1]) for x in filelist])
        opt.expID = filelist[-1] + 1
    else:
        opt.expID = 1
    (results_dir / f'exp_{opt.expID}').mkdir(parents=True)

    main()
