import torch
from tqdm import tqdm
import numpy as np
import copy
from lib.vis_lib import plot_loss
from config import opt


def save_state(state):
    torch.save(state, f'outputs/exps/exp_{opt.expID}/last_state.pth')


def save_best_model(best_model):
    torch.save(best_model, f'outputs/exps/exp_{opt.expID}/best.pth')


def save_best_weights(weights):
    torch.save(weights, f'outputs/exps/exp_{opt.expID}/best_weights.pth')


def training(model, train_dataloader, val_dataloader, optimizer, loss_function, scheduler, state):
    train_loss_array = []
    train_acc_array = []
    val_loss_array = []
    val_acc_array = []
    lowest_val_loss = np.inf
    best_val_acc = 0
    best_model = None
    start_epoch = 0

    if state is not None:
        train_loss_array = state['train_loss_array']
        train_acc_array = state['train_acc_array']
        val_loss_array = state['val_loss_array']
        val_acc_array = state['val_acc_array']
        lowest_val_loss = state['lowest_val_loss']
        best_val_acc = np.max(val_acc_array)
        start_epoch = state['epoch']

    for epoch in range(start_epoch, opt.num_epochs):
        lr = scheduler.get_last_lr()
        print(f'Epoch: {epoch + 1} / {opt.num_epochs} | Learning rate: {lr}')
        for phase in ['train', 'val']:
            epoch_loss = 0
            epoch_correct_items = 0
            epoch_gts_num = 0
            epoch_items = 0

            if phase == 'train':
                model.train()
                with torch.enable_grad():
                    for samples, targets in tqdm(train_dataloader):
                        samples = samples.to(opt.DEVICE)
                        targets = targets.to(opt.DEVICE)

                        optimizer.zero_grad()
                        outputs = model(samples)
                        loss = loss_function(outputs, targets)

                        preds = outputs.argmax(dim=1)
                        correct_items = (preds == targets.argmax(dim=1)).float().sum()
                        epoch_correct_items += correct_items.item()
                        epoch_gts_num += len(targets)

                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                        epoch_items += len(targets)

                train_loss_array.append(epoch_loss / epoch_items)
                train_acc_array.append(epoch_correct_items / epoch_gts_num)
                print(f'    | Train Loss: {train_loss_array[-1]} | Train Acc: {train_acc_array[-1]}')
                scheduler.step()

            elif phase == 'val':
                model.eval()
                with torch.no_grad():
                    for samples, targets in tqdm(val_dataloader):
                        samples = samples.to(opt.DEVICE)
                        targets = targets.to(opt.DEVICE)

                        outputs = model(samples)
                        loss = loss_function(outputs, targets)

                        preds = outputs.argmax(dim=1)
                        correct_items = (preds == targets.argmax(dim=1)).float().sum()
                        epoch_correct_items += correct_items.item()
                        epoch_gts_num += len(targets)

                        epoch_loss += loss.item()
                        epoch_items += len(targets)

                val_loss_array.append(epoch_loss / epoch_items)
                val_acc_array.append(epoch_correct_items / epoch_gts_num)
                print(f'    | Val.Loss: {val_loss_array[-1]} | Val.Acc: {val_acc_array[-1]}')

                if val_loss_array[-1] < lowest_val_loss:
                    lowest_val_loss = val_loss_array[-1]
                    print("\t| New lowest val loss: {}".format(lowest_val_loss))
                if val_acc_array[-1] > best_val_acc:
                    best_val_acc = val_acc_array[-1]
                    print("\t| New best val acc: {}".format(best_val_acc))
                    best_model = copy.deepcopy(model)
                    save_best_model(best_model)
                    #save_best_weights(model.state_dict())

                save_state({'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss_array': train_loss_array,
                            'train_acc_array': train_acc_array,
                            'val_loss_array': val_loss_array,
                            'val_acc_array':val_acc_array,
                            'lowest_val_loss': lowest_val_loss})

        plot_loss(train_loss_array, val_loss_array, epoch + 1)

    return best_model, train_loss_array, train_acc_array, val_loss_array, val_acc_array


