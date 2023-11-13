"""Visualizing random items from the datasets"""
import glob
import os.path

import matplotlib.pyplot as plt
import numpy as np
import config.opt as opt
import cv2
import pandas as pd


def plot_train_data(dataset, phase):
    fig, axs = plt.subplots(3, 6, figsize=(20, 11))
    fig.suptitle(f'Random pictures from {phase} dataset', fontsize=20)
    for ax in axs.flatten():
        n = np.random.randint(len(dataset))
        img = dataset[n][0]
        img = img.numpy().transpose((1, 2, 0))
        mean = np.array(opt.mean)
        std = np.array(opt.std)
        img = std * img + mean
        img = np.clip(img, 0, 1)
        ax.set_title(dataset[n][1])
        ax.imshow(img)
    plt.show()
    fig.savefig(f'exps/exp_{opt.expID}/random_{phase}_data.jpg')


def plot_eval_data(img_dir, label_file):
    fig, axs = plt.subplots(3, 6, figsize=(20, 11))
    fig.suptitle('Random pictures from eval dataset', fontsize=20)
    file_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    labels = pd.read_csv(label_file, sep=' ', header=None, index_col=0)
    n = 203
    for k, ax in enumerate(axs.flatten()):
        img = cv2.imread(file_list[n])
        lbls = labels.loc[os.path.split(file_list[n])[-1]]
        text = f'img#{k+1}: '
        for i in range(lbls.size):
            text += f' {lbls.iloc[i]}'
        ax.set_title(text)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        n += 10
    plt.show()
    fig.savefig(f'exps/exp_{opt.expID}/random_eval_data.jpg')


def plot_train_results_data(training_results, num_epochs, batch_size):
    model, train_loss_array, train_acc_array, val_loss_array, val_acc_array = training_results

    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Model training | Batch size: {}".format(batch_size), fontsize=16)
    axs[0].plot(list(range(1, num_epochs + 1)), train_loss_array, label="train_loss")
    axs[0].plot(list(range(1, num_epochs + 1)), val_loss_array, label="val_loss")
    axs[0].legend(loc='best')
    axs[0].set(xlabel='epochs', ylabel='loss')
    axs[1].plot(list(range(1, num_epochs + 1)), train_acc_array, label="train_acc")
    axs[1].plot(list(range(1, num_epochs + 1)), val_acc_array, label="val_acc")
    axs[1].legend(loc='best')
    axs[1].set(xlabel='epochs', ylabel='accuracy')
    #plt.show()
    fig.savefig(f'exps/exp_{opt.expID}/train_results.jpg')


def plot_loss(train_loss_array, val_loss_array, epoch):
    eps = np.arange(epoch) + 1

    plt.plot(eps, train_loss_array, 'bo', label='Training loss')
    plt.plot(eps, val_loss_array, 'r', label='Test loss')
    plt.title('Training (blue) and Val (red) loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid('off')
    plt.savefig(f'exps/exp_{opt.expID}/loss.jpg')


def vis_labels(fnames, labels, image_dir, vis_image_dir, color=(0, 0, 255), org=(5, 10)):
    '''
        :param fnames: list of fnames in right order
        :param labels: list of classes in the same order
    '''

    for idx in range(len(fnames)):
        file = os.path.join(image_dir, fnames[idx])
        image = cv2.imread(file)
        lbls = labels[idx]
        step = 0
        for l in lbls:
            image = cv2.putText(image, l, org=(5, 10 + step), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.3, color=color, thickness=1)
            step += 10
        cv2.imwrite(os.path.join(vis_image_dir, fnames[idx]), image)

def plot_confusion(
        confusion: np.ndarray,
        normalize=True,
        names: [] = None,
        cmap: str = 'Blues',
        title: str = None,
) -> plt.Figure:
    import seaborn as sn

    num_classes = confusion.shape[0]
    # if names is None and num_classes == 5:
    #     names = ['pedestrian', 'wheelchair', 'rollator', 'crutch', 'cane']
    # else:
    #     names = [str(x) for x in range(num_classes)]

    array = confusion / ((confusion.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)  # normalize columns

    array[array < 0.0005] = np.nan  # don't annotate (would appear as 0.00)

    fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
    sn.set(font_scale=1.0 if num_classes < 50 else 0.8)  # for label size
    labels = (0 < len(names) < 99) and (len(names) == num_classes)  # apply names to ticklabels
    if labels:
        ticklabels = names
    else:
        ticklabels = "auto"

    annotations = []
    for row_idx in range(num_classes):
        if normalize:
            annotations.append([f'{array[row_idx, col_idx]:.1%}\n{int(confusion[row_idx, col_idx]):d}' for
                                col_idx in range(num_classes)])
        else:
            annotations.append([f'{int(confusion[row_idx, col_idx]):d}' for col_idx in range(num_classes)])
    annotations = np.array(annotations)

    sn.heatmap(array,
               ax=ax,
               # annot=nc < 30,
               annot=annotations if num_classes < 20 else False,
               annot_kws={
                   "size": 10},
               cmap=cmap,
               # fmt='.2f',
               fmt='',
               square=True,
               vmin=0.0,
               xticklabels=ticklabels,
               yticklabels=ticklabels).set_facecolor((1, 1, 1))
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title(title if title is not None else 'Confusion Matrix')

    return fig

def plot_confusion2(
        confusion: np.ndarray,
        normalize=True,
        names: [] = None,
        cmap: str = 'Blues',
        title: str = None,
) -> plt.Figure:
    import seaborn as sn

    num_classes = confusion.shape[0]
    if names is None and num_classes == 5:
        names = ['pedestrian', 'wheelchair', 'rollator', 'crutch', 'cane']
    else:
        names = [str(x) for x in range(num_classes)]

    array = confusion / ((confusion.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)  # normalize columns

    array[array < 0.0005] = np.nan  # don't annotate (would appear as 0.00)

    fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
    sn.set(font_scale=1.0 if num_classes < 50 else 0.8)  # for label size
    labels = (0 < len(names) < 99) and (len(names) == num_classes)  # apply names to ticklabels
    if labels:
        ticklabels = names
    else:
        ticklabels = "auto"

    annotations = []
    for row_idx in range(num_classes):
        if normalize:
            annotations.append([f'{array[row_idx, col_idx]:.1%}\n{int(confusion[row_idx, col_idx]):d}' for
                                col_idx in range(num_classes)])
        else:
            annotations.append([f'{int(confusion[row_idx, col_idx]):d}' for col_idx in range(num_classes)])
    annotations = np.array(annotations)

    sn.heatmap(array,
               ax=ax,
               # annot=nc < 30,
               annot=annotations if num_classes < 30 else False,
               annot_kws={
                   "size": 16},
               cmap=cmap,
               # fmt='.2f',
               fmt='',
               square=True,
               vmin=0.0,
               xticklabels=ticklabels,
               yticklabels=ticklabels).set_facecolor((1, 1, 1))
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title(title if title is not None else 'Confusion Matrix')

    return fig