"""
Evaluates model performance
"""

import numpy as np
import pandas as pd
#import sklearn.metrics as metrics
from pathlib import Path
from lib.vis_lib import plot_confusion
from config import opt
from lib.decode_labels import decode_labels
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ResNet50(V1)',
                        help='Model name to display as a title of Confusion matrix')
    parser.add_argument('--gt', default='mobility_dataset/extracted/test/labels.txt',
                        help='Path to the file with labels of patches')
    parser.add_argument('--pred', default='outputs/resnet50_V1/predictions.txt',
                        help='Path to classifier predicitons')
    parser.add_argument('--out', default='outputs', help='Path where to save evaluation results')


    args = parser.parse_args()
    return args


if __name__ == '__main__':

    dataset_root = opt.dataset_root
    attributes = opt.attributes

    args = parse_args()
    model_name = args.model
    gt_file = args.gt
    pred_file = args.pred
    output_path = Path(args.out)
    res_eval_file = output_path / f'{model_name}_evaluation.csv'
    fig_confusion_file = output_path / f'{model_name}_confusion.jpg'

    if not output_path.exists():
        output_path.mkdir()

    # Download GTs & Preds
    fnames, gts = decode_labels(gt_file, attributes)
    pred_df = pd.read_csv(pred_file, delimiter=' ', header=None, index_col=0)
    if len(gts) != pred_df.index.size:
        print('WARNING: The number of GTs is not equal to the number of Predictions!')

    # TPs and total Acc
    preds = []
    for i, f in enumerate(fnames):
        if f in pred_df.index:
            preds.append(int((pred_df.loc[f].values == gts[i])[0]))
        else:
            print(f'WARNING: ID {i} from GTs is not in predictions')
    total_acc = sum(preds) / len(preds)
    print(f"Total Acc: {total_acc}")

    # Acc by class
    acc_by_cls = {}
    for a in attributes:
        acc_by_cls[a] = [0, 0]
    for i in range(len(gts)):
        for key, value in acc_by_cls.items():
            if gts[i] == key:
                acc_by_cls[key][1] = value[1] + 1
                acc_by_cls[key][0] = value[0] + preds[i]
    print(acc_by_cls)
    for key, value in acc_by_cls.items():
        acc = value[0] / value[1] if value[1] != 0 else 0
        print(f'{key} acc: {acc:.4f}')

    # Confustion Matrix
    confusion = np.zeros((len(attributes), len(attributes)))
    for f, row in pred_df.iterrows():
        gt = gts[fnames.index(f)]
        pred = pred_df.loc[f].values
        confusion[attributes.index(gt), attributes.index(pred)] += 1

    fig_confusion = plot_confusion(confusion.T, names=opt.attributes, title=model_name)
    fig_confusion.savefig(fig_confusion_file)

    with open(res_eval_file, 'w') as f:
        print(f"\tTPs\tGTs\tAcc\n", file=f)
        for key, value in acc_by_cls.items():
            acc = value[0] / value[1] if value[1] != 0 else 0
            print(f"{key}\t{value[0]}\t{value[1]}\t{acc:.4f}", file=f)
        print(f"\nTotal:\t{sum(preds)}\t{len(preds)}\t{total_acc:.4f}", file=f)


