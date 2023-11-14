import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from pathlib import Path
from lib.reformat_labels import decode_labels
from lib.vis_lib import plot_confusion
from config import opt
from lib.decode_labels import decode_labels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ResNet50 (V2)',
                        help='Model name to display as a title of Confusion matrix')
    parser.add_argument('--gt', default='/media/nadesha/hdd/INTERACT-DATASET/dataset_ludwig_gpv_icg_regular_6class/extracted_nov23/val',
                        help='Path to the file with labels')
    parser.add_argument('--pred', default='',
                        help='Path to classifier predicitons')
    parser.add_argument('--out', default='outputs', help='Path where to save evaluation results')


    args = parser.parse_args()
    return args


if __name__ == '__main__':

    dataset_root = opt.dataset_root
    attributes = opt.attributes

    args = parse_args()
    plot_titel = args.model
    gt_file = args.gt
    pred_file = args.pred
    res_eval_file = Path(args.out) / 'eval_results.txt'
    fig_confusion_file = Path(args.out) / 'confusion.jpg'

    # Download GTs & Preds
    fnames, gts = decode_labels(gt_file, attributes)
    pred_df = pd.read_csv(pred_file, delimiter=' ', header=None, index_col=0)
    if len(gts) != pred_df.index.size:
        print('WARNING: The number of GTs is not equal to the number of Predictions!')

    # TPs and total Acc
    preds = []
    for i, f in enumerate(fnames):
        if f in preds_df.index:
            preds.append(int(preds_df.loc[f].values == gts[i]))
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
    for f, row in preds_df.iterrows():
        gt = gts[fnames.index(f)]
        pred = preds_df.loc[f].values
        confusion[attributes.index(gt), attributes.index(pred)] += 1

    fig_confusion = plot_confusion(confusion[1:, 1:].T, title=plot_titel)
    fig_confusion.savefig(fig_confusion_file)

    with open(res_eval_file, 'w') as f:
        print(f"\t\tTPs\tGTs\tAcc\n", file=f)
        for key, value in acc_by_cls.items():
            if key == 'person':
                continue
            acc = value[0] / value[1] if value[1] != 0 else 0
            if key == 'cane' or key == 'crutch':
                print(f"{key}\t\t{value[0]}\t{value[1]}\t{acc:.4f}", file=f)
            else:
                print(f"{key}\t{value[0]}\t{value[1]}\t{acc:.4f}", file=f)
        print(f"\nTotal:\t\t{sum(preds)}\t{len(preds)}\t{total_acc:.4f}", file=f)


