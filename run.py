"""
Runs model to predict Human Attributes
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from lib import data_processing
from lib import vis_lib
import config.opt as opt
import numpy as np
import argparse


def makedir(dir):
    if not dir.exists():
        dir.mkdir()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet50_V1/best.pth',
                        help='Model to use for predictions')
    parser.add_argument('--data', default='mobility_dataset/extracted/test/images',
                        help='Path to data')
    parser.add_argument('--out', default='outputs/resnet50_V1', help='Path where to save prediction results')
    parser.add_argument('--vis', action='store_true', help='Activate prediction visualization')


    args = parser.parse_args()
    return args


def main(test_dir, model, output_dir, vis_flag):

    res_pred_file = output_dir / 'predictions.txt'
    res_scores_file = output_dir / 'prediction_scores.txt'
    res_labels_file = output_dir / 'prediction_one_hot_vectors.txt'

    test_dataset = data_processing.test_dataset(test_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    model = torch.load(model)
    model.eval()

    fnames = []
    preds = []
    preds_classes = []
    preds_scores = []
    with torch.no_grad():
        for samples, batch_fnames in tqdm(test_dataloader, desc='Prediction:'):
            samples = samples.to(opt.DEVICE)
            batch_outputs = model(samples)

            batch_outputs_np = batch_outputs.to('cpu').numpy()
            batch_preds = np.zeros_like(batch_outputs_np)
            batch_preds[np.arange(len(batch_outputs_np)), batch_outputs_np.argmax(1)] = 1
            batch_preds_cls = list(opt.attributes[i] for i in batch_outputs_np.argmax(axis=1))

            fnames.extend(list(batch_fnames))
            preds_classes.extend(batch_preds_cls)
            preds_scores.extend(batch_outputs_np)
            preds.extend(batch_preds.astype(int))

    # Write word preds
    with open(res_pred_file, 'w') as f:
        for i in range(len(fnames)):
            print(f"{fnames[i]} {preds_classes[i]}", file=f)

    if vis_flag:
        vis_image_dir = output_dir / 'vis_preds'
        makedir(vis_image_dir)
        vis_lib.vis_labels(fnames, preds_classes, test_dir, vis_image_dir)

    # Write score preds
    with open(res_scores_file, 'w') as f:
        for i in range(len(fnames)):
            s = fnames[i]
            for j in preds_scores[i]:
                s += ' ' + str(j)
            print(s, file=f)
            #print(f"{fnames[i]} {preds_scores[i]}", file=f)


    # Write label preds
    with open(res_labels_file, 'w') as f:
        for i in range(len(fnames)):
            s = fnames[i]
            for j in preds[i]:
                s += ' ' + str(j)
            print(s, file=f)


if __name__ == '__main__':
    args = parse_args()
    test_images =  Path(args.data)
    model = args.model
    output_dir = Path(args.out)
    vis_flag = args.vis

    makedir(output_dir)

    if torch.cuda.is_available():
        print('CUDA is available. Working on GPU')
        opt.DEVICE = torch.device('cuda')
    else:
        print('CUDA is not available. Working on CPU')
        opt.DEVICE = torch.device('cpu')

    main(test_images, model, output_dir, vis_flag)

