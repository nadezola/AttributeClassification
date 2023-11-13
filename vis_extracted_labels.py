from pathlib import Path
from lib.vis_lib import vis_labels
from config import opt
import pandas as pd


def decode_labels(label_file, attributes):
    """ from [0 1 0 0 ..] to 'wheel' """
    labels = pd.read_csv(label_file, delimiter=' ', index_col=0, header=None)
    labels_encoded = []
    fnames = []
    for f, row in labels.iterrows():
        fnames.append(f)
        labels_encoded.append(attributes[row.idxmax() - 1])
    return fnames, labels_encoded


if __name__ == '__main__':
    phase = 'val'
    root = Path(opt.dataset_root) / opt.extracted_data_root / phase
    label_file = root / 'labels.txt'
    img_dir = root / 'images'
    output_vis_dir = root / 'vis'

    if not output_vis_dir.exists():
        output_vis_dir.mkdir()

    fnames, labels = decode_labels(label_file, opt.attributes)
    vis_labels(fnames, labels, img_dir, output_vis_dir)
