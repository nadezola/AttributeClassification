from pathlib import Path
from lib.vis_lib import vis_labels
from lib.reformat_labels import encode_labels


if __name__ == '__main__':
    task = 'test'
    root = Path(f'/media/nadesha/hdd/02_INTERACT-DATASET/ludwig_mobility_dataset/extracted/{task}')
    label_file = root / 'labels.txt'
    img_dir = root / 'images'
    output_vis_dir = root / 'vis'
    if not output_vis_dir.exists():
        output_vis_dir.mkdir()

    fnames, labels = encode_labels(label_file)
    vis_labels(fnames, labels, img_dir, output_vis_dir)
