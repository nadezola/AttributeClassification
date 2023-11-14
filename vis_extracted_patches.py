"""
Draws Human Attributes labels on extracted patches
"""

from pathlib import Path
from lib.vis_lib import vis_labels
from config import opt
from lib.decode_labels import decode_labels


if __name__ == '__main__':
    phase = 'val'
    root = Path(opt.dataset_root) / opt.extracted_data_dir / phase
    label_file = root / 'labels.txt'
    img_dir = root / 'images'
    output_vis_dir = root / 'vis_objects'

    if not output_vis_dir.exists():
        output_vis_dir.mkdir()

    fnames, labels = decode_labels(label_file, opt.attributes)
    vis_labels(fnames, labels, img_dir, output_vis_dir)
