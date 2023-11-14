import numpy as np
import cv2
from config import opt
from pathlib import Path
import argparse
from tqdm import tqdm


def make_dir(dir):
    if not dir.exists():
        dir.mkdir(parents=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phases', default='train, val', help='Choose phases to process, separate by commas')
    parser.add_argument('--padding', default=20, help='Padding size to extract objects')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    phases = args.phases.replace(" ", "").split(',')
    padding = args.padding

    for phase in phases:
        dataset_root = Path(opt.dataset_root)
        imgs_dir = dataset_root / 'images' / phase
        lbls_dir = dataset_root / 'labels' / phase
        out_imgs_dir = dataset_root / opt.extracted_data_dir / phase / 'images'
        out_lbls_f = dataset_root / opt.extracted_data_dir / phase / 'labels.txt'

        make_dir(out_imgs_dir)

        atts = opt.attributes
        bbox_width_threshold = 60

        f = open(out_lbls_f, 'w')
        lbls_filelist = sorted(list(lbls_dir.glob('*')))
        for lbl_f in tqdm(lbls_filelist, desc=f'Phase {phase}'):
            if lbl_f.stat().st_size == 0:
                continue
            frame_bboxes_yolo = np.loadtxt(lbl_f)
            frame_bboxes_yolo = frame_bboxes_yolo.reshape((-1, 5))
            img = cv2.imread(str(imgs_dir/f'{lbl_f.stem}.jpg'))
            for idx, bbox in enumerate(frame_bboxes_yolo):

                # Extract image
                x1, y1 = bbox[[1, 2]] - bbox[[3, 4]] / 2
                x2, y2 = bbox[[1, 2]] + bbox[[3, 4]] / 2
                x1, x2 = x1 * img.shape[1], x2 * img.shape[1]
                y1, y2 = y1 * img.shape[0], y2 * img.shape[0]

                imgcrop = img[max(0, int(y1) - padding):min(int(y2) + padding, img.shape[0]),
                          max(0, int(x1) - padding):min(int(x2) + padding, img.shape[1])]
                fname = f'{lbl_f.stem}_{idx}.png'
                cv2.imwrite(str(out_imgs_dir / fname), imgcrop)

                # Create one-hot label
                cls = bbox[0]
                one_hot_lbl = np.zeros(len(atts), dtype=int)
                one_hot_lbl[int(cls)] = 1
                bbox_lbl = ' '.join(str(d) for d in one_hot_lbl)
                print(f'{fname} {bbox_lbl}', file=f)
        f.close()
