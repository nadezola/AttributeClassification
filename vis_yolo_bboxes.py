import numpy as np
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
from config import opt
import viren2d


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phases', default='val', help='Choose phases to process, separate by commas')
    parser.add_argument('--out', default='outputs/vis', help='Folder to save results')

    args = parser.parse_args()
    return args


def get_bbox_style(color='navy-blue'):
    line_style = viren2d.LineStyle(
            width=1,
            color=color,
            dash_pattern=[5, 10],
            dash_offset=0.0,
            cap='round',
            join='miter'
    )
    text_style = viren2d.TextStyle(
            family='monospace',
            size=12,
            color=color,
            #bold=True,
            italic=True,
            #line_spacing=1,
            halign='center',
            valign='top'
    )
    box_style = viren2d.BoundingBox2DStyle(
        line_style=line_style,
        text_style=text_style,
        box_fill_color='same!10',
        text_fill_color='white!20',
        clip_label=True)

    return box_style


if __name__ == '__main__':
    args = parse_args()
    phases = args.phases.replace(" ", "").split(',')
    atts = opt.attributes
    dataset_root = Path(opt.dataset_root)

    for phase in phases:
        imgs_dir = dataset_root / 'images' / phase
        lbls_dir = dataset_root / 'labels' / phase
        out_imgs_dir = Path(args.out) / phase / 'images'

        if not out_imgs_dir.exists():
            out_imgs_dir.mkdir(parents=True)

        lbls_filelist = sorted(list(lbls_dir.glob('*')))
        for lbl_f in tqdm(lbls_filelist, desc=f'Phase {phase}'):

            # Read image
            img = cv2.imread(str(imgs_dir / f'{lbl_f.stem}.jpg'))
            fname = f'{lbl_f.stem}.jpg'
            if lbl_f.stat().st_size == 0:
                cv2.imwrite(str(out_imgs_dir / fname), img)
                continue

            # Read bboxes
            frame_bboxes_yolo = np.loadtxt(lbl_f)
            frame_bboxes_yolo = frame_bboxes_yolo.reshape((-1, 5))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            painter = viren2d.Painter(img_rgb)

            # Visualize bboxes
            for bbox in frame_bboxes_yolo:
                class_id = int(bbox[0])
                class_name = atts[class_id]
                x1, y1 = bbox[[1, 2]] - bbox[[3, 4]] / 2
                x2, y2 = bbox[[1, 2]] + bbox[[3, 4]] / 2
                x1, x2 = x1 * img.shape[1], x2 * img.shape[1]
                y1, y2 = y1 * img.shape[0], y2 * img.shape[0]

                color = viren2d.color_from_object_category(class_name)
                box_style = get_bbox_style(color)
                rect = viren2d.Rect.from_lrtb(bbox[1], bbox[2], bbox[3], bbox[4], radius=0.2)
                painter.draw_bounding_box_2d(rect, box_style=box_style, label_bottom=[class_name])
                imgvis = cv2.cvtColor(np.array(painter.canvas), cv2.COLOR_BGR2RGB)

                cv2.imwrite(str(out_imgs_dir / fname), imgvis)



