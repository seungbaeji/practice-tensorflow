'''It loads a savedModel and inference the inputs images.
'''
import os
import argparse
import dataclasses as dc
from typing import NamedTuple

import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# FONT = 'FreeSerif.ttf'  # Ubuntu
FONT = 'AppleGothic.ttf'  # MacOs


class Point(NamedTuple):
    x: int
    y: int


class BoundBox(NamedTuple):
    top: int
    bottom: int
    left: int
    right: int

    @property
    def centre(self) -> Point:
        x = (self.right + self.left) / 2
        y = (self.bottom + self.top) / 2
        return Point(int(x), int(y))


@dc.dataclass
class NumberCandidate:
    score: np.ndarray
    class_id: np.ndarray
    bndbox: BoundBox

    @property
    def number(self) -> int:
        num = int(self.class_id)
        return 0 if num == 10 else num

    @property
    def prob(self) -> float:
        return float(self.score)

    @property
    def loc(self) -> Point:
        return self.bndbox.centre


def get_bndbox(height: int, width: int, bndbox: np.ndarray) -> BoundBox:
    top, left, bottom, right = bndbox
    _top = int(top * height)
    _left = int(left * width)
    _bottom = int(bottom * height)
    _right = int(right * width)
    return BoundBox(_top, _bottom, _left, _right)


def crop_img(img: Image, bndbox: BoundBox) -> Image:
    height = bndbox.bottom - bndbox.top
    width = bndbox.right - bndbox.left
    return img.crop(bndbox.top, bndbox.left, width, height)


def draw_rect_to_img(img: Image, bndbox: BoundBox) -> Image:
    draw = ImageDraw.Draw(img)
    tl, br = (bndbox.left, bndbox.top), (bndbox.right, bndbox.bottom)
    draw.rectangle((tl, br), outline=(255, 0, 0), width=3)
    return img


def draw_text_to_img(img: Image, bndbox: BoundBox, text: str) -> Image:
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 16)  # font-file, font-size
    draw.text((bndbox.right, bndbox.bottom), text, (0, 0, 255), font=font)
    return img


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('modeldir', help='dirpath for a savedmodel')
    parser.add_argument('--outdir', '-o', default='data')
    parser.add_argument('--imgs', '-i', nargs='+')
    parser.add_argument('--thresh_prob', '-t', default=0.3, type=float)
    return parser.parse_args()


def main(args: argparse.Namespace):
    imgs = list(map(lambda x: np.array(Image.open(x).convert('RGB')),
                    args.imgs))

    cmd = f'saved_model_cli show \
            --dir {args.modeldir} \
            --tag_set serve --signature_def serving_default'

    os.system(cmd)

    model = tf.saved_model.load(args.modeldir)
    infer = model.signatures["serving_default"]

    for idx, img in enumerate(imgs):
        height, width, _ = img.shape
        arr = np.expand_dims(img, axis=0).tolist()
        result = infer(input_tensor=tf.constant(arr, dtype=tf.uint8))

        scores = result['detection_scores'][0]
        classes = result['detection_classes'][0]
        boxes = result['detection_boxes'][0]

        numbers = []
        for score, class_id, box in zip(scores, classes, boxes):
            bndbox = get_bndbox(height, width, box)
            num_candidate = NumberCandidate(score, class_id, bndbox)
            if num_candidate.prob > args.thresh_prob:
                numbers.append(num_candidate)

        numbers = sorted(numbers, key=lambda x: x.loc.x)
        result = ''.join([str(number.number) for number in numbers])
        print(f'{idx}th img: {result}')

        _img = Image.fromarray(img)
        for number in numbers:
            draw_rect_to_img(_img, number.bndbox)
            draw_text_to_img(_img, number.bndbox, str(number.number))
        _img.save(os.path.join(args.outdir, f'{idx}.jpeg'))


if __name__ == "__main__":
    args = get_args()
    main(args)
