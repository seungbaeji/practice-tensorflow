'''It loads a savedModel and inference the inputs images.
'''
import os
import argparse

import tensorflow as tf
from PIL import Image
import numpy as np


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('modeldir', help='dirpath for a savedmodel')
    parser.add_argument('--imgs', '-i', nargs='+')
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

    for img in imgs:
        arr = np.expand_dims(img, axis=0).tolist()
        probs = infer(input=tf.constant(arr, dtype=tf.int32))
        label = tf.math.argmax(probs['predict'], 1)
        print(label)


if __name__ == "__main__":
    args = get_args()
    main(args)
