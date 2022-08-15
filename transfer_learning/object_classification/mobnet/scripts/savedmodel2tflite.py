import logging
import logging.config
import os
import argparse

import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('modeldir')
    parser.add_argument('--outdir', '-o', default='tflite')
    return parser.parse_args()


def main(args: argparse.Namespace):
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(args.modeldir)
    tflite_model = converter.convert()

    # Save the model.
    modelname = os.path.basename(args.modeldir)
    exportpath = os.path.join(args.outdir, f'{modelname}.tflite')

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    with open(exportpath, 'wb') as f:
        f.write(tflite_model)
    logger.info(f'TFLite model is saved to: {exportpath}')


if __name__ == "__main__":
    args = get_args()
    main(args)