'''It encodes dataset into a TFRecord file.

refer to: https://keras.io/examples/keras_recipes/creating_tfrecords/
'''
import logging
import logging.config
import argparse
import os

import tensorflow as tf
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def _image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[tf.io.encode_jpeg(value).numpy()]))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[value.encode()]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example(image, path, label):
    feature = {
        "image": _image_feature(image),
        "path": _bytes_feature(path),
        "label": _int64_feature(label),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--outdir',
                        '-o',
                        help='dirpath for saving tfrecords',
                        required=True)
    parser.add_argument('--imgdir',
                        '-i',
                        help='image dirpath of the image in the csvfile',
                        required=True)
    parser.add_argument('--csvfile',
                        '-l',
                        help='csvfile for labels with a file name',
                        type=pd.read_csv,
                        required=True)
    parser.add_argument('--fname', '-n', default='mydata')
    parser.add_argument('--valid',
                        '-v',
                        help='flag for spliting valid dataset',
                        action='store_true')
    parser.add_argument('--test',
                        '-t',
                        help='flag for spliting test dataset',
                        action='store_true')
    return parser.parse_args()


def main(args: argparse.Namespace):
    outdir: str = args.outdir
    imgdir: str = args.imgdir
    csvfile: pd.DataFrame = args.csvfile
    fname: str = args.fname

    data = {'train': csvfile}
    if args.valid:
        data['valid'] = csvfile.sample(frac=0.3, random_state=0)
        data['train'] = csvfile.drop(data['valid'].index)
        logger.info(f'Split dataset into {list(data.keys())}')

    if args.test:
        splitfrom = 'valid' if 'valid' in data else 'train'
        data['test'] = data[splitfrom].sample(frac=0.2, random_state=0)
        data[splitfrom] = data[splitfrom].drop(data['test'].index)
        logger.info(f'Split dataset into {list(data.keys())}')

    # meta data for describing splitted dataset
    for key, df in data.items():
        df['purpose'] = key
    df = pd.concat(data.values())
    metafpath = os.path.join(args.outdir, f'{fname}.meta.csv')
    df.to_csv(metafpath, index=False)
    logger.info(f'Meta data is saved to :{metafpath}')

    # generates tfrecords
    for dtype, csvdata in data.items():
        tfrecordpath = os.path.join(outdir, f'{fname}.{dtype}.tfrecord')
        with tf.io.TFRecordWriter(tfrecordpath) as writer:
            for row in csvdata.itertuples(index=False):
                image_path = os.path.join(imgdir, row.path)
                image = tf.io.decode_jpeg(tf.io.read_file(image_path))

                example = create_example(image, row.path, row.label)
                writer.write(example.SerializeToString())
        logger.info(f'{dtype} tfrecord file is done')


if __name__ == '__main__':
    args = get_args()
    main(args)