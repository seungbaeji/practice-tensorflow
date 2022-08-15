'''It trains a model base on the pre-trained mobilenet classification model.
'''
import os
import argparse
import logging
import logging.config

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers.experimental import preprocessing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

MODELPATH = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
WEIGHTPATH = "https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/feature_vector/5"  # input shape (224, 224, 3)
MOBILENET_WIDTH = 244
MOBILENET_HEIGHT = 244

EXPORTDIR = os.path.join('models', 'export')
CHECKPOINTDIR = os.path.join('models', 'ckpt')

NUM_CLASSES = 10

AUTOTUNE = tf.data.AUTOTUNE


def prepare_sample(features):
    # resizing is essential for batch training
    image = tf.image.resize(features["image"],
                            size=(MOBILENET_WIDTH, MOBILENET_HEIGHT))
    return image, features["label"]


def parse_tfrecord_fn(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
    return example


def get_dataaset(filenames, batch_size):
    return tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)\
                  .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)\
                  .map(prepare_sample, num_parallel_calls=AUTOTUNE)\
                  .shuffle(batch_size * 10)\
                  .batch(batch_size)\
                  .prefetch(AUTOTUNE)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('trainset', help='train tfrecord')
    parser.add_argument('--validset', '-v', help='validation tfrecord')
    parser.add_argument('--modelname', '-n', default='my_model')
    parser.add_argument('--batch_size',
                        '-b',
                        help='batch size',
                        type=int,
                        default=256)
    parser.add_argument('--epochs',
                        '-e',
                        help='num of epochs',
                        type=int,
                        default=10)
    parser.add_argument('--augument',
                        '-a',
                        help='data augmentation',
                        action='store_true')
    parser.add_argument('--ckptdir', '-c', help='dirpath for checkpoint')
    return parser.parse_args()


def main(args: argparse.Namespace):
    train_filenames = tf.io.gfile.glob(args.trainset)
    valid_filenames = tf.io.gfile.glob(args.validset)

    logger.info(f'Build a Model')

    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=[None, None, 3],
                              name='input',
                              dtype='int32'),
        preprocessing.Rescaling(1. / 255),
        tf.keras.layers.Resizing(MOBILENET_WIDTH,
                                 MOBILENET_HEIGHT,
                                 interpolation='bilinear',
                                 crop_to_aspect_ratio=False),
        hub.KerasLayer(WEIGHTPATH,
                       trainable=True,
                       arguments=dict(batch_norm_momentum=0.997)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name='predict')
    ])

    m.build([None, None, None, 3])  # Batch input shape.
    m.summary()

    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'],
    )
    if args.ckptdir:
        m.load_weights(os.path.join(args.ckptdir, 'cp.ckpt'))
        logger.info(f'Load weights from {args.ckptdir} dir')

    logger.info(f'Train a Model')
    checkpoint_path = os.path.join(CHECKPOINTDIR, args.modelname, 'cp.ckpt')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     monitor='val_acc',
                                                     mode="auto",
                                                     verbose=1)

    m.fit(x=get_dataaset(train_filenames, args.batch_size),
          epochs=args.epochs,
          verbose=1,
          callbacks=[cp_callback],
          validation_data=get_dataaset(valid_filenames, args.batch_size))

    modelpath = os.path.join(EXPORTDIR, args.modelname)
    tf.saved_model.save(m, modelpath)

    cmd = f'''saved_model_cli show \
            --dir {modelpath} \
            --tag_set serve --signature_def serving_default'''
    os.system(cmd)

    logger.info(f'Model is exported to {modelpath}')


if __name__ == '__main__':
    args = get_args()
    main(args)