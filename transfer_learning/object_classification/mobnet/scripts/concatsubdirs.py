'''It concatnates categorized subdirs into one image dir with a label csv file.
'''
import logging
import logging.config
import argparse
import os
import shutil
import json

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('rootdir', help='rootdir of subdirs')
    parser.add_argument('--outdir',
                        '-o',
                        help='concat dirpath for images in the subdirs',
                        default='dataset')
    parser.add_argument('--labelfname',
                        help='filename of a label files',
                        default='label')
    parser.add_argument('--imgfnamelength',
                        help='filename length',
                        default=6,
                        type=int)
    return parser.parse_args()


def main(args: argparse.Namespace):
    if not os.path.exists(os.path.join(args.outdir, 'images')):
        os.makedirs(os.path.join(args.outdir, 'images'))

    dirpaths = []
    for dirname in os.listdir(args.rootdir):
        dirpath = os.path.join(args.rootdir, dirname)
        if os.path.isdir(dirpath):
            dirpaths.append(dirpath)

    i = 0
    labels = {}
    for dirpath in dirpaths:
        logger.info(f'Read: {dirpath}')
        label = dirpath.split('/')[-1]
        fnames = os.listdir(dirpath)
        for fname in fnames:
            src = os.path.join(dirpath, fname)
            dst = os.path.join(args.outdir, 'images',
                               f'{str(i).zfill(args.imgfnamelength)}.jpg')

            shutil.copy(src, dst)
            labels.update({i: label})

            i += 1
        logger.info(f'Complete: {dirpath}')

    _labels = set(labels.values())
    label2int = {label: i for i, label in enumerate(sorted(_labels))}
    logger.info(f'Labels: {label2int}')

    records = []
    for i, label in labels.items():
        path = f'{str(i).zfill(args.imgfnamelength)}.jpg'
        record = (path, label2int[label])
        records.append(record)

    df = pd.DataFrame.from_records(records, columns=['path', 'label'])
    df.to_csv(os.path.join(args.outdir, f'{args.labelfname}.csv'))

    with open(os.path.join(args.outdir, f'{args.labelfname}.json'), 'w') as f:
        json.dump(label2int, f)


if __name__ == '__main__':
    args = get_args()
    main(args)
