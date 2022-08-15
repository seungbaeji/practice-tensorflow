'''It converts a label json file to a label pbtxt file for a object detection model.
'''
import os
import json
import argparse


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='convert a json file to a pbtxt file')
    parser.add_argument('labeljson', help='label json filepath')
    parser.add_argument('--outdir', '-o', help='file out dirpath', default='.')
    return parser.parse_args()


def main(args: argparse.Namespace):
    with open(args.labeljson, 'r') as f:
        labels = json.load(f)

    fpath = os.path.join(args.outdir, 'label_map.pbtxt')
    with open(fpath, 'w') as f:
        for label in labels:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')

    print(f'Label for object detection is created to `{fpath}`')


if __name__ == "__main__":
    args = get_args()
    main(args)