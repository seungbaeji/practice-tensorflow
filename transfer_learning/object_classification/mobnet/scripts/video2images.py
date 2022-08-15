import os
import argparse

import cv2


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('videopath', help='video filepath')
    parser.add_argument('--outdir',
                        '-o',
                        help='dirpath for images',
                        default='images')
    parser.add_argument('--add',
                        help='add images to the outdir',
                        action='store_true')
    return parser.parse_args()


def main(args: argparse.Namespace):
    cam = cv2.VideoCapture(args.videopath)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # frame
    currentframe = 0
    if args.add:
        getnumfromfname = lambda x: int(x.split('.')[0])
        fnames = os.listdir(args.outdir)
        currentframe = max(map(getnumfromfname, fnames)) if fnames else 0
        print(f'Start from {currentframe}')

    while True:
        # reading from frame
        ret, frame = cam.read()

        if ret:
            # if video is still left continue creating images
            fname = f'{currentframe}.jpg'
            fpath = os.path.join(os.path.abspath(args.outdir), fname)
            print(f'Creating... {fpath}')

            # writing the extracted images
            cv2.imwrite(fpath, frame)
            currentframe += 1
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = get_args()
    main(args)
