import argparse
import os


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('modeldir')
    parser.add_argument('--outdir', '-o', default='export')
    return parser.parse_args()


def main(args: argparse.Namespace):
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # set filepaths
    ckptdir = args.modeldir
    pipeline_cfgpath = os.path.join(args.modeldir, 'pipeline.config')
    modelname = os.path.basename(args.modeldir)
    exportdir = os.path.join(args.outdir, modelname)

    # export to saved model
    cmd = f'''python -m object_detection.exporter_main_v2 \
            --input_type=image_tensor \
            --pipeline_config_path={pipeline_cfgpath} \
            --trained_checkpoint_dir={ckptdir} \
            --output_directory={exportdir}'''
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    args = get_args()
    main(args)