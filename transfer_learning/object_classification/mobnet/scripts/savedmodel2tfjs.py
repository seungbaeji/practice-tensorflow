import os
import argparse


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('modeldir')
    parser.add_argument('--outdir', '-o', default='tfjs')
    return parser.parse_args()


def main(args: argparse.Namespace):
    modelname = os.path.basename(args.modeldir)
    exportdir = os.path.join(args.outdir, modelname)

    cmd = f'''tensorflowjs_converter --input_format=tf_saved_model \
                --output_node_names='predict' \
                --output_format=tfjs_graph_model \
                --signature_name=serving_default \
                {args.modeldir} \
                {exportdir}'''
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    args = get_args()
    main(args)