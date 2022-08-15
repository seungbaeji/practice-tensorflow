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
    pipeline_cfg_fname = 'pipeline.config'
    pipeline_cfgpath = os.path.join(args.modeldir, pipeline_cfg_fname)
    ckptdirpath = os.path.join(args.modeldir, 'checkpoint')

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # export compatible svaed model with TFLite
    cmd = f'''python -m object_detection.export_tflite_graph_tf2 \
                --pipeline_config_path={pipeline_cfgpath} \
                --trained_checkpoint_dir={ckptdirpath} \
                --output_directory={args.outdir}'''
    print(cmd)
    os.system(cmd)
    logger.info(
        f'Saved Model for TFLite is saved to: {args.outdir}/saved_model')

    # Convert the model
    saved_model_dir = os.path.join(args.outdir, 'saved_model')
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    # Save the model.
    modelname = os.path.basename(args.modeldir)
    exportpath = os.path.join(args.outdir, f'{modelname}.tflite')

    with open(exportpath, 'wb') as f:
        f.write(tflite_model)
    logger.info(f'TFLite model is saved to: {exportpath}')

    # 아래는 레퍼런스에서 사용했던 tflite export 코드. Tensorflow에서 추천하는 방식은 아니나, 문제 발생시 참고바람
    # cmd = f'''tflite_convert --saved_model_dir={saved_model_dir} \
    #         --output_file={exportpath} \
    #         --input_shapes=1,300,300,3 \
    #         --input_arrays=normalized_input_image_tensor \
    #         --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
    #         --inference_type=FLOAT \
    #         --allow_custom_ops'''
    # print(cmd)
    # os.system(cmd)


if __name__ == "__main__":
    args = get_args()
    main(args)