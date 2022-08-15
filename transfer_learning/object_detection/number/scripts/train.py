import os
import argparse
import shutil

import tensorflow as tf
from object_detection.utils import config_util, label_map_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('pretrained_modeldir')
    parser.add_argument('--labelpath', '-l', required=True)
    parser.add_argument('--trainsetpath', '-train', required=True)
    parser.add_argument('--outdir', '-o', default='my_model')
    parser.add_argument('--testsetpath', '-test', default='test.record')
    parser.add_argument('--batch', '-b', default=8, type=int)
    parser.add_argument('--epochs', '-e', default=100, type=int)
    return parser.parse_args()


def main(args: argparse.Namespace):
    label_map = label_map_util.load_labelmap(args.labelpath)
    label_map_dict = label_map_util.get_label_map_dict(label_map)

    pipeline_cfg_fname = 'pipeline.config'
    pipeline_cfgpath = os.path.join(args.outdir, pipeline_cfg_fname)
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # copy pretrained model's pipeline config
    pre_pipeline_cfg = os.path.join(args.pretrained_modeldir,
                                    pipeline_cfg_fname)
    shutil.copy(pre_pipeline_cfg, pipeline_cfgpath)
    pipeline_cfg = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(pipeline_cfgpath, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_cfg)

    # change a training setting
    pipeline_cfg.model.ssd.num_classes = len(label_map_dict.keys())
    pipeline_cfg.train_config.batch_size = args.batch
    pipeline_cfg.train_config.fine_tune_checkpoint = \
        os.path.join(args.pretrained_modeldir, 'checkpoint', 'ckpt-0')
    pipeline_cfg.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_cfg.train_input_reader.label_map_path = args.labelpath
    pipeline_cfg.train_input_reader.tf_record_input_reader.input_path[:] = [
        args.trainsetpath
    ]
    pipeline_cfg.eval_input_reader[0].label_map_path = args.labelpath
    pipeline_cfg.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
        args.testsetpath
    ]

    # update the pipeline config
    config_text = text_format.MessageToString(pipeline_cfg)
    with tf.io.gfile.GFile(pipeline_cfgpath, "wb") as f:
        f.write(config_text)

    # train
    cmd = f'''
    python -m object_detection.model_main_tf2 \
        --model_dir={args.outdir} \
        --pipeline_config_path={pipeline_cfgpath} \
        --num_train_steps={args.epochs}
    '''
    print(cmd)
    os.system(cmd)

    print('Labels:', label_map_dict)
    print(config_util.get_configs_from_pipeline_file(pipeline_cfgpath))


if __name__ == "__main__":
    args = get_args()
    main(args)
