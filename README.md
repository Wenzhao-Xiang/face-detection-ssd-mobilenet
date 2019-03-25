# Face-Detection-SSD-MobileNet

## Prerequisites

### Install TensorFlow Object Detection API  

https://github.com/tensorflow/models/tree/master/research/object_detection

**Remember to export the library in PYTHONPATH in your environment**.

## Preprocess the dataset

Please run the following scripts:

```shell
python 1_download_data.py

python3 2_data_to_pascal_xml.py

python 3_xml_to_csv.py

python 4_generate_tfrecord.py --images_path=data/tf_wider_train/images --csv_input=data/tf_wider_train/train.csv  --output_path=data/train.record

python 4_generate_tfrecord.py --images_path=data/tf_wider_val/images --csv_input=data/tf_wider_val/val.csv  --output_path=data/val.record

```

## Modify the config file

Read the comments and modify the config information in ssd_mobilenet_v1_face.config/ssd_mobilenet_v2_face.config/ssdlite_mobilenet_v2_face.config

Maybe [this blog](https://becominghuman.ai/tensorflow-object-detection-api-tutorial-training-and-evaluating-custom-object-detector-ed2594afcf73) can help you.

## Train

Just run:

```shell
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
```

where ${PIPELINE_CONFIG_PATH} points to the pipeline config and ${MODEL_DIR} points to the directory in which training checkpoints and events will be written to. Note that this binary will interleave both training and evaluation.

## Export Model

You can export the trained models using this:

```shell
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix ${CHECKPOINT_PATH}/model.ckpt-${CHECKPOINT_NUMBER} \
    --output_directory ${MODEL_OUTPUT_PATH}/
```

Please modify the name of trained_checkpoint_prefix, like checkpoints_dir/model.ckpt-*number*, where *number* is the current num_step of the checkpoint which you want to export.

## Eval

You can evaluate the performance of your models using:

```shell
tensorboard --loir=${checkpoint_path}/
```

## Run

Just run:

```shell
python detect_face.py
```
