# Real-time Object Detection Product

_IVE FYP 1920 Module_

- [Real-time Object Detection Product Price](#real-time-object-detection-product-price)
  - [System Requirements](#system-requirements)
  - [Installation instructions](#installation-instructions)
  - [Model Info](#model-info)

## System Requirements

- Python 3.6
- TensorFlow 1.14
- Flask
- Pandas

## Installation instructions

1. Extract the archive and put it in the folder you want

2. `git clone https://github.com/tensorflow/models`

3. install Python library
    ```shell script
    pip install Cython
    pip install contextlib2
    pip install pillow
    pip install lxml
    pip install matplotlib
    pip install pandas
    pip install opencv-python
    pip install tensorflow=1
    ```

4. Add necessary environment variables
    ```shell script
    export PYTHONPATH=$PYTHONPATH:<PATH_TO_TF>/TensorFlow/models/research
    export PYTHONPATH=$PYTHONPATH:<PATH_TO_TF>/TensorFlow/models/research/object_detection
    export PYTHONPATH=$PYTHONPATH:<PATH_TO_TF>/TensorFlow/models/research/slim
    ```

5. Compile Protobufs
    ```shell script
    protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto
    ```

6. navigate to `tensorflow/models/research` and run:
    ```shell script
    python setup.py build
    python setup.py install
    ```

7. navigate to this project folder

8. run `python webstreaming.py --ip 127.0.0.1 --port 8000`

9. enjoy [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Model Info

- Labeling data tools : [LabelImg](https://github.com/tzutalin/labelImg)
- Training model:  [faster_rcnn_inception_v2_coco](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models)
- Dataset size: 313 photos
- Training step: 20001 
- Number of products: 11

