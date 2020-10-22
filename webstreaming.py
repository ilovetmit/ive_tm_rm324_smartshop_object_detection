# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
from pandas import np
import threading
import argparse
import datetime
import imutils
import requests
import six
import json
import time
import os
import cv2
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

product_data = []
product_score_data = []
product_json_data = []
json_time = 10

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 11

# Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# initialize the output frame and a lock used to ensure thread-safe exchanges of the output frames
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to warmup
vs = VideoStream(src=0).start()
time.sleep(2.0)


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


def detect_motion(frameCount):
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, lock

    # loop over frames from the video stream
    while True:
        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        frame = vs.read()
        frame = imutils.resize(frame, width=600)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # draw the results of the detection
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=6,
            min_score_thresh=0.80)

        load_object_data(
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            min_score_thresh=0.80)

        # grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)

        # acquire the lock, set the output frame, and release the lock
        with lock:
            outputFrame = frame.copy()


def load_object_data(
        boxes,
        classes,
        scores,
        p_category_index,
        min_score_thresh=.75,
        agnostic_mode=False,
        skip_scores=False,
        skip_labels=False):
    product_data.clear()
    product_score_data.clear()
    for i in range(min(boxes.shape[0], boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            product_name = ''
            display_str = ''
            product_score = 0
            if not skip_labels:
                if not agnostic_mode:
                    if classes[i] in six.viewkeys(p_category_index):
                        class_name = p_category_index[classes[i]]['name']
                    else:
                        class_name = 'N/A'
                    display_str = str(class_name)
                    product_name = str(class_name)
            if not skip_scores:
                if not display_str:
                    display_str = '{}%'.format(int(100 * scores[i]))
                    product_score = '%.2f' % (100 * scores[i])
                else:
                    display_str = '{}: {}%'.format(display_str, int(100 * scores[i]))
                    product_score = '%.2f' % (100 * scores[i])
            product_data.append(str(product_name))
            product_score_data.append(product_score)
    product_json_data.append(list(product_data))
    # print(list(product_data))
    # print(product_json_data)
    if len(product_json_data) >= json_time:
        data = {
            'product': product_json_data,
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url='http://192.168.15.175/system/api/v1/object_detection', headers=headers,
                                 data=json.dumps(data))
        product_json_data.clear()
        print(response)


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/product_detail")
def product_detail():
    return Response(detail())


def detail():
    timestamp = datetime.datetime.now()
    data = {
        'time': timestamp.strftime("%A %d %B %Y %I:%M:%S%p"),
        'product': product_data,
        'score': product_score_data,
    }
    return json.dumps(data)


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
                    help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_motion, args=(
        args["frame_count"],))
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()
