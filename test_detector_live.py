import csv
import datetime
import glob
import math
import os
import random
import time
from ctypes import *

import cv2
import numpy as np
import pyzed.camera as zcam
import pyzed.core as core
import pyzed.defines as sl
import pyzed.types as tp


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


# lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/tmp/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res


def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c, h, w = arr.shape[0:3]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr

def translate_coordinates(res, image_width, image_height, yolo_img_width, yolo_img_height):
    trans_matrix = ([image_width / yolo_img_width, 0, 0],
                    [0, image_height / yolo_img_height, 0],
                    [0, 0, 1])
    coordinates_list = []
    for coordinates in res:
        coordinates_list.append(coordinates[2])

    i = 0
    coordinates_matrix = np.ones((3, int(2 * coordinates_list.__len__())))
    for coordinate in coordinates_list:
        coordinates_matrix[0][i] = coordinate[0]
        coordinates_matrix[1][i] = coordinate[1]
        coordinates_matrix[0][i + 1] = coordinate[2]
        coordinates_matrix[1][i + 1] = coordinate[3]
        i = i + 2
    trans_matrix = np.asarray(trans_matrix)
    new_coordinates = np.dot(trans_matrix, coordinates_matrix)

    return new_coordinates


def new_res_value(res, new_coordinates):
    res2 = []
    i = 0
    for item in res:
        res2.append((item[0], item[1], (
        new_coordinates[0][i], new_coordinates[1][i], new_coordinates[0][i + 1], new_coordinates[1][i + 1]), item[3]))
        i = i + 2
    return res2


def run_on_image_ref_bak(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
    """
    Performs the detection on image
    """
    class_list = ['person', 'car']
    custom_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    custom_image = cv2.resize(custom_image, (lib.network_width(
        net), lib.network_height(net)), interpolation=cv2.INTER_LINEAR)
    im, arr = array_to_image(custom_image)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    res = []
    if debug:
        print("about to range")
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                lbl = meta.names[i].decode("utf-8")
                class_flag = lbl in class_list
                if class_flag == True:
                    x1 = int(b.x - b.w / 2)
                    y1 = int(b.y - b.h / 2)
                    yExtent = int(b.h)
                    xEntent = int(b.w)
                    x2 = x1 + xEntent
                    y2 = y1 + yExtent
                    res.append((lbl, dets[j].prob[i], (x1, y1, x2, y2), i))
    new_coordinates = translate_coordinates(res, image.shape[1], image.shape[0], lib.network_width(net),
                                            lib.network_height(net))
    res2 = new_res_value(res, new_coordinates)
    # res2 = sorted(res2, key=lambda x: -x[1])
    free_detections(dets, num)
    return res2

## img list
def get_img_list(in_dir, ext, is_srt=True):
    src_dir = in_dir + '/*.' + ext
    list_images = glob.glob(src_dir)
    if is_srt:
        list_images.sort()
    return list_images


def current_date_time():
    ts = time.time()
    current_date_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    return current_date_time


def create_dir():
    # create dir based on current date time
    time_stamp = current_date_time()
    if not os.path.exists('zed'):
        os.mkdir('./zed')
    os.mkdir('./zed/' + time_stamp)
    dir_path = './zed/' + time_stamp
    return dir_path


directory_path = create_dir()
log_file_path = os.path.join(directory_path, current_date_time() + '.csv')


def main_without_zed():
    ##### yolo ###
    net = load_net(b"/tmp/darknet/cfg/yolov2.cfg", b"/tmp/darknet/yolov2.weights", 0)
    meta = load_meta(b"/tmp/darknet/cfg/coco.data")

    # for log file name time stamping
    images_dir = '/tmp/darknet/zed-python-api/pyzed/zed_pycharm/Images'
    list_images = get_img_list(images_dir, ext='png')
    for item in list_images:
        frame = cv2.imread(item)
        out_list = run_on_image_ref_bak(net, meta, frame)
        for item2 in out_list:
            lbl = item2[0]
            roi = item2[2]
            cv2.rectangle(frame, (int(roi[0]), int(roi[1])), (int(roi[2]), int(roi[3])), (0, 255, 0), 2)
            cv2.putText(frame, 'Class = ' + str(lbl), (int(roi[0]) + 2, int(roi[1]) + 15), 1, 1,
                        (0, 255, 255), 2, cv2.LINE_AA)

        print('done')
        cv2.imshow('Detected_Output_Image', frame)
        key = cv2.waitKey(0)

        if key == 27:  # if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            break
    print('-------------------------program ends------------------------------')


def main():
    ##### yolo ###

    net = load_net(b"/tmp/darknet/cfg/yolov2.cfg", b"/tmp/darknet/yolov2.weights", 0)
    meta = load_meta(b"/tmp/darknet/cfg/coco.data")

    # Create a PyZEDCamera object
    zed = zcam.PyZEDCamera()

    # Create a PyInitParameters object and set configuration parameters
    init_params = zcam.PyInitParameters()
    # These settings adjust the level of accuracy, range and computational performance of the depth sensing module.
    # available modes are Ultra, quality, medium & performance
    init_params.depth_mode = sl.PyDEPTH_MODE.PyDEPTH_MODE_ULTRA  # Use ULTRA depth mode for better depth accuracy
    init_params.coordinate_units = sl.PyUNIT.PyUNIT_METER  # Use meter units (for depth measurements)
    init_params.camera_fps = 30  # camera FPS
    init_params.camera_resolution = sl.PyRESOLUTION.PyRESOLUTION_HD1080  # camera resolution

    # Open the camera
    err = zed.open(init_params)
    if err != tp.PyERROR_CODE.PySUCCESS:
        exit(1)

    # Create and set PyRuntimeParameters after opening the camera
    runtime_parameters = zcam.PyRuntimeParameters()
    runtime_parameters.sensing_mode = sl.PySENSING_MODE.PySENSING_MODE_FILL  # Use STANDARD sensing mode

    right_image = core.PyMat()
    left_image = core.PyMat()
    depth = core.PyMat()
    point_cloud = core.PyMat()

    # this varaible we only used to display the depth view. its display in 8 bit
    depth_image_for_view_8_bit = core.PyMat()

    # for log file name time stamping

    while True:
        # A new image is available if grab() returns PySUCCESS
        if zed.grab(runtime_parameters) == tp.PyERROR_CODE.PySUCCESS:
            # Retrieve left image
            zed.retrieve_image(left_image, sl.PyVIEW.PyVIEW_LEFT)
            # Retrieve right image
            zed.retrieve_image(right_image, sl.PyVIEW.PyVIEW_RIGHT)
            # Retrieve depth map. Depth is aligned on the left image
            zed.retrieve_measure(depth, sl.PyMEASURE.PyMEASURE_DEPTH)
            # Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieve_measure(point_cloud, sl.PyMEASURE.PyMEASURE_XYZRGBA)
            # Retrieve depth image for display only
            zed.retrieve_image(depth_image_for_view_8_bit, sl.PyVIEW.PyVIEW_DEPTH)

            # flipping the image 180 degree(vertically)
            # left_flipped_image_180 = cv2.rotate(left_image.get_data(), rotateCode=cv2.ROTATE_180)

            frame = left_image.get_data()
            # frame = cv2.resize(frame, (lib.network_width(
            # net), lib.network_height(net)), interpolation=cv2.INTER_LINEAR)

            out_list = run_on_image_ref_bak(net, meta, frame)
            for item in out_list:
                lbl = item[0]
                roi = item[2]
                x_mid = int((roi[0] + roi[2]) / 2)
                y_mid = int((roi[1] + roi[3]) / 2)
                dist = round(get_depth(x_mid, y_mid, point_cloud), 2)
                dist = get_depth(x_mid, y_mid, point_cloud)
                dist = "{0:.2f}".format(dist)

                cv2.rectangle(frame, (int(roi[0]), int(roi[1])), (int(roi[2]), int(roi[3])), (0, 255, 0), 2)
                # cv2.putText(frame, lbl + ' ' + dist + ' mtr', (roi[0] + 2, roi[1] + 15), 1, 1,(0, 255, 255), 2, cv2.LINE_AA)

            print('done')
            # frame = cv2.resize(frame, (1280, 960), interpolation=cv2.INTER_LINEAR)
            cv2.imshow('Detected_Output_Image', frame)
            key = cv2.waitKey(1)

            if key == 27:  # if ESC is pressed, exit loop
                cv2.destroyAllWindows()
                break
    print('-------------------------program ends------------------------------')


def get_depth(x, y, point_cloud):
    # x = round(image_width / 2)
    # y = round(image_height/ 2)
    err, point_cloud_value = point_cloud.get_value(x, y)
    # print('X = ', x, 'Y = ', y, 'Err = ', err )
    distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                         point_cloud_value[1] * point_cloud_value[1] +
                         point_cloud_value[2] * point_cloud_value[2])
    # print('distance = ', distance)
    return distance


def write_log(log_values, i):
    f = open(log_file_path, 'a')
    writer = csv.writer(f, delimiter=',')
    if (i == 0):
        header = ['ImageName', 'DepthValue', 'PointCloudDepth', 'Xpoint', 'Ypoint', 'Zpoint']
        writer.writerow(header)
    writer.writerow(log_values)
    f.close()


if __name__ == "__main__":
    main_without_zed()