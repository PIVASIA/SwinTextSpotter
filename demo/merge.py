# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import torch
import numpy as np
import shapely
from shapely.geometry import Point, Polygon
from genericmask import GenericMask
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.structures import BitMasks, Boxes, BoxMode, Keypoints, PolygonMasks, RotatedBoxes
from predictor import VisualizationDemo
# from beamsearch import beam_search
from editdistance import eval
import operator
# constants
WINDOW_NAME = "COCO detections"

dictionary = "aàáạảãâầấậẩẫăằắặẳẵAÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪeèéẹẻẽêềếệểễEÈÉẸẺẼÊỀẾỆỂỄoòóọỏõôồốộổỗơờớợởỡOÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠiìíịỉĩIÌÍỊỈĨuùúụủũưừứựửữƯỪỨỰỬỮUÙÚỤỦŨyỳýỵỷỹYỲÝỴỶỸ"


def make_groups():
    groups = []
    i = 0
    while i < len(dictionary) - 5:
        group = [c for c in dictionary[i : i + 6]]
        i += 6
        groups.append(group)
    return groups

groups = make_groups()

TONES = ["", "ˋ", "ˊ", "﹒", "ˀ", "˜"]
SOURCES = ["ă", "â", "Ă", "Â", "ê", "Ê", "ô", "ơ", "Ô", "Ơ", "ư", "Ư", "Đ", "đ"]
TARGETS = ["aˇ", "aˆ", "Aˇ", "Aˆ", "eˆ", "Eˆ", "oˆ", "o˒", "Oˆ", "O˒", "u˒", "U˒", "D‑", "d‑"]

def parse_tone(word):
    res = ""
    tone = ""
    for char in word:
        if char in dictionary:
            for group in groups:
                if char in group:
                    if tone == "":
                        tone = TONES[group.index(char)]
                    res += group[0]
        else:
            res += char
    res += tone
    return res

def full_parse(word):
    word = parse_tone(word)
    res = ""
    for char in word:
        if char in SOURCES:
            res += TARGETS[SOURCES.index(char)]
        else:
            res += char
    return res

def correct_tone_position(word):
    word = word[:-1]
    if len(word) < 2:
        pass
    first_ord_char = ""
    second_order_char = ""
    for char in word:
        for group in groups:
            if char in group:
                second_order_char = first_ord_char
                first_ord_char = group[0]
    if word[-1] == first_ord_char and second_order_char != "":
        pair_chars = ["qu", "Qu", "qU", "QU", "gi", "Gi", "gI", "GI"]
        for pair in pair_chars:
            if pair in word and second_order_char in ["u", "U", "i", "I"]:
                return first_ord_char
        return second_order_char
    return first_ord_char

def decoder(recognition):
    for char in TARGETS:
        recognition = recognition.replace(char, SOURCES[TARGETS.index(char)])
    if len(recognition) < 1:
        return recognition
    if recognition[-1] in TONES:
        if len(recognition) < 2:
            return recognition
        replace_char = correct_tone_position(recognition)
        tone = recognition[-1]
        recognition = recognition[:-1]
        for group in groups:
            if replace_char in group:
                recognition = recognition.replace(replace_char, group[TONES.index(tone)])
    return recognition

def decode_recognition(rec):
    CTLABELS = [" ","!",'"',"#","$","%","&","'","(",")","*","+",",","-",".","/","0","1","2","3","4","5","6","7","8","9",":",";","<","=",">","?","@","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","[","\\","]","^","_","`","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","{","|","}","~","ˋ","ˊ","﹒","ˀ","˜","ˇ","ˆ","˒","‑",]
    last_char = False
    s = ''
    for c in rec:
        c = int(c)
        if 0<c < 107:
                s += CTLABELS[c-1]
                last_char = c
        elif c == 0:
            s += u''
        else:
            last_char = False
    if len(s) == 0:
        s = ' '
    s = decoder(s)
    return s

def get_mini_boxes(contour, max_x, min_x, thr):
    bounding_box = cv2.minAreaRect(contour)
    # print('bbox', bounding_box)
    bounding_box = list(bounding_box)
    bounding_box[1] = list(bounding_box[1])
    if bounding_box[2]<=45:
        bounding_box[1][0] = bounding_box[1][0]*thr
    else:
        bounding_box[1][1] = bounding_box[1][1]*thr
    bounding_box[1] = tuple(bounding_box[1])
    bounding_box = tuple(bounding_box)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2
    # p1 = np.array([min_x, points[index_1][1]])
    # p2 = np.array([max_x, points[index_2][1]])
    # p3 = np.array([max_x, points[index_3][1]])
    # p4 = np.array([min_x, points[index_4][1]])
    # box = [p1, p2, p3, p4]
    box = [
        points[index_1], points[index_2], points[index_3], points[index_4]
    ]
    return box

def get_mini_boxes_1(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [
        points[index_1], points[index_2], points[index_3], points[index_4]
    ]
    return box

def calculate_iou(box_1, box_2):
    # print(box_1, box_2)
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    # print(poly_1.union(poly_2).area)
    try:
        iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    except:
        iou = 0
    return iou

# def get_key(val, my_dict):
#     for key, value in my_dict.items():
#         if val in value:
#             return key

def merge_boxes(boxes, recs, trh):
    dict_bbox = {}
    x=0
    for i in range(len(boxes)-2):
        tmp_box = [i]
        db_copy1 = dict_bbox.copy()
        for key, value in db_copy1.items():
            if i in value:
                tmp_box = db_copy1[key]
                del dict_bbox[key]
                break
        for j in range(i+1, len(boxes)-1):
            ba = cv2.minAreaRect(boxes[i].reshape(-1,1,2).astype(int))
            bb = cv2.minAreaRect(boxes[j].reshape(-1,1,2).astype(int))
            iou = calculate_iou(boxes[i], boxes[j])
            # scr = min(ba[1][0], bb[1][0])/max(ba[1][0], bb[1][0])
            if iou > trh:
                db_copy = dict_bbox.copy()
                check = False
                for key, value in db_copy.items():
                    if i in value:
                        check = True
                        tmp_box.remove(i)
                        tmp_box.extend(db_copy[key])
                        del dict_bbox[key]
                        break
                if check == False:
                    tmp_box.append(j)
        dict_bbox[x] = tmp_box
        x+=1
    recs_out = []
    db_clone = {}
    for key, value in dict_bbox.items():
        db_clone[key] = list(set(value))
    for key, value in db_clone.items():
        tmp_str = []
        for i in value:
            tmp_str.append([recs[i], cv2.minAreaRect(boxes[i].reshape(-1,1,2).astype(int))[0][0]])
        recs_out.append(tmp_str)
    return db_clone, recs_out

def combine(dict_box, h, w, boxes):
    bboxs = []
    for key, db in dict_box.items():
        list_box = []
        for j in db:
            list_box.append(boxes[j])
        h1 = h
        h2 = 0
        h3 = 0
        h4 = h
        w1 = w
        w2 = w
        w3 = 0
        w4 = 0
        for box in list_box:
            if box[0,0] < h1:
                h1 = box[0,0]
            if box[1,0] > h2:
                h2 = box[1,0]
            if box[2,0] > h3:
                h3 = box[2,0]
            if box[3,0] < h4:
                h4 = box[3,0]
            if box[0,1] < w1:
                w1 = box[0,1]
            if box[1,1] < w2:
                w2 = box[1,1]
            if box[2,1] > w3:
                w3 = box[2,1]
            if box[3,1] > w4:
                w4 = box[3,1]                       
            tmp = np.array([[h1,w1],[h2,w2],[h3,w3],[h4,w4]])
        bboxs.append(tmp.astype(np.int16))
    return bboxs

def rec_to_str(recs):
    rec_1 = []
    for rec in recs:
        i =  sorted(rec, key=lambda x: x[1])
        print(i)
        i = " ".join(decoder(item[0]) for item in i)
        rec_1.append(i)
    return rec_1


def scale_points(mask):
    mask_tmp = mask.copy()
    for i in range(2,len(mask_tmp)-2):
        for j in range(2,len(mask_tmp[i])-2):
            if mask_tmp[i][j] != 0:
                mask[i-2][j-2] = mask[i-2][j-1] = mask[i-2][j] = mask[i-2][j+1] = mask[i-2][j+2] = mask[i-1][j-2] = mask[i-1][j-1] = mask[i-1][j] = mask[i-1][j+1] = mask[i-1][j+2] = mask[i][j-2] = mask[i][j-1] = mask[i][j+1] = mask[i][j+2] = mask[i+1][j-2] = mask[i+1][j-1] = mask[i+1][j] = mask[i+1][j+1] = mask[i+1][j+2] = mask[i+2][j-2] = mask[i+2][j-1] = mask[i+2][j] = mask[i+2][j+1] = mask[i+2][j+2] = mask_tmp[i][j]
    return mask

def convert_boxes(boxes):
    if isinstance(boxes, Boxes) or isinstance(boxes, RotatedBoxes):
        return boxes.tensor.numpy()
    else:
        return np.asarray(boxes)

def convert_masks(masks_or_polygons, h, w):
    m = masks_or_polygons
    if isinstance(m, PolygonMasks):
        m = m.polygons
    if isinstance(m, BitMasks):
        m = m.tensor.numpy()
    if isinstance(m, torch.Tensor):
        m = m.numpy()
    ret = []
    for x in m:
        if isinstance(x, GenericMask):
            ret.append(x)
        else:
            ret.append(GenericMask(x, h, w))
    return ret

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()

    from projects.SWINTS.swints import add_SWINTS_config
    add_SWINTS_config(cfg)
    # -----

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--inputfile", nargs="+", help="A list of array of segmentation")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    hh = []
    if args.inputfile:
        path_segment = args.inputfile[0]
    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            print(path)
            txt_name = str(path.split("/")[-1].split(".")[0]) + '.txt'
            txt_file = os.path.join(path_segment, txt_name)
            img = read_image(path, format="BGR")
            h, w, _ = img.shape
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img, args.confidence_threshold, path)
            # time_1 = time.time()-start_time

            mask = np.loadtxt(txt_file,  dtype=np.int32)
            # time_2 = time.time()-time_1
            mmax = np.amax(mask)
            if mmax == 0:
                mmax = 1
            mask = scale_points(mask)
            # time_3 = time.time()-time_2

            outs = cv2.findContours((mask * int(255/mmax)).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if len(outs) == 3:
                img, contours, _ = outs[0], outs[1], outs[2]
            elif len(outs) == 2:
                contours, _ = outs[0], outs[1]

            box_sign = []
            for contour in contours:
                points = get_mini_boxes_1(contour)
                points = np.array(points)
                box_sign.append(points)
            # time_4 = time.time()-time_3
            dict_box_sign = {}
            dict_box_sign_out = {}
            dict_rec_sign = {}
            dict_rec_sign_out = {}
            in_signboard = 0
            # full_box = 0
            
            for i in range(len(box_sign)):
                dict_box_sign[i] = []
                dict_box_sign_out[i] = []
                dict_rec_sign[i] = []
                dict_rec_sign_out[i] = []
            list_limit = []
            for sig in box_sign:
                # print(sig)
                max_x = max(sig[0][0], sig[1][0], sig[2][0], sig[3][0])
                min_x = min(sig[0][0], sig[1][0], sig[2][0], sig[3][0])
                list_limit.append([max_x, min_x])
            if "instances" in predictions:

                beziers = []
                segments = []
                recc = []
                scoress = []
                instances = predictions["instances"].to(torch.device("cpu"))
                # print("instance",type(instances))
                instances = instances[instances.scores > args.confidence_threshold]
                boxes = instances.pred_boxes if instances.has("pred_boxes") else None
                scores = instances.scores if instances.has("scores") else None
                # classes = instances.pred_classes if instances.has("pred_classes") else None
                recs = instances.pred_rec if instances.has("pred_rec") else None
                # rec_score = instances.pred_rec_score if instances.has("pred_rec_score") else None

                masks = np.asarray(instances.pred_masks)
                masks = [GenericMask(x, h, w) for x in masks]
                masks = convert_masks(masks, h, w)
                polys = []
                for mask in masks:
                    polys.append(np.concatenate(mask.polygons).reshape(-1,2).tolist())

                #text box into signboard box
                for bezier, rec, score in zip(polys, recs, scores):
                    # print(bezier)
                    if score >=0.5:
                        bezier = np.array(bezier, dtype='int').reshape(-1,1,2)
                        bounding_box = cv2.minAreaRect(bezier)
                        midpoint = Point(bounding_box[0])
                        for i in range(len(box_sign)):
                            poly = Polygon(box_sign[i])
                            if midpoint.within(poly):
                                in_signboard+=1 
                                dict_box_sign[i].append(bezier)
                                dict_rec_sign[i].append(full_parse(decode_recognition(rec)))
                # time_5 = time.time()-time_4
                for i in range(len(dict_box_sign)):
                    boxes = []
                    reces = []
                    for bezier, rec in zip(dict_box_sign[i], dict_rec_sign[i]):
                        unclip_ratio = 1.0
                        bezier = bezier.reshape(-1,1,2)
                        points = get_mini_boxes(bezier, list_limit[i][0], list_limit[i][1], 3)
                        box = np.array(points, dtype=np.int16)

                        box[:, 0] = np.clip(np.round(box[:, 0]), 0, w)
                        box[:, 1] = np.clip(np.round(box[:, 1]), 0, h)
                        
                        boxes.append(box.astype(np.int16))
                        reces.append(rec)

                    dict_box, rec_out = merge_boxes(boxes, reces, 0.1)

                    rec_outs = rec_to_str(rec_out)
                    bboxs = combine(dict_box, h, w, boxes)
                    # print(rec_outs)
                    dict_box_sign_out[i] = bboxs
                    dict_rec_sign_out[i] = rec_outs
                # time_6 = time.time()-time_5
            #Visualize image after merge boxes
            img_draw = cv2.imread(path)
            for i in range(len(dict_box_sign_out)):
                for j in range(len(dict_box_sign_out[i])):
                    pts = dict_box_sign_out[i][j]
                    x, y = pts[0][0], pts[0][1]
                    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                    isClosed = True
                    color = (255, 0, 0)
                    thickness = 2
                    img_draw = cv2.polylines(img_draw, [pts], isClosed, color, thickness)
                    cv2.putText(img_draw, dict_rec_sign_out[i][j], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            # print(time_1, time_2, time_3, time_4, time_5, time_6)
            txt_name = str(path.split("/")[-1].split(".")[0]) + '.txt'
            img_name = str(path.split("/")[-1].split(".")[0]) + '.jpg'
            if args.output: 
                output_path = os.path.join(args.output, txt_name)
                output_file = open(output_path, 'w+', encoding='utf-8')
                # output_file.write(str(in_signboard) + " ")
                # output_file.write(str(full_box) + '\n')
                output_file.write(str(dict_box_sign_out))
                output_file.write(str(dict_rec_sign_out))
            cv2.imwrite("../output/output_visualize/"+img_name, img_draw)
            logger.info(
                "{}: detected {} instances in {:.2f}s".format(
                    path, len(predictions["instances"]), time.time() - start_time
                )
            )