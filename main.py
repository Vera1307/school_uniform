from models import *
from utils import *
from human_detector import DetectHumans

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import requests

import cv2
import ffmpeg

from tqdm import tqdm
import time
import csv

import sys, getopt
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

from model import CNN

model = torch.load("cnn_model.pt")

accetpable = {
    "top" : [0, 2, 4, 6],
    "bottom" : [1, 3],
    "foot" : [5, 7, 9]
}

index_class_mapper = {
    "0" : "tricou",
    "1" : "pantaloni",
    "2" : "pulover",
    "3" : "rochie",
    "4" : "palton",
    "5" : "sandale",
    "6" : "camasa",
    "7" : "adidasi",
    "8" : "geanta",
    "9" : "botine"
}

dh = DetectHumans()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def predict_img(img, body_part):
    if len(img) == 0:
        return None
    img = cv2.resize(img, (28, 28))
    img = 255 - cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    pil_img = Image.fromarray(img)
    tens = transforms.ToTensor()(pil_img)
    indices = torch.sort(model(tens.unsqueeze(dim=1))[0], descending = True).indices.tolist()
    acceptable_clothes = acceptable[body_part]
    for i in indices:
        if i in acceptable_clothes:
            predicted_class = index_class_mapper[str(i)]
            return predicted_class
    return None

def create_squares(landmarks, img_shape):
    top_indexes = [mp_pose.PoseLandmark.LEFT_SHOULDER,
                   mp_pose.PoseLandmark.RIGHT_SHOULDER,
                   mp_pose.PoseLandmark.LEFT_ELBOW,
                   mp_pose.PoseLandmark.RIGHT_ELBOW,
                   mp_pose.PoseLandmark.LEFT_WRIST,
                   mp_pose.PoseLandmark.RIGHT_WRIST,
                   mp_pose.PoseLandmark.LEFT_PINKY,
                   mp_pose.PoseLandmark.RIGHT_PINKY,
                   mp_pose.PoseLandmark.LEFT_INDEX,
                   mp_pose.PoseLandmark.RIGHT_INDEX,
                   mp_pose.PoseLandmark.LEFT_THUMB,
                   mp_pose.PoseLandmark.RIGHT_THUMB,
                   mp_pose.PoseLandmark.LEFT_HIP,
                   mp_pose.PoseLandmark.RIGHT_HIP]
    bottom_indexes = [23, 24, 25, 26, 27, 28]
    left_foot_indexes = [27, 29, 31]
    right_foot_indexes = [28, 30, 32]
    top_landmarks = np.array([
        (landmarks[i].x, landmarks[i].y)
        for i in top_indexes
    ])
    top_landmarks = np.multiply(top_landmarks, np.array(img_shape[:2][::-1]))
    bottom_landmarks = np.array([
        (landmarks[i].x, landmarks[i].y)
        for i in bottom_indexes
    ])
    bottom_landmarks = np.multiply(bottom_landmarks, np.array(img_shape[:2][::-1]))
    left_foot_landmarks = np.array([
        (landmarks[i].x, landmarks[i].y)
        for i in left_foot_indexes
    ])
    left_foot_landmarks = np.multiply(left_foot_landmarks, np.array(img_shape[:2][::-1]))
    right_foot_landmarks = np.array([
        (landmarks[i].x, landmarks[i].y)
        for i in right_foot_indexes
    ])
    right_foot_landmarks = np.multiply(right_foot_landmarks, np.array(img_shape[:2][::-1]))
    top_coordinates = tuple([int(top_landmarks[:, 0].min() * 0.3),
                            int(top_landmarks[:, 1].min() * 0.3),
                            int(top_landmarks[:, 0].max() * 1.2),
                            int(top_landmarks[:, 1].max() * 1.2)])
    bottom_coordinates = tuple([int(bottom_landmarks[:, 0].min() * 0.3),
                            int(bottom_landmarks[:, 1].min()),
                            int(bottom_landmarks[:, 0].max() * 1.3),
                            int(bottom_landmarks[:, 1].max())])
    left_foot_coordinates = tuple([int(left_foot_landmarks[:, 0].min() * 0.9),
                            int(left_foot_landmarks[:, 1].min() * 0.9),
                            int(left_foot_landmarks[:, 0].max() * 1.1),
                            int(left_foot_landmarks[:, 1].max() * 1.1)])
    right_foot_coordinates = tuple([int(right_foot_landmarks[:, 0].min() * 0.9),
                            int(right_foot_landmarks[:, 1].min() * 0.9),
                            int(right_foot_landmarks[:, 0].max() * 1.1),
                            int(right_foot_landmarks[:, 1].max() * 1.1)])
    return top_coordinates, bottom_coordinates, left_foot_coordinates, right_foot_coordinates


def extract_clothes_groups(image):
    clothes_delta = {}
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:
        image_height, image_width, _ = image.shape
        if image_height != 0 and image_width != 0:
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                coordinates = create_squares(results.pose_landmarks.landmark, image.shape)
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                top_subimg = image[coordinates[0][1]:coordinates[0][3], coordinates[0][0]:coordinates[0][2]]
                top_class = predict_img(top_subimg, "top")

                clothes_delta["top"] = {
                    "x1" : coordinates[0][0],
                    "y1" : coordinates[0][1],
                    "x2" : coordinates[0][2],
                    "y2" : coordinates[0][3],
                    "class" : top_class
                }
                bottom_subimg = image[coordinates[1][1]:coordinates[1][3], coordinates[1][0]:coordinates[1][2]]
                bottom_class = predict_img(bottom_subimg, "bottom")

                clothes_delta["bottom"] = {
                    "x1" : coordinates[1][0],
                    "y1" : coordinates[1][1],
                    "x2" : coordinates[1][2],
                    "y2" : coordinates[1][3],
                    "class" : bottom_class
                }
                left_foot_subimg = image[coordinates[2][1]:coordinates[2][3], coordinates[2][0]:coordinates[2][2]]
                left_foot_class = predict_img(left_foot_subimg, "foot")

                clothes_delta["left_foot"] = {
                    "x1" : coordinates[2][0],
                    "y1" : coordinates[2][1],
                    "x2" : coordinates[2][2],
                    "y2" : coordinates[2][3],
                    "class" : left_foot_class
                }
                right_foot_subimg = image[coordinates[3][1]:coordinates[3][3], coordinates[3][0]:coordinates[3][2]]
                right_foot_class = predict_img(right_foot_subimg, "foot")

                clothes_delta["right_foot"] = {
                    "x1" : coordinates[3][0],
                    "y1" : coordinates[3][1],
                    "x2" : coordinates[3][2],
                    "y2" : coordinates[3][3],
                    "class" : right_foot_class
                }
                return clothes_delta
        else:
            return None


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    copy_img = img.copy()
    if img is not None:
        humans = dh.detect_image(img)
        print(len(humans))
        clothes = []

        for person in humans:
            sub_img = img[person["y1"]:person["y2"], person["x1"]:person["x2"]]
            if abs(person["x1"] - person["x2"]) * abs(person["y1"] - person["y2"]) > 0.05 * img.shape[0] * img.shape[1]:
                clothes_coordinates = extract_clothes_groups(sub_img)
                if clothes_coordinates:
                    clothes.append(
                        {
                            "top" : {
                                "x1" : person["x1"] + clothes_coordinates["top"]["x1"],
                                "y1" : person["y1"] + clothes_coordinates["top"]["y1"],
                                "x2" : person["x1"] + clothes_coordinates["top"]["x2"],
                                "y2" : person["y1"] + clothes_coordinates["top"]["y2"],
                                "class" : clothes_coordinates["top"]["class"]
                            },
                            "bottom" : {
                                "x1" : person["x1"] + clothes_coordinates["bottom"]["x1"],
                                "y1" : person["y1"] + clothes_coordinates["bottom"]["y1"],
                                "x2" : person["x1"] + clothes_coordinates["bottom"]["x2"],
                                "y2" : person["y1"] + clothes_coordinates["bottom"]["y2"],
                                "class" : clothes_coordinates["bottom"]["class"]
                            },
                            "left_foot" : {
                                "x1" : person["x1"] + clothes_coordinates["left_foot"]["x1"],
                                "y1" : person["y1"] + clothes_coordinates["left_foot"]["y1"],
                                "x2" : person["x1"] + clothes_coordinates["left_foot"]["x2"],
                                "y2" : person["y1"] + clothes_coordinates["left_foot"]["y2"],
                                "class" : clothes_coordinates["left_foot"]["class"]
                            },
                            "right_foot" : {
                                "x1" : person["x1"] + clothes_coordinates["right_foot"]["x1"],
                                "y1" : person["y1"] + clothes_coordinates["right_foot"]["y1"],
                                "x2" : person["x1"] + clothes_coordinates["right_foot"]["x2"],
                                "y2" : person["y1"] + clothes_coordinates["right_foot"]["y2"],
                                "class" : clothes_coordinates["right_foot"]["class"]
                            }
                        }
                    )

        for clothe in clothes:
            if clothe["top"]["class"]:
                cv2.rectangle(copy_img, (clothe["top"]["x1"], clothe["top"]["y1"]), (clothe["top"]["x2"], clothe["top"]["y2"]), (225, 255, 0), 2)
                cv2.putText(copy_img, clothe["top"]["class"], (clothe["top"]["x1"], clothe["top"]["y1"]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 0, 0), 2, cv2.LINE_AA)
            if clothe["bottom"]["class"]:
                cv2.rectangle(copy_img, (clothe["bottom"]["x1"], clothe["bottom"]["y1"]), (clothe["bottom"]["x2"], clothe["bottom"]["y2"]), (225, 255, 0), 2)
                cv2.putText(copy_img, clothe["bottom"]["class"], (clothe["bottom"]["x1"], clothe["bottom"]["y1"]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if clothe["left_foot"]["class"]:

                cv2.rectangle(copy_img, (clothe["left_foot"]["x1"], clothe["left_foot"]["y1"]), (clothe["left_foot"]["x2"], clothe["left_foot"]["y2"]), (225, 255, 0), 2)
                cv2.putText(copy_img, clothe["left_foot"]["class"], (clothe["left_foot"]["x1"], clothe["left_foot"]["y1"]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 0, 255), 2, cv2.LINE_AA)
            if clothe["right_foot"]["class"]:
                cv2.rectangle(copy_img, (clothe["right_foot"]["x1"], clothe["right_foot"]["y1"]), (clothe["right_foot"]["x2"], clothe["right_foot"]["y2"]), (225, 255, 0), 2)
                cv2.putText(copy_img, clothe["right_foot"]["class"], (clothe["right_foot"]["x1"], clothe["right_foot"]["y1"]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('image', copy_img)
        if cv2.waitKey(5) & 0xFF == 27:
            break