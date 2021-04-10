import cv2
from pytool import social_distancing_config as config
from pytool.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os
from PIL import Image
# defining face detector
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6
class VideoCamera(object):
    def __init__(self):
       #capturing video
       self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        #releasing camera
        self.video.release()
    def get_frame(self):
        #extracting frames

            # Our operations on the frame come here
            
#-------------------------------------------------------------------------------------
            ap = argparse.ArgumentParser()
            ap.add_argument("-i", "--input", type=str, default="",
                help="path to (optional) input video file")
            ap.add_argument("-o", "--output", type=str, default="",
                help="path to (optional) output video file")
            ap.add_argument("-d", "--display", type=int, default=1,
                help="whether or not output frame should be displayed")
            args = vars(ap.parse_args())

            # load the COCO class labels our YOLO model was trained on
            labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
            LABELS = open(labelsPath).read().strip().split("\n")

            # derive the paths to the YOLO weights and model configuration
            weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
            configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])
            net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

            # check if we are going to use GPU
            if config.USE_GPU:
                # set CUDA as the preferable backend and target
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

            # determine only the *output* layer names that we need from YOLO
            ln = net.getLayerNames()
            ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            # initialize the video stream and pointer to output video file
            vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
            writer = None
            (grabbed, frame) = vs.read()
            frame = imutils.resize(frame, width=700)
            results = detect_people(frame, net, ln,
                personIdx=LABELS.index("person"))

            violate = set()
            if len(results) >= 2:
                centroids = np.array([r[2] for r in results])
                D = dist.cdist(centroids, centroids, metric="euclidean")

                # loop over the upper triangular of the distance matrix
                for i in range(0, D.shape[0]):
                    for j in range(i + 1, D.shape[1]):
                        if D[i, j] < config.MIN_DISTANCE:
                            violate.add(i)
                            violate.add(j)

            # loop over the results
            for (i, (prob, bbox, centroid)) in enumerate(results):
                (startX, startY, endX, endY) = bbox
                (cX, cY) = centroid
                color = (0, 255, 0)
                if i in violate:
                    color = (0, 0, 255)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.circle(frame, (cX, cY), 5, color, 1)

            # draw the total number of social distancing violations on the
            # output frame
            text = "Social Distancing Violations: {}".format(len(violate))
            cv2.putText(frame, text, (10, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
            if args["output"] != "" and writer is None:
                # initialize our video writer
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(args["output"], fourcc, 25,
                    (frame.shape[1], frame.shape[0]), True)
            if writer is not None:
                writer.write(frame)
            frame=cv2.resize(frame,None,fx=ds_factor,fy=ds_factor,
            interpolation=cv2.INTER_AREA)                    
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # encode OpenCV raw frame to jpg and displaying it
            ret, jpeg = cv2.imencode('.jpg', frame)

            return jpeg.tobytes()