import cv2
import numpy as np
import torch
from time import time
class objectAndMovementDetection:
    def __init__(self, inputURL, outputFile):
        self.inputURL = inputURL
        self.output = outputFile
        self.model = self.loadmodel()
        self.model.conf = 0.4  # set inference threshold at 0.3
        self.model.iou = 0.3  # set inference IOU threshold at 0.3
        self.model.classes = [0] #Detect only the person class
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def getStream(self):
        stream = cv2.VideoCapture(self.inputURL)
        assert stream is not None
        return stream

    def loadmodel(self):
        #Using YOLO v5 Pretrained model
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        results = self.model([frame])
        labels, cord = results.xyxyn[0][:, -1].to('cpu').numpy(), results.xyxyn[0][:, :-1].to('cpu').numpy()
        return labels, cord

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            bgr = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 1)
            label = f"{int(row[4] * 100)}"
            cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(frame, f"Total Targets: {n}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def call(self):
        stream = self.getStream()
        assert stream.isOpened()
        x_shape = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(self.output, cv2.VideoWriter_fourcc(*"MJPG"), 20, (x_shape, y_shape))
        while True:
            ret, frame = stream.read()
            if not ret:
                break
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            out.write(frame)
        stream.release()

link = "out.avi"
output_file = "test.avi"
a = objectAndMovementDetection(link, output_file)
a.call()