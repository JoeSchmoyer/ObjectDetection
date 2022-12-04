import cv2
import numpy as np

class motionDetection:

    def __init__(self,link, out):
        self.streamLink = link
        self.stream = self.getStream()
        self.output = out
    def getStream(self):
        stream = cv2.VideoCapture(self.streamLink)
        assert stream is not None
        return stream

    def detectMotion(self):
        x_shape = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(self.output, four_cc, 20, (x_shape, y_shape))
        ret, frame1 = self.stream.read()
        ret, frame2 = self.stream.read()
        while self.stream.isOpened():
            #First look for differences between pixels
            dif = cv2.absdiff(frame1,frame2)
            #grayscale the difference
            gray = cv2.cvtColor(dif, cv2.COLOR_BGR2GRAY)
            #blur the grayscale
            blur = cv2.GaussianBlur(gray, (5,5),0)
            _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            dil = cv2.dilate(thresh, None, iterations=3)
            contours, _ = cv2.findContours(dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            #cv2.drawContours(frame1,contours,-1,(255,0,0),2)

            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                if cv2.contourArea(contour) < 50000:
                    continue
                cv2.rectangle(frame1, (x,y),(x+w, y+h), (0,255,0),2)
                out.write(frame1)

            cv2.imshow("LIVE",frame1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame1 = frame2
            ret, frame2 = self.stream.read()
        out.release()

a = motionDetection(0,"out.avi")
a.detectMotion()