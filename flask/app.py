import os
import fileinput
import sys

from flask import Flask, render_template, redirect, Response, request
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import pickle
import sqlite3, pyttsx3
from keras.models import load_model
from uni import join_jamos
import tensorflow.compat.v1 as tf

UPLOAD_FOLDER = '/static'
ALLOWED_EXTENSTION = set(['txt', 'png', 'jpg', 'jpeg'])
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = load_model('cnn_model.h5',
                   custom_objects=None,
                   compile = False
                   )
graph = tf.get_default_graph()
app = Flask(__name__, static_url_path="/static")


@app.route("/")
def main():
    return render_template("main.html")

def gen(camera):
    while True:
        if camera.stopped:
            break
        frame = camera.read()
        ret, jpeg = cv2.imencode('.jpg', frame)

        if jpeg is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        else:
               print("frame is none")

'''
@app.route("/reset")
def reset():
    with open("static/output.txt", "w"):
        pass
    return "reset"
'''

@app.route("/guide")
def guide():
	return render_template("guide.html")


@app.route('/output')
def read_txt():
    f = open('static/output.txt', 'r', encoding='UTF-8')
    return "</br>".join(f.readlines())

@app.route('/code')
def code():
	return render_template("code.html");

@app.route('/color')
def get_hand_hist():
    def build_squares(img):
        x, y, w, h = 420, 140, 10, 10
        d = 10
        imgCrop = None
        crop = None
        for i in range(10):
            for j in range(5):
                if np.any(imgCrop == None):
                    imgCrop = img[y:y + h, x:x + w]
                else:
                    imgCrop = np.hstack((imgCrop, img[y:y + h, x:x + w]))
                # print(imgCrop.shape)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                x += w + d
            if np.any(crop == None):
                crop = imgCrop
            else:
                crop = np.vstack((crop, imgCrop))
            imgCrop = None
            x = 420
            y += h + d
        return crop

    cam = cv2.VideoCapture(1)
    if cam.read()[0] == False:
        cam = cv2.VideoCapture(0)
    x, y, w, h = 300, 100, 300, 300
    flagPressedC, flagPressedS = False, False
    imgCrop = None
    while True:
        img = cam.read()[1]
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
            flagPressedC = True
            hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        elif keypress == ord('s'):
            flagPressedS = True
            break
        if flagPressedC:
            dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
            dst1 = dst.copy()
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            cv2.filter2D(dst, -1, disc, dst)
            blur = cv2.GaussianBlur(dst, (11, 11), 0)
            blur = cv2.medianBlur(blur, 15)
            ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = cv2.merge((thresh, thresh, thresh))
            # cv2.imshow("res", res)
            cv2.imshow("output", thresh)
        if not flagPressedS:
            imgCrop = build_squares(img)
        # cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.imshow("Set hand color", img)
    cam.release()
    cv2.destroyAllWindows()
    with open("hist", "wb") as f:
        pickle.dump(hist, f)

    return render_template("main.html")


@app.route('/predict', methods = ['GET', 'POST'])
def prediction():
    def get_hand_hist():
        with open("hist", "rb") as f:
            hist = pickle.load(f)
        return hist

    image_x, image_y = (50,50)
    def keras_process_image(img):  # test할 이미지를 배열 값으로 반환
        img = cv2.resize(img, (image_x, image_y))
        img = np.array(img, dtype=np.float32)
        img = np.reshape(img, (1, image_x, image_y, 1))
        return img

    def keras_predict(model, image):  # 배열값을 가지고 이미지 예측
        processed = keras_process_image(image)
        with graph.as_default():
            pred_probab = model.predict(processed)[0]
        pred_class = list(pred_probab).index(max(pred_probab))
        return max(pred_probab), pred_class

    def get_pred_text_from_db(pred_class):  # 데이터 베이스에서 이름 찾기
        conn = sqlite3.connect("gesture_db.db")
        cmd = "SELECT g_name FROM gesture WHERE g_id=" + str(pred_class)
        cursor = conn.execute(cmd)
        for row in cursor:
            return row[0]

    def get_pred_from_contour(contour, thresh):  # 이미지 판정
        x1, y1, w1, h1 = cv2.boundingRect(contour)
        save_img = thresh[y1:y1 + h1, x1:x1 + w1]
        name = ""
        if w1 > h1:
            save_img = cv2.copyMakeBorder(save_img, int((w1 - h1) / 2), int((w1 - h1) / 2), 0, 0, cv2.BORDER_CONSTANT,
                                          (0, 0, 0))
        elif h1 > w1:
            save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1 - w1) / 2), int((h1 - w1) / 2), cv2.BORDER_CONSTANT,
                                          (0, 0, 0))
        pred_probab, pred_class = keras_predict(model, save_img)
        if pred_probab * 100 > 70:
            name = get_pred_text_from_db(pred_class)
        return name

    keras_predict(model, np.zeros((50, 50), dtype=np.uint8))

    hist = get_hand_hist()  # 저장된 손의 히스토그램
    x, y, w, h = 300, 100, 300, 300

    global cam
    cam = cv2.VideoCapture(1)
    if cam.read()[0] == False:
        cam = cv2.VideoCapture(0)
    name = ""
    line = ""

    path_name = "static/output.txt"

    T = open(path_name, 'wt', encoding='utf-8')
    T.write("")
    T.close()
    end = 0
    while True:
        img = cam.read()[1]  # 이미지 컨투어링 시키는 함수
        img = cv2.resize(img, (640, 480))
        img = cv2.flip(img, 1)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # bgr색공간의 이미지를 hsv 색공간 이미지로 변환
        dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)  # 배경 투영, 원하는 객체 영역 검출
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))  # 윤곽선 보정에 필요함
        cv2.filter2D(dst, -1, disc, dst)  # 윤곽선 부드러워지도록 필터 끼워줌
        blur = cv2.GaussianBlur(dst, (11, 11), 0)
        blur = cv2.medianBlur(blur, 15)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # 이미지 문턱값 지정해주고 이미지 픽셀값이 문턱값보다 크면 255 값을 갖고 문턱보다 작으면 0을 갖도록(픽셀분류)
        thresh = cv2.merge((thresh, thresh, thresh))  # 1채널의 바이너리 이미지를 3채널 이미지로 변환
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)  # 흑백으로 색변환
        thresh = thresh[y:y + h, x:x + w]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]  # 만들어진 윤곽선 값
        if len(contours) > 0:  # 컨투어링 값이 생기면
            end = 0
            contour = max(contours, key=cv2.contourArea)  # 최대값 뽑아서
            keypress = cv2.waitKey(1)
            if cv2.contourArea(contour) > 10000 and keypress == ord('c'):  # C 눌렀을때
                name = get_pred_from_contour(contour, thresh)  # 데이터베이스에서 컨투어링, 임계값에 일치하는 이름 들고옴
                print("name : " + name)
                R = open(path_name, 'rt', encoding='utf-8')
                line = ""
                while True:
                    line = line + R.readline()
                    if not R.readline(): break
                R.close()
                T = open(path_name, 'wt', encoding='utf-8')
                name = line + name
                T.write(name)
                T.close()
                sentence = join_jamos(name)
                print(sentence)
            elif cv2.contourArea(contour) < 1000 and keypress == ord('c'):
                name = " "
                #print("not found")
            elif keypress == ord('f'):
                end = 1
            elif keypress == ord('e'):
                R = open(path_name, 'rt', encoding='utf-8')
                line = ""
                while True:
                    line = line + R.readline()
                    if not R.readline(): break
                R.close()
                newline = ""
                for i in range(0, len(line) - 1):
                    newline = newline + line[i]
                T = open(path_name, 'wt', encoding='utf-8')
                T.write(newline)
                line = newline
                print("", line)
                T.close()

        R = open(path_name, 'r', encoding='utf-8')
        new_line = ""
        while True:
            new_line = new_line + R.readline()
            if not R.readline(): break
        T = open(path_name, 'wt', encoding='utf-8')
        sentence = join_jamos(new_line)
        T.write(sentence)

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # cv2.putText(img, str(line), (30, 400), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255))
        cv2.imshow("Capturing gesture", img)
        cv2.imshow("output", thresh)
        if end == 1:
            cam.release()
            cv2.destroyAllWindows()
            break