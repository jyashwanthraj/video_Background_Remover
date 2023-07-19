from flask import Flask, render_template, Response
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

app = Flask(__name__)

segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()

listImg = os.listdir("Images")
print(listImg)
imgList = []
for imgPath in listImg:
    img = cv2.imread(f'Images/{imgPath}')
    imgList.append(img)
print(len(imgList))

indexImg = 0

# Generator function to capture frames
def gen_frames():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)

    while True:
        success, img = cap.read()
        if not success:
            break

        imgOut = segmentor.removeBG(img, imgList[indexImg], threshold=0.8)

        imgStacked = cvzone.stackImages([img, imgOut], 2, 1)
        _, imgStacked = fpsReader.update(imgStacked, color=(0, 0, 255))

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', imgStacked)
        frame = buffer.tobytes()

        # Yield the frame in the byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
