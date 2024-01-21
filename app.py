from flask import Flask, request, jsonify, Response
from functions import getMetrics, process_image_from_url, findMedicine
from scipy.signal import butter, lfilter, detrend
import cv2
import numpy as np
import pandas as pd
import json
from datetime import datetime
import cv2
from keras.preprocessing import image
from keras.applications.resnet import preprocess_input
import tensorflow as tf
from keras.models import load_model, model_from_json
import pickle


SKIN_CLASSES = {
    0: 'Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)',
    1: 'Basal Cell Carcinoma',
    2: 'Benign Keratosis',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic Nevi',
    6: 'Vascular skin lesion'

}


def butter_bandpass(lowcut, highcut, fs, order = 5):
    nyq = 0.5 * fs # fs는 sampling rate, fs/2를 나이퀴스트 주파수(nyq)라고 함
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype = 'band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order = 5):
    b, a = butter_bandpass(lowcut, highcut, fs, order = order)
    y = lfilter(b, a, data)
    return y



app = Flask(__name__)


# Load the Random Forest CLassifier model
FILENAME = 'models/diabetes-model.pkl'
CANCER_FILE = 'models/cancer-model.pkl'
classifier = pickle.load(open(FILENAME, "rb"))
rf = pickle.load(open(CANCER_FILE, 'rb'))

brain = "models/cnn-parameters-improvement-24-0.85.model"



@app.route('/')
def index():
    return 'Welcome to the health wizard!'


@app.route('/metrics')
def metrics():
    url = request.args.get('url')
    # result= getMetrics(url)

    cap = cv2.VideoCapture(url)

    # get frame rate    
    fps = cap.get(cv2.CAP_PROP_FPS)

    # get total number of frames
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # calculate duration of the video

    duration = frame_count/fps

    print('duration (seconds): ', duration)

    # 
    avg_red_2 = []
    avg_green_2 = []
    avg_blue_2 = []

    i = 0

    sec = 40

    # while(cap.isOpened()):
    while(cap.isOpened() and i < sec*fps):
        ret, frame = cap.read()

        if ret == True:
            blue,green,red = cv2.split(frame)
            # print(blue, green, red)
            # show the frame continuously until q is pressed
            # cv2.imshow('Frame',frame)
            # blue,green,red = cv2.split(frame)
            avg_red_2.append((i/30*5, red.mean()))
            avg_green_2.append((i/30*5, green.mean()))
            avg_blue_2.append((i/30*5, blue.mean()))

        

        i = i + 1

        # frames.append(frame)

    # release the video capture object
    cap.release()

    avg_red_2 = np.asarray(avg_red_2)
    avg_green_2 = np.asarray(avg_green_2)
    avg_blue_2 = np.asarray(avg_blue_2)

    avg_red_2_1 = avg_red_2.copy()
    avg_red_2_1[:,1] = detrend(avg_red_2_1[:,1])
    avg_green_2_1 = avg_green_2.copy()
    avg_green_2_1[:,1] = detrend(avg_green_2_1[:,1])
    avg_blue_2_1 = avg_blue_2.copy()
    avg_blue_2_1[:,1] = detrend(avg_blue_2_1[:,1])


    avg_red_2_1[:,1] = butter_bandpass_filter(avg_red_2_1[:,1], 0.1, 30, 100, 5)
    avg_green_2_1[:,1] = butter_bandpass_filter(avg_green_2_1[:,1], 0.1, 30, 100, 5)
    avg_blue_2_1[:,1] = butter_bandpass_filter(avg_blue_2_1[:,1], 0.1, 30, 100, 5)




    # fourier transform

    nfft = len(avg_red_2_1[:,0])
    fs = 6                            #sampling rate = 6
    df = fs/nfft
    k = np.arange(nfft)
    f = k*df

    nfft_half = int(nfft/2)
    f0 = f[range(nfft_half)]                  # only half size check for get hz. 
    y = np.fft.fft(avg_red_2_1[:,1])/nfft*2   # 증폭을 두 배
    y0 = y[range(nfft_half)]                  # one side. 
    amp = abs(y0)                             # 벡터(복소수) norm 측정. 신호 강도. 


    # heart rate measurement
    x = pd.DataFrame()
    x['Frequency'] = f0
    x['Amplitude'] = amp

    hr = x[x['Amplitude']== max(x['Amplitude'])]['Frequency'].values[0]*60

    print('Heart Rate : {:.4f}'.format(hr))


    I_Tsys = min(avg_red_2[:,1])
    I_Tdia = max(avg_red_2[:,1])
    I_S = max(avg_red_2[:,1]) - min(avg_red_2[:,1])

    SBP = -0.599*I_Tsys - 0.656*I_S + 249.942
    DBP = -0.212*I_Tdia - 0.251*I_S + 153.211

    print("SBP : {}".format(SBP))
    print("DBP : {}".format(DBP))
    
    # result = "https://res.cloudinary.com/" + url 
    return json.dumps({
        "heart_rate": hr,
        "systolic_blood_pressure": SBP,
        "diastolic_blood_pressure": DBP, \
        "graph" : {
            "red": {
                "x": avg_red_2[:,0].tolist(),
                "y": avg_red_2[:,1].tolist()
            },
            "green": {
                "x": avg_green_2[:,0].tolist(),
                "y": avg_green_2[:,1].tolist()
            },
            "blue": {
                "x": avg_blue_2[:,0].tolist(),
                "y": avg_blue_2[:,1].tolist()
            }
        }
    })



# @app.route("/eye"): 
# def eye() :

#     return json.dumps()

@app.route("/predict_tumor")
def predict_tumor():
    try: 
        brain_model = load_model(brain)
        
        return json.dumps({
            "status": True
        })
    except Exception: 
        return json.dumps({
            "status": True
        })


@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    if request.method == 'POST':
        try:
            preg = int(request.form['pregnancies'])
            glucose = int(request.form['glucose'])
            bp = int(request.form['bloodpressure'])
            st = int(request.form['skinthickness'])
            insulin = int(request.form['insulin'])
            bmi = float(request.form['bmi'])
            dpf = float(request.form['dpf'])
            age = int(request.form['age'])

            data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
            my_prediction = classifier.predict(data)

            return json.dumps({
                "data" : data,
                "prediction": my_prediction
            })
        except ValueError:
            return json.dumps({ 
                "status" : False, 
                "message" : "Error!"
            })


# eye cataract disease prediction
@app.route("/predict_eye")
def predict_image():
    url = request.args.get('url')

    model = load_model('models/vgg16.h5')
    # img = image.load_img("uploads/"+name, target_size=(224, 224))
    result = process_image_from_url(url, target_size=(224, 224))
    img =""
    if(result["status"]): 
        img = result["image"]

    else: 
        return {
            "status" : False, 
            "message" : "Error! No image."
        }
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    img_data = x
    # model.predict(img_data)
    a = np.argmax(model.predict(img_data), axis=1)
    if(a == 0):
        temp = "Cataract"
    else:
        temp = "Normal"
    return json.dumps({
        "prediction": {
            "confidence" : "", 
            "result" : temp
        }
    })



# detect skin disease
@app.route('/predict_skin')
def detect():
    url = request.args.get("url")
    try:
        j_file = open('models/model.json', 'r')
        loaded_json_model = j_file.read()
        j_file.close()
        model = model_from_json(loaded_json_model)
        model.load_weights('models/model.h5')
        result = process_image_from_url(url, target_size=(224, 224))
        if(result["status"]): 
            img = result["image"]

        else: 
            return {
                "status" : False, 
                "message" : "Error! No image."
            }
        img = np.array(img)
        img = img.reshape((1, 224, 224, 3))
        img = img/255
        prediction = model.predict(img)
        pred = np.argmax(prediction)
        disease = SKIN_CLASSES[pred]
        accuracy = prediction[0][pred]
        accuracy = round(accuracy*100, 2)
        medicine=findMedicine(pred)

        json_data = json.dumps({
            "status" : True,
            "message" : "Success",
            "detected": False if pred == 2 else True,
            "disease": disease,
            "accuracy": accuracy,
            "medicine" : medicine,
        })


        
        return Response(json_data, content_type='application/json')

    except Exception:
        return Response(json.dumps({
            "status" : False, 
            "message" : "Error!"
        }), content_type="Application/json")


if __name__ == '__main__':
    app.run(debug=True)
