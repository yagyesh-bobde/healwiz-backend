from io import BytesIO
from scipy.signal import butter, lfilter, detrend
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import requests
import base64

def process_image_from_url(url, target_size=(300, 300)):
    try:
        
        image_data = base64.b64decode(url.split("Base64 URL: ")[1])
        
        # Open the image using Pillow (PIL)
        image = Image.open(BytesIO(image_data))

        # # Open the image using Pillow (PIL)
        # image = Image.open(BytesIO(response.content))

        # Resize the image to the target size
        resized_image = image.resize(target_size)

        print(resized_image)
        return {
            "status" : True, 
            "image" : resized_image,
            "message" : "Success!"
        }

    except Exception as e:
        print(f"Error processing image: {e}")
        return {
            "status" : False, 
            "message" : "Error!"
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



def getMetrics(urL) : 
    cap = cv2.VideoCapture(url)

    # get frame rate    
    fps = cap.get(cv2.CAP_PROP_FPS)

    # get total number of frames
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # calculate duration of the video

    duration = frame_count/fps

    print('duration (seconds): ', duration)


    # read all frames 
    frames = []

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



    avg_red_2_1[:,1] = butter_bandpass_filter(avg_red_2_1[:,1], 0.1, 30, 100, 5)

    # fig, ax = plt.subplots(figsize=(15,5))
    # plt.plot(avg_red_2_1[:,0] ,avg_red_2_1[:,1],color ='red')
    # plt.xlabel('Time(s)')
    # plt.ylabel('Mean pixel intensity')
    # plt.grid(True)
    # plt.show()


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




# find medicine
def findMedicine(pred):
    if pred == 0:
        return "fluorouracil"
    elif pred == 1:
        return "Aldara"
    elif pred == 2:
        return "Prescription Hydrogen Peroxide"
    elif pred == 3:
        return "fluorouracil"
    elif pred == 4:
        return "fluorouracil (5-FU):"
    elif pred == 5:
        return "fluorouracil"
    elif pred == 6:
        return "fluorouracil"

if __name__ == '__main__':
    url = 'https://res.cloudinary.com/dxgfv3aco/video/upload/v1705178248/ikqs7o9owqibq5ey2zhy.mp4'
    getMetrics(url)