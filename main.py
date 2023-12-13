import cv2 as cv
import numpy as np
import time
import pyttsx3
import pickle
from tensorflow.keras.preprocessing import image
import speech_recognition as sr
import webbrowser
from tensorflow.keras.preprocessing import image


engine = pyttsx3.init()
converter = pyttsx3.init()
voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
converter.setProperty('voice', voice_id)


render_model=pickle.load(open("notesmodel.pkl", 'rb'))

reference_distance = 46
person_diameter_inch = 17 
mobile_diameter_inch = 3.1 
book_diameter_inch=7.5
bottle_diameter_inch=2.9

# Object detection configuration
confidence_value = 0.35
nms_value = 0.35
type_fonts = cv.FONT_HERSHEY_SIMPLEX
COLORS = [(227, 95, 54), (20, 95, 54), (20, 41, 54), (20, 41, 194), (74, 145, 194), (255, 0, 0),
          (159, 112, 83), (90, 64, 112), (69, 50, 114), (111, 147, 114)]
text_color = (107, 68, 202)
rect_color = (0, 0, 0)

# Load YOLO model
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
class_names = []

# Read class names from file
with open("objects_in_universe.txt", "r") as f:
    class_names = [class_name.strip() for class_name in f.readlines()]

# Initialize YOLO detection model
model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

def draw_object_info(image, classid, score, box):
    color = COLORS[int(classid) % len(COLORS)]
    adjust_score = (100 - score) / 800
    unique_distort = abs(score - nms_value) / 100

    label = "%s : %f" % (class_names[int(classid)], score)
    label2 = "Adjust_score:%f" % (adjust_score)
    label3 = "Unique_distorted_score:%f" % (unique_distort)

    cv.rectangle(image, box, color, 2)
    cv.putText(image, label, (box[0], box[1] - 14), type_fonts, 0.5, color, 2)
    cv.putText(image, label2, (box[0], box[1] - 30), type_fonts, 0.5, color, 2)
    cv.putText(image, label3, (box[0], box[1] - 46), type_fonts, 0.5, color, 2)

def object_detector(image):
    classes, scores, boxes = model.detect(image, confidence_value, nms_value)
    data_list = []
    for classid, score, box in zip(classes, scores, boxes):
        draw_object_info(image, classid, score, box)
        if classid in [0, 39, 67]:
            data_list.append([class_names[classid], box[2], (box[0], box[1] - 2)])
    return data_list

def speakit(dist, entity):
    engine.say(f'A {entity} is {dist} meter away')
    engine.runAndWait()
    converter.runAndWait()

def focal_len_cmos_finder(reference_distance, diameter_inch, diameter_inch_in_rf):
    return (diameter_inch_in_rf * reference_distance) / diameter_inch

def dis_wrtreference_finder(focal, diameter_inch, width_in_frame):
    return (diameter_inch * focal) / width_in_frame

def process_reference_image(ref_image_path, ref_diameter_inch):
    ref_image = cv.imread(ref_image_path)
    ref_data = object_detector(ref_image)
    print(ref_data)
    ref_diameter_inch_in_rf = ref_data[0][1]
    focal_length = focal_len_cmos_finder(reference_distance, ref_diameter_inch, ref_diameter_inch_in_rf)
    return focal_length

def recognize_speech():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        query = recognizer.recognize_google(audio)
        print(f"User said: {query}")
        return query.lower()
    
    except sr.UnknownValueError:
        return ""
    
def speak(txt):  
    engine.say(txt) 
    engine.runAndWait() 
    converter.runAndWait() 

# Process reference images
focal_person = process_reference_image('ReferenceImages/person_ref.jpg', person_diameter_inch)
focal_mobile = process_reference_image('ReferenceImages/mobile_ref.png', mobile_diameter_inch)
focal_bottle = process_reference_image('ReferenceImages/bottle_ref.jpg', bottle_diameter_inch)

# Capture video from camera
cap = cv.VideoCapture(0)
interval_sec = 10
last_speak_time = time.time() - interval_sec

while True:
    ret, frame = cap.read()
    data = object_detector(frame)
    user_command = recognize_speech()
    spoken=user_command
    if(spoken=="check note"):
        currentframe = 0
        while currentframe<3:
            name = './data/frame' + str(currentframe) + '.jpg'
            print ('Creating...' + name) 
            cv.imwrite(name, frame) 
            currentframe += 1
            
        img = image.load_img("./data/frame1.jpg",target_size=(200,200))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        images=np.vstack([x])
        val=render_model.predict(images)
        max_value = np.argmax(val)
        print(val)
        if max_value==1:
            valu="500Rs Note"
            print(valu)
            speak(valu)
        elif max_value==2:
            valu="10Rs Note"
            print(valu)
            speak(valu)
        elif max_value==3:
            valu="20Rs Note"
            print(valu)
            speak(valu)
        elif max_value==4:
            valu="20Rs Note"
            print(valu)
            speak(valu)
        spoken=""
 
    for detected_obj in data:
        if detected_obj[0] == 'person':
            distance = dis_wrtreference_finder(focal_person, person_diameter_inch, detected_obj[1])
            x, y = detected_obj[2]
        elif detected_obj[0] == 'cell phone':
            distance = dis_wrtreference_finder(focal_mobile, mobile_diameter_inch, detected_obj[1])
            x, y = detected_obj[2]
        elif detected_obj[0] == 'bottle':
            distance = dis_wrtreference_finder(focal_mobile, mobile_diameter_inch, detected_obj[1])
            x, y = detected_obj[2]

        distance_text = round(distance * 0.0254, 2)
        cv.rectangle(frame, (x, y - 3), (x + 150, y + 23), rect_color, -1)
        cv.putText(frame, f'Distance: {distance_text}m', (x + 5, y + 13), type_fonts, 0.48, text_color, 2)
        entity = detected_obj[0]

        current_time = time.time()
        if current_time - last_speak_time >= interval_sec:
            speakit(distance_text, entity)
            last_speak_time = current_time

    cv.imshow('frame', frame)

    key = cv.waitKey(1)
    if key == ord('x'):
        break

cv.destroyAllWindows()
cap.release()
