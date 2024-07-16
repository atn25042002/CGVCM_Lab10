import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st
from texify.inference import batch_inference
from texify.model.model import load_model
from texify.model.processor import load_processor 
 
st.set_page_config(layout="wide")
st.image('MathCamera.png')
 
col1, col2 = st.columns([3,2])
with col1:
    run = st.checkbox('Ejecutar', value=True)
    FRAME_WINDOW = st.image([])
 
with col2:
    st.title("Respuesta")
    output_text_area = st.subheader("")
 
# Initialize the webcam to capture video
# The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
 
# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)
 
 
def getHandInfo(img):
    # Find hands in the current frame
    # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
    # The 'flipType' parameter flips the image, making it easier for some detections
    hands, img = detector.findHands(img, draw=False, flipType=True)
 
    # Check if any hands are detected
    if hands:
        # Information for the first hand detected
        hand = hands[0]  # Get the first hand detected
        lmList = hand["lmList"]  # List of 21 landmarks for the first hand
        # Count the number of fingers up for the first hand
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    else:
        return None
 
def draw(info,prev_pos,canvas):
    fingers, lmList = info
    current_pos= None
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None: prev_pos = current_pos
        cv2.line(canvas,current_pos,prev_pos,(255,0,255),10)
    elif fingers == [1, 0, 0, 0, 0]:
        canvas = np.zeros_like(img)
 
    return current_pos, canvas
 
def sendToAI(canvas,fingers):
    if fingers == [1,1,1,1,1]:
        pil_image = Image.fromarray(canvas)
        img = np.array(pil_image)
        response= procesar(img)
        return str(response)


model = load_model()
processor = load_processor()

def procesar(img):
    # Crear una máscara para el fondo negro (típicamente (0, 0, 0) en BGR)
    black_mask = cv2.inRange(img, (0, 0, 0), (0, 0, 0))
    
    # Crear una máscara para el color magenta (típicamente (255, 0, 255) en BGR)
    magenta_mask = cv2.inRange(img, (255, 0, 255), (255, 0, 255))
    
    # Crear una copia de la imagen para modificar
    new_image = img.copy()
    
    # Cambiar los píxeles negros a blancos en la nueva imagen
    new_image[black_mask == 255] = [255, 255, 255]  # Blanco
    
    # Cambiar los píxeles magenta a negros en la nueva imagen
    new_image[magenta_mask == 255] = [0, 0, 0]  # Negro

    cv2.imwrite('modificado.jpg', new_image)
    pil_image = Image.fromarray(new_image)
    results = batch_inference([pil_image], model, processor)
    print(results)
    return results

prev_pos= None
canvas=None
image_combined = None
output_text= ""
# Continuously get frames from the webcam
while True:
    # Capture each frame from the webcam
    # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
    success, img = cap.read()
    img = cv2.flip(img, 1)
 
    if canvas is None:
        canvas = np.zeros_like(img)
 
 
    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        prev_pos,canvas = draw(info, prev_pos,canvas)
        output_text = sendToAI(canvas,fingers)
 
    image_combined= cv2.addWeighted(img,0.7,canvas,0.3,0)
    FRAME_WINDOW.image(image_combined,channels="BGR")
 
    if output_text:
        output_text_area.text(output_text)
 
    # # Display the image in a window
    # cv2.imshow("Image", img)
    # cv2.imshow("Canvas", canvas)
    # cv2.imshow("image_combined", image_combined)
 
 
    # Keep the window open and update it for each frame; wait for 1 millisecond between frames
    cv2.waitKey(1)