import numpy as np
import cv2
from PIL import Image
from time import time

import tflite_runtime.interpreter as tflite

#Importe les bibliothèques numpy et cv2 et la classe Image de la bibliothèque PIL. Importe également la fonction time et le module tflite_runtime.interpreter.



def processImg(img):
    img_tensor = np.array(img).astype(np.float32)                # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)              # (1, height, width, channels), ajouter une dimension parce que le mode expect shape: (batch_size, height, width, channels)
    # img_tensor /= 255.                                           # expects valeurs entre  [0, 1]

    return img_tensor

     #Définit une fonction qui prend en entrée une image et la traite de façon à la préparer pour être passée au modèle TFLite. La fonction convertit l'image en tenseur numpy (un tableau multidimensionnel), ajoute une dimension au tenseur pour le mettre au format attendu par le modèle (un tenseur de dimensions [batch_size, height, width, channels]), et renvoie le tenseur.


# Sigmoid function pour transformation de model output
def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

#Définit une fonction qui calcule la fonction sigmoïde d'un nombre ou d'un tableau de nombres.


interpreter = tflite.Interpreter(model_path="mobilenetv2_BSD.tflite")
interpreter.allocate_tensors()

#Charge le modèle TFLite et prépare les tenseurs nécessaires pour l'exécuter.

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#Récupère les détails sur les tenseurs d'entrée et de sortie du modèle.


cams = [cv2.VideoCapture(0), cv2.VideoCapture(2)] # Check v4l2-ctl --list-devices for cam ids

for i, cam in enumerate(cams):
    if not cam.isOpened():
        print(f"Couldn't open camera {i}")
        exit()

#Ouvre deux cameras vidéo à l'aide de la fonction cv2.VideoCapture et les stocke dans une liste.

font = cv2.FONT_HERSHEY_SIMPLEX

def getPredictionFromRetAndFrame(ret, frame, winName):
    if not ret:
        print("Can't recieve frame, exiting...")
        exit()
    
    timer = time()
    #démarre un chronomètre pour mesurer le temps de traitement et de prédiction.



    LINE_THICKNESS = 10
    LINE_COLOUR    = (0, 0, 255)
    ALPHA          = 0.5
    
    # Top layer
    cv2.line(frame,   (0,     678),     (1280, 450),    LINE_COLOUR,    thickness=LINE_THICKNESS)
    cv2.line(frame,   (1280,  450),     (1750, 450),    LINE_COLOUR,    thickness=LINE_THICKNESS)

    # Farther side layer
    cv2.line(frame,   (1280,  450),     (1280, 656),    LINE_COLOUR,    thickness=LINE_THICKNESS)
    cv2.line(frame,   (1750,  450),     (1739, 627),    LINE_COLOUR,    thickness=LINE_THICKNESS)

    # Bottom layer
    cv2.line(frame,   (1739,  627),     (1254, 1080),   LINE_COLOUR,    thickness=LINE_THICKNESS)
    cv2.line(frame,   (1280,  656),     (0,    1080),   LINE_COLOUR,    thickness=LINE_THICKNESS)
    cv2.line(frame,   (1280,  656),     (1739,  627),   LINE_COLOUR,    thickness=LINE_THICKNESS)
    
    # Apply overlay with 50% transparency to original frame, takes 0.5s
    # cv2.addWeighted(overlay, ALPHA, output, 1 - ALPHA, 0, output)
    
    # Crop frame and resize
    cropped = frame[360:1080, 0:1920]
    resized = cv2.resize(cropped, (160, 160))

    # Preprocess image and predict
    interpreter.set_tensor(input_details[0]['index'], processImg(resized))
    interpreter.invoke()

    #utilisent l'interpréteur TFLite pour préparer les données d'entrée, et lancer la prédiction sur l'image vidéo traitée.


 
    pred_raw = interpreter.get_tensor(output_details[0]['index'])
    pred_sig = sigmoid(pred_raw)
    # récupèrent les résultats de la prédiction et les convertissent en niveau de confiance en utilisant la fonction sigmoïde.
    pred = np.where(pred_sig < 0.5, 0, 1)
   # convertit les niveaux de confiance en étiquette de classe (0 ou 1) en utilisant un seuil de 0.5.
    timer = time() - timer


    readable_val = winName if pred[0][0] == 0 else ""
    print(readable_val)
    print("----------------------\n\n")

    print()


cameraNames = ["Left", "Right"]

# Les images vidéo de la webcam sont capturées, traitées et affichées avec la prédiction du modèle dessinée dessus.
while True:
    for i in range(len(cams)):
        ret, frame = cams[i].read()
        getPredictionFromRetAndFrame(ret, frame, cameraNames[i])

    # Quit on q
    if cv2.waitKey(1) == ord('q'):
        break