import cv2
import numpy as np
from PIL import Image
from tensorflow.keras import models
from gui import imagem_predicao_analise

modelo = models.load_model('./modelo.h5')
video = cv2.VideoCapture(0)

while True:
    _, frame = video.read()
    imagem_pil = Image.fromarray(frame, 'RGB')
    imagem_pil = imagem_pil.resize((200, 200))
    imagem_array = np.array(imagem_pil) 

    resultado = imagem_predicao_analise(modelo, imagem_array)
    cv2.imshow("Resultado", resultado)

    key = cv2.waitKey(1)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()
