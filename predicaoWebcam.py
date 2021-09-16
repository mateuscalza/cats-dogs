import cv2
import numpy as np
from PIL import Image
from tensorflow.keras import models

modelo = models.load_model('./modelo.h5')
video = cv2.VideoCapture(0)

while True:
    _, frame = video.read()
    imagem_pil = Image.fromarray(frame, 'RGB')
    imagem_pil = imagem_pil.resize((200, 200))
    imagem_array = np.array(imagem_pil) 

    cv2.imshow("Redimensionada", imagem_array)

    imagem_array = imagem_array.reshape(1, 200, 200, 3)
    imagem_array = imagem_array.astype('float32')
    imagem_array = imagem_array / 255.0

    resultado = modelo.predict(imagem_array)
    if resultado[0] <= 0.5:
      classificacao = 'Gato'
    else:
      classificacao = 'Cachorro'
    print(classificacao, '{0:.2f}'.format(resultado[0][0]))

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
