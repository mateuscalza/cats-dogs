from os import listdir
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import models
from gui import imagem_predicao_analise
import cv2

modelo = models.load_model('./modelo.h5')

pasta_predicao = './predicao'
pasta_predicao_saida = './predicaoSaida'


for file in listdir(pasta_predicao):
  arquivo_predicao = pasta_predicao + '/' + file

  if file.lower().endswith('.jpg') or file.lower().endswith('.jpeg'):
    arquivo_predicao_saida = pasta_predicao_saida + '/' + file

    imagem = load_img(arquivo_predicao, target_size=(200, 200))
    imagem = img_to_array(imagem)
    imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2BGR)
    
    resultado = imagem_predicao_analise(modelo, imagem)

    cv2.imwrite(arquivo_predicao_saida, resultado)
