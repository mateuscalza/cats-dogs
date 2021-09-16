from tf_explain.core.grad_cam import GradCAM
import numpy as np
import cv2

fonte = cv2.FONT_HERSHEY_SIMPLEX
fonte_escala = 0.8
texto_posicao = (10, 230)
texto_cor = (255, 0, 0)
texto_largura = 1
rodape = np.zeros((50,400,3), np.uint8)

def imagem_predicao_analise(modelo, imagem):
  imagem_array = imagem.reshape(1, 200, 200, 3)
  imagem_array = imagem_array.astype('float32')
  imagem_array = imagem_array / 255.0

  resultado = modelo.predict(imagem_array)
  if resultado[0][0] <= 0.5:
    acuracia = 1 - resultado[0][0]
    classificacao = 'Gato'
  else:
    acuracia = resultado[0][0]
    classificacao = 'Cachorro'
  texto = classificacao + ' {0:.2f}%'.format(acuracia * 100)

  dados_explainer = (imagem_array, None)
  explainer = GradCAM()
  imagem_analise = explainer.explain(dados_explainer, modelo, class_index=0)
  imagem_analise = cv2.cvtColor(imagem_analise, cv2.COLOR_RGB2BGR)

  resultado = np.hstack((imagem, imagem_analise))
  
  resultado = np.vstack((resultado, rodape))
  resultado = cv2.putText(resultado, texto, texto_posicao, fonte, 
                    fonte_escala, texto_cor, texto_largura, cv2.LINE_AA)

  return resultado
