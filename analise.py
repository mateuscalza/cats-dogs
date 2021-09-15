from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tf_explain.core.grad_cam import GradCAM

imagem = load_img('./predicao/cachorro2.jpg', target_size=(200, 200))
imagem = img_to_array(imagem)
imagem = imagem.reshape(1, 200, 200, 3)
imagem = imagem / 255

modelo = load_model('./modelo.h5')
resultado = modelo.predict(imagem)
if resultado[0][0] <= 0.5:
  acuracia = 1 - resultado[0][0]
  classificacao = 'Gato'
else:
  acuracia = resultado[0][0]
  classificacao = 'Cachorro'
print(classificacao, '{0:.2f}%'.format(acuracia * 100))

data = (imagem, None)
explainer = GradCAM()
grid = explainer.explain(data, modelo, class_index=0)
explainer.save(grid, ".", "grad_cam.png")
