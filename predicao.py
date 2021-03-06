from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

imagem = load_img('./predicao/gato.jpg', target_size=(200, 200))
imagem = img_to_array(imagem)
imagem = imagem.reshape(1, 200, 200, 3)
imagem = imagem / 255

modelo = load_model('./modelo.h5')
resultado = modelo.predict(imagem)
print(resultado[0])
