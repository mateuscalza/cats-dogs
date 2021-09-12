from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

imagem = load_img('./evaluate/cat.jpeg', target_size=(200, 200))
imagem = img_to_array(imagem)
imagem = imagem.reshape(1, 200, 200, 3)
imagem = imagem / 255

modelo = load_model('./model.h5')
resultado = modelo.predict(imagem)
print(resultado[0])

# curl -X POST -H "Content-Type: application/json" -d '{"value1":"Cão"}' https://maker.ifttt.com/trigger/cnn/with/key/bR-SPDtEx7xff4ATykxN05bD1VrBz-ULUzuyyIV0Vyl
