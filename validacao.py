from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

modelo = load_model('./modelo.h5')

gerador_imagens = ImageDataGenerator(rescale=1.0/255.0,
                              shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
iterador_imagens_validacao = gerador_imagens.flow_from_directory('./validacao',
                                        class_mode='binary', batch_size=64, target_size=(200, 200))

_, acc = modelo.evaluate(iterador_imagens_validacao, steps=len(iterador_imagens_validacao), verbose=0)
print('Precisão = %.2f' % (acc * 100.0))
