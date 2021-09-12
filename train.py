from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

modelo = Sequential()
modelo.add(Conv2D(32, (3, 3), activation='relu',
          kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
modelo.add(MaxPooling2D((2, 2)))
modelo.add(Conv2D(64, (3, 3), activation='relu',
          kernel_initializer='he_uniform', padding='same'))
modelo.add(MaxPooling2D((2, 2)))
modelo.add(Conv2D(128, (3, 3), activation='relu',
          kernel_initializer='he_uniform', padding='same'))
modelo.add(MaxPooling2D((2, 2)))
modelo.add(Flatten())
modelo.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
modelo.add(Dense(1, activation='sigmoid'))
modelo.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

gerador_imagens = ImageDataGenerator(rescale=1.0/255.0,
                              shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
iterador_imagens = gerador_imagens.flow_from_directory('./dataset/',
                                        class_mode='binary', batch_size=64, target_size=(200, 200))
resultados = modelo.fit(iterador_imagens, steps_per_epoch=len(iterador_imagens),
                    epochs=20, verbose=2)

modelo.save('model.h5')

pyplot.subplot(211)
pyplot.title('Perda em Entropia Cruzada')
pyplot.plot(resultados.history['loss'], color='blue', label='train')
pyplot.plot(resultados.history['val_loss'], color='orange', label='test')
pyplot.subplot(212)
pyplot.title('Precisão da classificação')
pyplot.plot(resultados.history['accuracy'], color='blue', label='train')
pyplot.plot(resultados.history['val_accuracy'], color='orange', label='test')
pyplot.savefig('resultados.png')
pyplot.tight_layout()
pyplot.show()
pyplot.close()
