from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
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
iterador_imagens = gerador_imagens.flow_from_directory('./treinamento',
                                        class_mode='binary', batch_size=64, target_size=(200, 200))
iterador_imagens_validacao = gerador_imagens.flow_from_directory('./validacao',
                                        class_mode='binary', batch_size=64, target_size=(200, 200))
resultados = modelo.fit(iterador_imagens, steps_per_epoch=len(iterador_imagens),
                    validation_data=iterador_imagens_validacao, validation_steps=len(iterador_imagens_validacao),
                    epochs=20, verbose=2)

modelo.save('modelo.h5')

_, acc = modelo.evaluate(iterador_imagens_validacao, steps=len(iterador_imagens_validacao), verbose=0)
print('Precisão = %.2f' % (acc * 100.0))
