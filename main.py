from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator



#inicializa rede neural
classifier = Sequential()

#criação de camadas da rede neural convolucional

#camada convolucional
#qtd de detectores de características(filtros) - pode aumentar em maquinas mais potentes
#feature detector - 3x3
#formato entrada da imagem - 64x64x3 - o 3 é por ser colorida(rgb)
#função de ativação
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3), activation='relu'))

#camada de pooling - reduz complexidade da imagem sem perder as caracteristicas importantes
#mais comum é um filtro(matriz) de 2x2

classifier.add(MaxPooling2D(pool_size=(2,2)))

#PAra melhorar a qualidade da rede neural, vamos adicionar + 1 camada convolucional e + 1 camada de max pooling
classifier.add(Conv2D(32,(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#adicionar camada de flattening
#transforma matriz do mapa de caracteristicas em pool, em um vetor

classifier.add(Flatten())

#criar camada full connection - especificar camada de entrada e saída
#entrada - definir numero de neuronios - 128 neuronios (pode ser mais ou menos)
classifier.add(Dense(units=128, activation='relu'))

#saída - 1 neurônio, afinal é cachorro ou gato
#sigmoid traz o % de chance entre ser gato ou cachorro
classifier.add(Dense(units=1, activation='sigmoid'))

#compilar rede neural
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


#Tratamentos na imagem
#ajustar imagens para evitar super ajuste

#rescale = altera tamanho imagem
#shear_range = transformações geométricas - o parametro diz em quantos % das imagens isso será feito
#zoom_range = imagens em que será aplicado zoom
#horizontal_flip = girar imagem horizontalmente

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

#reajuste dos valores dos pixels entre 0 e 255
test_datagen = ImageDataGenerator(rescale=1./255)

#carregar dados de treino
#target size = input shape
#batch size = qtd de detectores

batch_size = 1

training_set = train_datagen.flow_from_directory('dataset/training_set', target_size=(64,64), batch_size=batch_size, class_mode = 'binary')

#criar variável de teste
test_set = test_datagen.flow_from_directory('dataset/test_set', target_size=(64,64), batch_size=batch_size, class_mode = 'binary')


#treinamento da rede neural
#steps per epoch - qts imagens serão utilizadas em cada epoch
#epoch = qtd de vezes que será repetido o treinamento
#validations steps = qtd de imagens de teste

steps_per_epoch = len(training_set)
validation_steps = len(test_set)

print(steps_per_epoch)
print(validation_steps)

classifier.fit_generator(training_set, steps_per_epoch=steps_per_epoch, epochs=25, validation_data=test_set, validation_steps=validation_steps)









