# Detecção de Pneumonia com CNN

Este projeto usa uma rede neural convolucional (CNN) para detectar pneumonia em imagens de raio-X do tórax. O modelo foi treinado com o TensorFlow e Keras para classificar imagens como normais ou com pneumonia.

## Estrutura do Projeto

- **`train_dir`**: Diretório contendo imagens para treinamento.
- **`val_dir`**: Diretório contendo imagens para validação.
- **`test_dir`**: Diretório contendo imagens para teste.

## Dependências

Certifique-se de ter as seguintes bibliotecas instaladas:

- TensorFlow
- NumPy
- Matplotlib
- PIL (Pillow)

Você pode instalar as dependências necessárias usando o `pip`:

```bash
pip install tensorflow numpy matplotlib pillow
```

## Código

### Pré-processamento dos Dados

Os dados são carregados e pré-processados usando o `ImageDataGenerator` do Keras:

```python
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)
```

### Construção do Modelo

Um modelo CNN é construído com camadas de convolução e pooling, seguido por camadas densas:

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

### Treinamento e Avaliação

O modelo é treinado e avaliado usando as funções `fit` e `evaluate`:

```python
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=20,
    validation_data=val_generator,
    validation_steps=50
)

test_loss, test_acc = model.evaluate(test_generator, steps=50)
print(f'Test accuracy: {test_acc}')
```
- **`pneumonia_detection_model.h5`**: Modelo treinado salvo após o treinamento.

### Predição de Novas Imagens

Função para carregar e prever se uma nova imagem contém pneumonia:

```python
def predict_image(model, img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        print('Pneumonia Detectada')
    else:
        print('Pneumonia Não Detectada')
```

## Contribuições

Sinta-se à vontade para contribuir com melhorias ou correções. Crie um pull request ou abra uma issue para discutir mudanças.
