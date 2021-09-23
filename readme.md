# Reconhecimento de imagem

Dataset Dogs vs Cats.

## Requisitos

- Python 3, recomendável 3.8
- OpenCV
- Tensorflow 2
- TF Explain
- GPU/CUDA, opcional para treinar o modelo 10x mais rápido

## Configuração

### Treinamento

Caso não queira usar o modelo pré-treinado ou queira fazer alterações siga os seguintes passos.

Faça o download do train.zip dataset (https://www.kaggle.com/c/dogs-vs-cats), crie a pasta "./original" na raiz deste projeto, e extraia nela todas as imagens da "./train". Na sequência rode o seguinte comando que irá organizar as imagens:

```bash
python dataset.py
```

Para executar o treinamento, basta rodar o seguinte comando:

```bash
python treinamento.py
```

### Validação

Caso queira validar o modelo treinado, o seguinte comando retornará a acurácia. Para isso, o dataset precisa estar configurado como no treinamento.

```bash
python validacao.py
```

### Predição de arquivos

Para fazer a predição usando o modelo em arquivos de imagem, crie as pastas "./predicao" e "./predicaoSaida" na raiz, e execute o comando. Os resultados ficarão disponíveis na pasta de saída.

```bash
python predicaoArquivos.py
```

### Predição com a webcam

Para fazer a predição em tempo real usando a webcam, execute o seguinte comando:

```bash
python main.py
```
