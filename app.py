import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import uuid

app = Flask(__name__)

# Diretório para salvar imagens enviadas via upload
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Diretórios com imagens de treino e teste (conforme seu código original)
train_dir = 'dataset/Training'
test_dir = 'dataset/Testing'

IMG_SIZE = 300  # Tamanho para redimensionar as imagens
orb = cv2.ORB_create()  # Inicializa o detector ORB para extração de características

# Função que implementa o pré-processamento e a extração de características ORB
def processar_imagens(diretorio):
    X = []  # Lista para armazenar vetores de características
    y = []  # Lista para armazenar labels/classes das imagens
    classes = os.listdir(diretorio)  # Obtém as pastas/classes no diretório

    for classe in classes:
        caminho_classe = os.path.join(diretorio, classe)
        for imagem_nome in os.listdir(caminho_classe):
            caminho_img = os.path.join(caminho_classe, imagem_nome)
            imagem = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)  # Lê a imagem em escala de cinza
            if imagem is None:
                continue  # Ignora caso a imagem não possa ser lida

            # Pré-processamento da imagem conforme seu código original:
            imagem = cv2.resize(imagem, (IMG_SIZE, IMG_SIZE))  # Redimensiona
            imagem = cv2.GaussianBlur(imagem, (3, 3), 0)  # Suaviza com blur gaussiano

            # Sharpening - realça os detalhes da imagem
            kernel_sharp = np.array([[0, -1, 0],
                                     [-1, 5, -1],
                                     [0, -1, 0]])
            imagem = cv2.filter2D(imagem, -1, kernel_sharp)

            imagem = cv2.equalizeHist(imagem)  # Equaliza histograma para melhorar contraste

            # Rotaciona a imagem em 15 graus (data augmentation)
            centro = (IMG_SIZE // 2, IMG_SIZE // 2)
            matriz_rotacao = cv2.getRotationMatrix2D(centro, 15, 1.0)
            imagem = cv2.warpAffine(imagem, matriz_rotacao, (IMG_SIZE, IMG_SIZE))

            # Segmentação com threshold e bordas com Canny
            _, imagem_thresh = cv2.threshold(imagem, 120, 255, cv2.THRESH_BINARY)
            imagem_canny = cv2.Canny(imagem_thresh, 70, 150)

            # Extração de características ORB
            kp, des = orb.detectAndCompute(imagem_canny, None)
            vetor = np.mean(des, axis=0) if des is not None else np.zeros(32)  # Vetor fixo de características

            X.append(vetor)  # Adiciona o vetor na lista de entradas
            y.append(classe)  # Adiciona a classe correspondente

    return np.array(X), np.array(y)  # Retorna arrays numpy para treinamento

# --- Treinamento do modelo ao iniciar a aplicação Flask ---

# Processa as imagens dos diretórios de treino e teste
X_train, y_train = processar_imagens(train_dir)
X_test, y_test = processar_imagens(test_dir)

# Normaliza os vetores de características entre 0 e 1 para melhorar o treinamento
X_train = X_train / 255.0
X_test = X_test / 255.0

# Codificação dos labels/classes para valores numéricos (0, 1, 2, 3)
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_test_enc = encoder.transform(y_test)

# One-hot encoding para saída da rede neural (necessário para softmax)
y_train_cat = to_categorical(y_train_enc)
y_test_cat = to_categorical(y_test_enc)

# Criação do modelo sequencial Keras conforme seu main.py
modelo = Sequential()
modelo.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))  # Camada densa com 64 neurônios ReLU
modelo.add(Dense(32, activation='relu'))  # Segunda camada densa com 32 neurônios ReLU
modelo.add(Dense(4, activation='softmax'))  # Camada de saída com 4 neurônios para 4 classes, softmax para probabilidades

# Compilação do modelo com otimizador Adam e função de perda categórica crossentropy
modelo.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento do modelo com 20 épocas e batch size 8
history = modelo.fit(X_train, y_train_cat, epochs=20, batch_size=8, validation_split=0.2)

# Avaliação do modelo no conjunto de teste
loss, acc = modelo.evaluate(X_test, y_test_cat)
print(f"Acurácia final no conjunto de teste: {acc:.2f}")

# Predição para avaliação detalhada
y_pred = modelo.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Exibe relatório de classificação com métricas como precisão, recall e f1-score
print(classification_report(
    y_test_enc, 
    y_pred_classes, 
    target_names=encoder.classes_, 
    labels=np.arange(len(encoder.classes_))
))

# (Opcional) Plotagem dos gráficos de treinamento - comentado pois não roda na web
"""
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treinamento')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia por Época')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda por Época')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
"""

# Salva o modelo treinado para uso posterior
modelo.save("modelo_tumor.keras")

# Salva as classes do encoder para uso posterior
np.save("classes_labels.npy", encoder.classes_)


# --- Função para pré-processar uma imagem e extrair características para predição ---
def prever(caminho_img):
    # Lê a imagem de entrada em escala de cinza
    imagem = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)
    # Aplica os mesmos passos de pré-processamento usados no treinamento
    imagem = cv2.resize(imagem, (IMG_SIZE, IMG_SIZE))
    imagem = cv2.GaussianBlur(imagem, (3, 3), 0)
    kernel_sharp = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])
    imagem = cv2.filter2D(imagem, -1, kernel_sharp)
    imagem = cv2.equalizeHist(imagem)
    centro = (IMG_SIZE // 2, IMG_SIZE // 2)
    matriz_rotacao = cv2.getRotationMatrix2D(centro, 15, 1.0)
    imagem = cv2.warpAffine(imagem, matriz_rotacao, (IMG_SIZE, IMG_SIZE))
    _, imagem_thresh = cv2.threshold(imagem, 120, 255, cv2.THRESH_BINARY)
    imagem_canny = cv2.Canny(imagem_thresh, 70, 150)

    # Extrai características ORB da imagem processada
    kp, des = orb.detectAndCompute(imagem_canny, None)
    vetor = np.mean(des, axis=0) if des is not None else np.zeros(32)

    # Normaliza vetor e adapta forma para predição no modelo
    vetor = vetor.reshape(1, -1) / 255.0

    # Realiza predição e retorna a classe prevista e as confianças para cada classe
    pred = modelo.predict(vetor)[0]
    classe = encoder.inverse_transform([np.argmax(pred)])[0]
    confianca = {encoder.classes_[i]: round(float(pred[i]) * 100, 2) for i in range(len(pred))}
    return classe, confianca


# --- Rota principal do Flask para upload e classificação ---
@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    confianca = None
    imagem_path = None

    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            filename = str(uuid.uuid4()) + ".png"
            caminho = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(caminho)

            # Caminho de sistema para a função de predição
            classe, confianca = prever(caminho)
            resultado = classe

            # Caminho da imagem para exibir no HTML (ajustado para URL)
            imagem_path = os.path.join('uploads', filename).replace("\\", "/")

    # Renderiza o template HTML, enviando resultados e caminho da imagem para exibição
    return render_template("index.html", resultado=resultado, confianca=confianca, imagem=imagem_path, etapas={})

if __name__ == '__main__':
    # Executa o Flask em modo debug para facilitar testes
    app.run(debug=True)
