import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.optimizers import Adam

# Caminhos das pastas
train_dir = 'dataset/Training'
test_dir = 'dataset/Testing'

IMG_SIZE = 300
orb = cv2.ORB_create()  

# Pré-processamento e extração de características com ORB
def processar_imagens(diretorio):
    X = []
    y = []
    classes = os.listdir(diretorio)

    for classe in classes:
        caminho_classe = os.path.join(diretorio, classe)
        for imagem_nome in os.listdir(caminho_classe):
            caminho_img = os.path.join(caminho_classe, imagem_nome)
            imagem = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)
            if imagem is None:
                continue

            imagem = cv2.resize(imagem, (IMG_SIZE, IMG_SIZE))
            imagem = cv2.GaussianBlur(imagem, (3, 3), 0)

            # Sharpening
            kernel_sharp = np.array([[0, -1, 0],
                                     [-1, 5, -1],
                                     [0, -1, 0]])
            imagem = cv2.filter2D(imagem, -1, kernel_sharp)

            imagem = cv2.equalizeHist(imagem)

            # Rotação
            centro = (IMG_SIZE // 2, IMG_SIZE // 2)
            matriz_rotacao = cv2.getRotationMatrix2D(centro, 15, 1.0)
            imagem = cv2.warpAffine(imagem, matriz_rotacao, (IMG_SIZE, IMG_SIZE))

            # Segmentação
            _, imagem_thresh = cv2.threshold(imagem, 120, 255, cv2.THRESH_BINARY)
            imagem_canny = cv2.Canny(imagem_thresh, 70, 150)

            # ORB
            kp, des = orb.detectAndCompute(imagem_canny, None)
            vetor = np.mean(des, axis=0) if des is not None else np.zeros(32)

            X.append(vetor)
            y.append(classe)

    return np.array(X), np.array(y)

# Carregar e processar dados
X_train, y_train = processar_imagens(train_dir)
X_test, y_test = processar_imagens(test_dir)

# Normalizar
X_train = X_train / 255.0
X_test = X_test / 255.0

# Codificar classes
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_test_enc = encoder.transform(y_test)

# One-hot
y_train_cat = to_categorical(y_train_enc)
y_test_cat = to_categorical(y_test_enc)

# Modelo Keras
modelo = Sequential()
modelo.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
modelo.add(Dense(32, activation='relu'))
modelo.add(Dense(4, activation='softmax'))

modelo.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = modelo.fit(X_train, y_train_cat, epochs=20, batch_size=8, validation_split=0.2)

# Avaliação
loss, acc = modelo.evaluate(X_test, y_test_cat)
print(f"Acurácia final: {acc:.2f}")

# Predições
y_pred = modelo.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print(classification_report(y_test_enc, y_pred_classes, target_names=encoder.classes_))

# Gráficos
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
# Salvar modelo treinado
modelo.save("modelo_tumor.keras")

# Salvar classes para o encoder
np.save("classes_labels.npy", encoder.classes_)
