import streamlit as st
import cv2
import numpy as np
import os
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

st.set_page_config(page_title="Visualizador de Processamento de RM", layout="wide")
st.title("🧠 Visualizador Etapa a Etapa de Processamento de Ressonância Magnética")

IMG_SIZE = 300
orb = cv2.ORB_create()

# Carrega o modelo e encoder
modelo = load_model("modelo_tumor.keras")
encoder = LabelEncoder()
encoder.classes_ = np.load("classes_labels.npy", allow_pickle=True)

# Função para validar se parece uma RM
def imagem_parece_cerebro(imagem):
    blur = cv2.GaussianBlur(imagem, (5, 5), 0)
    edges = cv2.Canny(blur, 70, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_total = sum([cv2.contourArea(cnt) for cnt in contours])
    return area_total > 1000  # limiar empírico

# Processamento passo a passo
def gerar_etapas(imagem_cv):
    original = imagem_cv
    resized = cv2.resize(original, (IMG_SIZE, IMG_SIZE))
    blur = cv2.GaussianBlur(resized, (5, 5), 0)
    kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(blur, -1, kernel_sharp)
    equalized = cv2.equalizeHist(sharpened)
    _, thresholded = cv2.threshold(equalized, 127, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresholded, 70, 150)
    kp, des = orb.detectAndCompute(edges, None)
    orb_img = cv2.drawKeypoints(edges, kp, None, color=(0, 255, 0))
    (h, w) = equalized.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, 30, 1.0)
    rotated = cv2.warpAffine(equalized, matrix, (w, h))

    imagens = [original, resized, blur, sharpened, equalized, thresholded, edges, orb_img, rotated]
    titulos = ["Original", "Redimensionada", "Blur", "Sharpening", "Equalizada", "Threshold", "Canny", "ORB Keypoints", "Rotacionada"]
    return imagens, titulos, des

# Classificação com base no vetor ORB médio
def classificar(descritor):
    if descritor is None:
        vetor = np.zeros(32)
    else:
        vetor = np.mean(descritor, axis=0)
    vetor = vetor.reshape(1, -1) / 255.0
    pred = modelo.predict(vetor)[0]
    classe_predita = encoder.inverse_transform([np.argmax(pred)])[0]
    return classe_predita, pred

# Upload da imagem
arquivo = st.file_uploader("📤 Envie uma imagem de RM (cérebro)", type=["jpg", "png"])

if arquivo:
    img_bytes = np.asarray(bytearray(arquivo.read()), dtype=np.uint8)
    imagem_cv = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

    st.subheader("Imagem Enviada")
    st.image(imagem_cv, width=300, channels="GRAY")

    if not imagem_parece_cerebro(imagem_cv):
        st.error("⚠️ A imagem enviada não parece ser uma ressonância magnética cerebral. Tente outra.")
    else:
        st.success("✅ Imagem válida. Mostrando etapas de processamento:")
        imagens, titulos, des = gerar_etapas(imagem_cv)

        cols = st.columns(3)
        for i in range(len(imagens)):
            with cols[i % 3]:
                st.image(imagens[i], caption=titulos[i], use_column_width=True, channels="GRAY")

        st.subheader("📈 Resultado da Classificação Final")
        classe, predicoes = classificar(des)

        st.success(f"🧠 Resultado: **{classe.upper()}**")

        confianca = {encoder.classes_[i]: float(predicoes[i]) for i in range(len(predicoes))}
        st.bar_chart(confianca)