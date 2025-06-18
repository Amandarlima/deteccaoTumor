import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Caminho base das imagens de exemplo
base_path = 'dataset/Training'

# Lista todas as classes disponíveis
classes = os.listdir(base_path)

print("Escolha uma classe:")
for i, classe in enumerate(classes):
    print(f"[{i}] {classe}")

classe_idx = int(input("Digite o número da classe desejada: "))
classe_escolhida = classes[classe_idx]

# Lista imagens dentro da classe
imagens = os.listdir(os.path.join(base_path, classe_escolhida))
print("\nImagens disponíveis:")
for i, nome_img in enumerate(imagens[:10]):  # Limita a 10 para facilitar
    print(f"[{i}] {nome_img}")

img_idx = int(input("Digite o número da imagem desejada: "))

# Caminho da imagem
caminho_imagem = os.path.join(base_path, classe_escolhida, imagens[img_idx])

# === PROCESSAMENTO ===
original = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
resized = cv2.resize(original, (300, 300))
blur = cv2.GaussianBlur(resized, (5, 5), 0)

# Sharpening
kernel_sharp = np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]])
sharpened = cv2.filter2D(blur, -1, kernel_sharp)

# Equalização
equalized = cv2.equalizeHist(sharpened)

# Threshold
_, thresholded = cv2.threshold(equalized, 127, 255, cv2.THRESH_BINARY)

# Canny
edges = cv2.Canny(thresholded, 70, 150)

# ORB
orb = cv2.ORB_create()
kp, des = orb.detectAndCompute(edges, None)
orb_keypoints = cv2.drawKeypoints(edges, kp, None, color=(0,255,0))

# Rotação
(h, w) = equalized.shape[:2]
center = (w // 2, h // 2)
matrix = cv2.getRotationMatrix2D(center, 30, 1.0)
rotated = cv2.warpAffine(equalized, matrix, (w, h))

# === EXIBIÇÃO ===
fig, axs = plt.subplots(3, 3, figsize=(12, 10))
fig.suptitle("Etapas do Processamento da Imagem", fontsize=16)

imagens = [original, resized, blur, sharpened, equalized, thresholded, edges, orb_keypoints, rotated]
titulos = ["Original", "Redimensionada", "Blur", "Sharpening", "Equalizada", "Threshold", "Canny", "ORB Keypoints", "Rotacionada"]

for i, ax in enumerate(axs.flat):
    ax.imshow(imagens[i], cmap='gray')
    ax.set_title(titulos[i])
    ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()
