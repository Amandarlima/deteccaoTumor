# Classificação de Tumores Cerebrais com Visão Computacional e IA 

Este projeto aplica técnicas de **Visão Computacional** e **Inteligência Artificial** para detectar tumores cerebrais em imagens de ressonância magnética (MRI), utilizando uma interface web amigável.

##  Objetivo

Desenvolver um sistema automatizado para **diagnóstico auxiliar** de tumores cerebrais, focando na detecção precoce, com base em imagens médicas.

##  Tecnologias Utilizadas

- **Python 3**
- [Flask](https://flask.palletsprojects.com/) - Framework web
- [OpenCV](https://opencv.org/) - Processamento de imagem
- [NumPy](https://numpy.org/) - Manipulação de arrays
- [TensorFlow + Keras](https://www.tensorflow.org/) - Treinamento e classificação com rede neural
- [scikit-learn (SVM)](https://scikit-learn.org/) - Suporte à vetores de suporte (opcional)

##  Dataset Utilizado

Base de dados com **7023 imagens** de ressonância magnética cerebral, separadas em quatro categorias:

- **Glioma**
- **Meningioma**
- **Pituitary Tumor**
- **Normal (Sem Tumor)**

> Fonte: [Brain Tumor MRI Dataset - Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)  
> Desenvolvido por: *Masoud Nickparvar*

##  Pipeline do Sistema

1. **Upload da Imagem**: Interface web em Flask
2. **Pré-processamento**: Grayscale, redimensionamento, filtro Gaussiano, equalização de histograma
3. **Segmentação**: Thresholding binário e detecção de bordas (Canny)
4. **Extração de Características**: ORB (Oriented FAST and Rotated BRIEF)
5. **Classificação**: Rede Neural (MLP com Keras) ou SVM
6. **Resultado**: Exibição de diagnóstico ("Tumor Detectado" ou "Normal")

##  Estrutura do Modelo MLP

- **Entrada**: vetor médio dos descritores ORB (32)
- **Camadas ocultas**:
  - Dense (64) + ReLU
  - Dense (32) + ReLU
- **Saída**: Dense (4) + Softmax  
  - Classes: Glioma, Meningioma, Pituitary, Sem Tumor

Treinado com **20 épocas**, validação de **20%**, otimizador **Adam**, e função de perda **Categorical Crossentropy**.

##  Executando o Projeto

1. Execute a aplicação:
```bash
python app.py
```

2. Acesse no navegador:
```
http://localhost:5000
```


