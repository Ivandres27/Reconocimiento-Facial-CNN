# 🧠 Reconocimiento Facial con CNN — Extended Yale Face Database B

Red neuronal convolucional (CNN) entrenada para identificar a 38 personas
a partir de imágenes faciales en escala de grises bajo distintas condiciones de iluminación.

**Accuracy final: 92.95%** — 448 de 482 predicciones correctas en el conjunto de prueba.

---

## 📸 Vista previa

> *Ejemplos de errores de clasificación. Las imágenes con iluminación extrema
> o ángulos difíciles representan los casos más desafiantes para el modelo.*

 ![Errores](assets/errores_clasificacion.png)

---

## 📁 Estructura del repositorio
```
face-recognition-cnn/
│
├── Red_Convolucional_Caras.ipynb   # Notebook principal con todo el pipeline
├── modelo_rostros.h5               # Modelo entrenado (pesos + arquitectura)
├── nombres_clases.npy              # Mapeo de índice a nombre de clase
├── data/
│   └── allFaces.mat                # Dataset Extended Yale Face Database B
├── assets/
│   ├── errores_clasificacion.png
│   └── confusion_matrix.png
└── README.md
```

---

## 📊 Dataset

**Extended Yale Face Database B**

| Característica | Detalle |
|---|---|
| Personas | 38 |
| Total de imágenes | ~2,414 |
| Dimensiones | 192 × 168 píxeles |
| Formato | Escala de grises (`.mat`) |
| Condiciones | Distintos ángulos e intensidades de iluminación |

El dataset está incluido en la carpeta `data/` como archivo `allFaces.mat`.

---

## 🏗️ Arquitectura del modelo

CNN secuencial construida con Keras:

- Capas **Conv2D** con activación ReLU y padding `same`
- Capas de **MaxPooling2D** para reducción espacial
- Capas de **Dropout** para regularización
- Capa densa final con activación **Softmax** (38 clases)

**Optimizador:** Adam (lr=0.001) | **Loss:** Categorical Crossentropy | **Épocas:** 30

---

## ✅ Resultados

| Métrica | Valor |
|---|---|
| Accuracy general | **92.95%** |
| Predicciones correctas | 448 / 482 |
| Personas con 100% de accuracy | 21 de 38 |
| Peor caso | Persona 29 — 61.5% |

Los errores se concentran principalmente en imágenes con iluminación extrema,
donde rasgos faciales distintivos quedan parcialmente ocultos.

---

## ⚙️ Cómo reproducir

### 1. Clonar el repositorio
```bash
git clone https://github.com/TU_USUARIO/face-recognition-cnn.git
cd face-recognition-cnn
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar el notebook

Abrí `Red_Convolucional_Caras.ipynb` en Jupyter y ejecutá todas las celdas
en orden con **Restart & Run All**.

---

## 🧰 Tecnologías

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-deeplearning-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green)

- Python 3.11
- TensorFlow / Keras
- NumPy · Pandas · Matplotlib · Seaborn
- scikit-learn · SciPy

---

## 📄 Contexto

Proyecto desarrollado como práctica académica durante la licenciatura,
enfocado en clasificación de imágenes con redes convolucionales.
