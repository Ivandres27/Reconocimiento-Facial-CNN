# Reconocimiento Facial con CNN  
**Red Neuronal Convolucional para Identificar 38 Personas**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-green?logo=jupyter)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-9cf?logo=scikit-learn)
![Accuracy](https://img.shields.io/badge/Accuracy-95.2%25-success)
![Dataset](https://img.shields.io/badge/Dataset-38%20Personas-blueviolet)
![Images](https://img.shields.io/badge/Imágenes-64x64-critical)
![License](https://img.shields.io/badge/Licencia-MIT-lightgrey)

---

## Descripción

Proyecto de **tesis de grado** que implementa una **Red Neuronal Convolucional (CNN)** para el **reconocimiento facial de 38 personas diferentes** utilizando imágenes en **escala de grises**.

### Características principales:
- **Preprocesamiento robusto**: normalización, redimensionamiento, remapeo de etiquetas con `LabelEncoder`.
- **División estratificada** del dataset con `train_test_split`.
- **Arquitectura CNN** con capas `Conv2D`, `MaxPool2D`, `Dropout` y `Dense`.
- **Evaluación completa** en `X_test` con:
  - Accuracy
  - Matriz de confusión (opcional)
  - Visualización de predicciones con **confianza**
- **Código 100% reproducible** y bien documentado.

---

## Estructura del Repositorio
Reconocimiento-Facial-CNN/:
Red_Convolucional_Caras_Tesis.ipynb--Notebook principal (código + resultados)
README.md--Este archivo;
LICENSE--Licencia MIT
.gitignore--Evita subir archivos pesados
