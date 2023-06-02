# Proyecto de reconocimiento de emociones

Este proyecto tiene como objetivo realizar el reconocimiento de emociones a partir de imágenes almacenadas en el directorio `Emociones_Dataset`. El código, `main.py`, utiliza diversas técnicas de aprendizaje automático y procesamiento de imágenes para llevar a cabo la tarea.

También se puede ejecutar el proyecto en el siguiente collab (los resultado se obtuvieron de aquí):
https://colab.research.google.com/drive/1nCTzBTcTkUPJttVY2klcQf-wmcyqDam1?usp=sharing



## Estructura del proyecto
```
.
├── Emociones_Dataset/    # Directorio que contiene imágenes de emociones
└── main.py               # Archivo de script principal
```

## Cómo empezar
Siga estas instrucciones para correr el proyecto en su máquina local.

### Prerrequisitos
Este proyecto requiere las siguientes librerías de Python:
* PIL
* os
* numpy
* pywt
* matplotlib
* sklearn
* scipy
* fer
* cv2

Puede instalar estos paquetes con pip:

```
pip install pillow numpy pywavelets matplotlib scikit-learn scipy fer-plus opencv-python
```

### Ejecutar el proyecto
Después de haber instalado las librerías necesarias, puedes ejecutar `main.py` usando Python 3:

```bash
python3 main.py
```
