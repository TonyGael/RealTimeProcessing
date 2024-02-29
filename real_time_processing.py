# importamos las librerías
import cv2
import numpy as np

# Modos de ejecución de lso frames
# vc = 0 --> 48 # captura de video
# fd = 1 --> 49 # filtro de desenfoque
# fe = 2 --> 50 # filtro detector de esquinas
# fb = 3 --> 51 # filtro de bordes

# Parámetros para el detector de esquinas
esquinas_param = dict(maxCorners=500,       # máximo número de esquinas a detectar
                      qualityLevel=0.2,     # umbral mínimo para la deteccion de esquinas
                      minDistance=15,       # distancia entre pixeles
                      blockSize=9)          # area de pixeles

# modo
mood = 48

# empezamos la video captura
cap = cv2.VideoCapture(0)

# creamos el ciclo para ejecutar y manteer activos los frames
while True:
    # leemos fotogramas
    ret, frame = cap.read()

    # elegimos el modo: mood
    # normal
    if mood == 48:
        # mostramos los frames
        resultado = frame

    # desenfoque
    elif mood == 49:
        # modificamos los frames
        resultado = cv2.blur(frame, (13, 13))  # (13, 13) = kernel de desenfoque

    # bordes
    elif mood == 51:
        # modificamos los frames
        resultado = cv2.Canny(frame, 135, 150)  # pasamos los par de umbral superior e inferior

    # para las esquinas
    elif mood == 50:
        # hay que obtener los frames
        resultado = frame
        # convertimos a EDG
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # CALCULAMOS LAS CARACTERISTICAS DE LAS ESQUINAS
        esquinas = cv2.goodFeaturesToTrack(gray, **esquinas_param)

        # evaluamos / preguntamos si detectamos esquinas con esas caracteristicas
        if esquinas is not None:
            # iteramos
            for x, y in np.float32(esquinas).reshape(-1, 2):
                # convertimos a enteros
                x, y = int(x), int(y)
                # dibujamos la ubicación de las esquinas
                cv2.circle(resultado, (x, y), 10, (255, 0, 0), 1)

    # si presionamos otras teclas
    elif mood != 48 or mood != 49 or mood != 50 or mood != 51 or mood != -1:
        # no se hace nada
        resultado = frame

        # imprimimos un mensaje
        print(f'Tecla incorrecta!')

    # mostramos los frames
    cv2.imshow('VIDEO CAPTURA', resultado)

    # cerramos con lectura de teclado
    t = cv2.waitKey(1)

    # salimos
    if t == 27:
        break

    # seteamos mood
    elif t != -1:
        mood = t

# liberamos la video captura
cap.release()

# cerramos la ventana
cv2.destroyAllWindows()
