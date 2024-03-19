import cv2
import numpy as np

# Cargar la imagen
img = cv2.imread('img.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Inicializar el descriptor ORB
orb = cv2.ORB_create()

# Detectar puntos clave y calcular descriptores
keypoints, descriptors = orb.detectAndCompute(gray, None)

# Dibujar los puntos clave en la imagen
img_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Mostrar la imagen con los puntos clave dibujados
cv2.imshow('Puntos Clave ORB', img_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()