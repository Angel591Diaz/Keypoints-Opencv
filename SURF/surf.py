import cv2
import numpy as np

# Cargar la imagen
img = cv2.imread('img.jpg', 0)  # Se carga en escala de grises

# Inicializar el detector y descriptor SURF
surf = cv2.xfeatures2d.SURF_create()

# Detectar puntos clave y calcular descriptores
keypoints, descriptors = surf.detectAndCompute(img, None)

# Dibujar los puntos clave en la imagen
img_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Mostrar la imagen con los puntos clave dibujados
cv2.imshow('Puntos Clave SURF', img_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()