import cv2
import numpy as np

# Cargar la imagen
img = cv2.imread('img.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Inicializar el detector de características FAST
fast = cv2.FastFeatureDetector_create()

# Detectar los puntos clave (keypoints) con FAST
keypoints = fast.detect(gray, None)

# Inicializar el descriptor BRIEF
brief = cv2.DescriptorExtractor_create("BRIEF")

# Calcular los descriptores BRIEF para los keypoints
keypoints, descriptors = brief.compute(gray, keypoints)

# Mostrar la imagen con los puntos clave dibujados
img_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Keypoints BRIEF', img_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
