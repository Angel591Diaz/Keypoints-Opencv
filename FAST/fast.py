import cv2

# Cargar la imagen
img = cv2.imread("img.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Crear el detector FAST
fast = cv2.FastFeatureDetector_create()

# Detectar keypoints
keypoints = fast.detect(gray, None)

# Dibujar los keypoints en la imagen
img_with_keypoints = cv2.drawKeypoints(gray, keypoints, None, color=(0,255,20), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

# Mostrar la imagen con keypoints
cv2.imshow('Imagen con Keypoints FAST', img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('fast_keypoint.jpg', img_with_keypoints)