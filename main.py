import cv2
import numpy as np

# Funkcja do regulowania kontrastu
def adjust_contrast(image, value):
    alpha = (value + 100) / 100.0
    adjusted_image = cv2.addWeighted(image, alpha, np.zeros(image.shape, image.dtype), 0, 0)
    return adjusted_image

# Funkcja do regulowania jasności
def adjust_brightness(image, value):
    adjusted_image = cv2.convertScaleAbs(image, beta=value)
    return adjusted_image
   
# Funkcja do regulowania składowych barwnych w przestrzeni RGB
def adjust_rgb(image, r_value, g_value, b_value):
    adjusted_image = image.copy()
    adjusted_image[:, :, 2] = cv2.addWeighted(adjusted_image[:, :, 2], r_value / 100.0, np.zeros(image.shape[:2], image.dtype), 0, 0)
    adjusted_image[:, :, 1] = cv2.addWeighted(adjusted_image[:, :, 1], g_value / 100.0, np.zeros(image.shape[:2], image.dtype), 0, 0)
    adjusted_image[:, :, 0] = cv2.addWeighted(adjusted_image[:, :, 0], b_value / 100.0, np.zeros(image.shape[:2], image.dtype), 0, 0)
    return adjusted_image

# Funkcja do regulowania składowych barwnych w przestrzeni HSV
def adjust_hsv(image, h_value, s_value, v_value):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 0] = cv2.addWeighted(hsv_image[:, :, 0], h_value / 100.0, np.zeros(image.shape[:2], image.dtype), 0, 0)
    hsv_image[:, :, 1] = cv2.addWeighted(hsv_image[:, :, 1], s_value / 100.0, np.zeros(image.shape[:2], image.dtype), 0, 0)
    hsv_image[:, :, 2] = cv2.addWeighted(hsv_image[:, :, 2], v_value / 100.0, np.zeros(image.shape[:2], image.dtype), 0, 0)
    adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return adjusted_image

# Funkcja do detekcji krawędzi
def detect_edges(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# Funkcja do wyostrzania obrazu
def sharpen_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    sharpened_image = cv2.addWeighted(image, 1, cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2BGR), 0.5, 0)
    return sharpened_image


# Inicjalizacja kamery
cap = cv2.VideoCapture(0)

# Ustawienie początkowych wartości suwaków
r_value = 100
g_value = 100
b_value = 100
h_value = 100
s_value = 100
v_value = 100
brightness_value = 0
contrast_value = 50

# Funkcja do aktualizacji suwaków
def update_trackbar_positions():
    cv2.setTrackbarPos('R', 'Controls', r_value)
    cv2.setTrackbarPos('G', 'Controls', g_value)
    cv2.setTrackbarPos('B', 'Controls', b_value)
    cv2.setTrackbarPos('H', 'Controls', h_value)
    cv2.setTrackbarPos('S', 'Controls', s_value)
    cv2.setTrackbarPos('V', 'Controls', v_value)
    cv2.setTrackbarPos('Brightness', 'Controls', brightness_value + 100)
    cv2.setTrackbarPos('Contrast', 'Controls', contrast_value)

# Utworzenie okna i suwaków
cv2.namedWindow('Camera')
cv2.namedWindow('Controls')
cv2.createTrackbar('R', 'Controls', r_value, 150, lambda x: None)
cv2.createTrackbar('G', 'Controls', g_value, 150, lambda x: None)
cv2.createTrackbar('B', 'Controls', b_value, 150, lambda x: None)
cv2.createTrackbar('H', 'Controls', h_value, 150, lambda x: None)
cv2.createTrackbar('S', 'Controls', s_value, 150, lambda x: None)
cv2.createTrackbar('V', 'Controls', v_value, 150, lambda x: None)
cv2.createTrackbar('Brightness', 'Controls', brightness_value + 100, 200, lambda x: None)
cv2.createTrackbar('Contrast', 'Controls', contrast_value, 100, lambda x: None)
update_trackbar_positions()

# Flagi dla efektów
grayscale_mode = False
edge_detection = False
sharpening = False

while True:
    # Odczyt obrazu z kamery
    ret, frame = cap.read()

    # Skalowanie jasności i kontrastu
    adjusted_frame = adjust_brightness(frame, brightness_value)
    adjusted_frame = adjust_contrast(adjusted_frame, contrast_value)

    # Regulowanie składowych barwnych w przestrzeni RGB
    adjusted_frame = adjust_rgb(adjusted_frame, r_value, g_value, b_value)

    # Regulowanie składowych barwnych w przestrzeni HSV
    adjusted_frame = adjust_hsv(adjusted_frame, h_value, s_value, v_value)

    # Włączanie/wyłączanie efektów
    if grayscale_mode:
        adjusted_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2GRAY)
        adjusted_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_GRAY2BGR)

    if edge_detection:
        adjusted_frame = detect_edges(adjusted_frame)

    if sharpening:
        adjusted_frame = sharpen_image(adjusted_frame)

    # Wyświetlanie obrazu
    cv2.imshow('Camera', adjusted_frame)

    # Obsługa przycisków klawiatury
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('g'):
        grayscale_mode = not grayscale_mode
    elif key == ord('e'):
        edge_detection = not edge_detection
    elif key == ord('s'):
        sharpening = not sharpening

    # Aktualizacja suwaków
    r_value = cv2.getTrackbarPos('R', 'Controls')
    g_value = cv2.getTrackbarPos('G', 'Controls')
    b_value = cv2.getTrackbarPos('B', 'Controls')
    h_value = cv2.getTrackbarPos('H', 'Controls')
    s_value = cv2.getTrackbarPos('S', 'Controls')
    v_value = cv2.getTrackbarPos('V', 'Controls')
    brightness_value = cv2.getTrackbarPos('Brightness', 'Controls') - 100
    contrast_value = cv2.getTrackbarPos('Contrast', 'Controls') + 50

# Zwalnianie zasobów
cap.release()
cv2.destroyAllWindows()
