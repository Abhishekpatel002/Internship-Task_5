import numpy as np

def get_dominant_color(image):
    # Resize for speed
    image = cv2.resize(image, (50, 50))
    pixels = image.reshape(-1, 3)
    avg_color = np.mean(pixels, axis=0)
    return avg_color

def classify_color(avg_color):
    b, g, r = avg_color
    if b > 100 and g < 100 and r < 100:
        return 'blue'
    else:
        return 'other'
