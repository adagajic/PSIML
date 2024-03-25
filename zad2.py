# imports for reading images
from PIL import Image
import numpy as np

if __name__ == '__main__':
    print("Hello, World!")
    # Load the images from .png files
    path = input()
    img = Image.open(path)
    img = np.array(img)
    # count all unique colors in the image
    
    x = img.shape[0]
    y = img.shape[1]
    matrix = np.ndarray((x, y), dtype=np.uint8)
    for i in range(x):
        for j in range(y):
            if img[i][j][0] == 255 and img[i][j][1] == 255 and img[i][j][2] == 255:
                matrix[i][j] = 0
            elif img[i][j][0] == 0 and img[i][j][1] == 0 and img[i][j][2] == 0:
                matrix[i][j] = 1
            else:
                matrix[i][j] = 2
    print(matrix)