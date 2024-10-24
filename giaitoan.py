import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh đầu vào
img = cv2.imread('input_image_path', 0)  # Đọc ảnh dưới dạng grayscale

# Phương pháp Sobel
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Đạo hàm theo hướng x
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Đạo hàm theo hướng y
sobel = np.sqrt(sobelx**2 + sobely**2)

# Phương pháp Laplace Gaussian
laplacian_gaussian_kernel = np.array([[0, 0, -1, 0, 0],
                                      [0, -1, -2, -1, 0],
                                      [-1, -2, 16, -2, -1],
                                      [0, -1, -2, -1, 0],
                                      [0, 0, -1, 0, 0]])

laplace_gaussian = cv2.filter2D(img, -1, laplacian_gaussian_kernel)

# Hiển thị kết quả
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(sobel, cmap='gray')
plt.title('Sobel Edge Detection')

plt.subplot(1, 2, 2)
plt.imshow(laplace_gaussian, cmap='gray')
plt.title('Laplace Gaussian Edge Detection')

plt.show()
