import cv2
import numpy as np
import matplotlib.pyplot as plt

ct = cv2.imread('003a91841da04a5a31f808fb5c21538a.tiff')

print("original data type", ct.shape, ct.dtype)
print("original min and max", np.amin(ct), np.amax(ct))
plt.subplot(1, 3, 1)
plt.tight_layout()
plt.title("origin "+str(ct.dtype))
plt.imshow(ct, cmap="gray")

# rescale(16-bitを8-bitへdown-scaleする)
# 先にピクセル地を0-255の範囲に変換する。
amin = np.amin(ct)
amax = np.amax(ct)
scale = 255.0/(amax-amin)
values = ((ct-amin)*scale)
values[values < 0] = 0
values[values > 255] = 255
pixelsint8 = np.uint8(values)
print("down-scaled(8-bit) min and max", np.amin(pixelsint8),
       np.amax(pixelsint8))

plt,subplot(1, 3 ,2)
plt.tight_layout()
plt.title("rescale 16-bit to 8-bit")
plt.imshow(pixelsint8, cmap="gray")

