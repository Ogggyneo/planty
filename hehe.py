import cv2
import numpy as np

print("OpenCV version:", cv2.__version__)

# tạo 1 ảnh đen 480x640
img = np.zeros((480, 640, 3), dtype=np.uint8)

# vẽ 1 hình chữ nhật cho dễ thấy
cv2.rectangle(img, (100, 100), (540, 380), (0, 255, 0), 2)
cv2.putText(img, "Test imshow", (180, 250),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

cv2.imshow("Test OpenCV Imshow", img)
print("Window opened. Press any key in the image window to close.")

key = cv2.waitKey(0)  # đợi ấn phím
print("Key pressed:", key)

cv2.destroyAllWindows()
print("Done.")
