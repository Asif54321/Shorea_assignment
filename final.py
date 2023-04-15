import cv2
import numpy as np

# Load the input image
img = cv2.imread('img4.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

# Create a mask for the detected face regions
face_mask = np.zeros_like(img)
for (x, y, w, h) in faces:
    
    cv2.rectangle(face_mask, (x, y), (x+w, y+h), (255, 255, 255), 2)

# Convert face_mask to grayscale
face_mask_gray = cv2.cvtColor(face_mask, cv2.COLOR_BGR2GRAY)

#FOR OIL FACE DETECT

# Apply silver mask to oil_only pixels
oil_only = img.copy()
oil_only[np.where(face_mask_gray > 0)] = (192, 192, 192)

# Convert oil_only to uint8 data type
oil_only = np.uint8(oil_only)

# Create a mask for the oiliness on the face
gray_face_mask = cv2.cvtColor(face_mask_gray, cv2.COLOR_GRAY2BGR)
oiliness_masked = cv2.bitwise_and(img, gray_face_mask)

# FOR ACNE DETECTION

gray = cv2.cvtColor(oiliness_masked, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 50, 150)

# Apply thresholding to obtain binary image
ret,thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Apply local contrast enhancement to acne regions
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_img = clahe.apply(gray)
clahe_acne = cv2.bitwise_and(clahe_img, clahe_img, mask=thresh)

# Combine original image and enhanced acne regions
result = cv2.addWeighted(img, 0.7, cv2.cvtColor(clahe_acne, cv2.COLOR_GRAY2BGR), 0.3, 0)


# FOR UV LIGHT

# Split image into color channels
b,g,r = cv2.split(result)

# Reduce intensity of red and green channels
r = cv2.multiply(r, 0.5)
g = cv2.multiply(g, 0.5)

# Merge color channels back into image
filtered_img = cv2.merge((b,g,r))

# Display the output 

face_roi = filtered_img[y:y+h, x:x+w]
cv2.imshow('Final Result', face_roi)
cv2.imwrite('output4.jpg', face_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()