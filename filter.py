import cv2
import numpy

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

sunglasses = cv2.imread(r"C:\Users\janra\OneDrive\Desktop\python project\sunglasses (1).png", cv2.IMREAD_UNCHANGED)

if sunglasses is None:
    print("Error loading image")
    exit()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        gw = int(w)
        gh = int(sunglasses.shape[0] * gw / sunglasses.shape[1])
        
        resized = cv2.resize(sunglasses, (gw, gh))
        
        y1 = y + int(h * 0.00001)
        y2 = y1 + gh
        x1 = x
        x2 = x + gw
        
        # Boundary check
        y1 = max(0, y1)
        y2 = min(frame.shape[0], y2)
        x1 = max(0, x1)
        x2 = min(frame.shape[1], x2)
        
        if y2 - y1 != gh or x2 - x1 != gw:
            continue
        
        if resized.shape[2] == 4:
            alpha = resized[:, :, 3] / 255.0
        else:
            alpha = numpy.ones((gh, gw))
        
        for c in range(3):
            frame[y1:y2, x1:x2, c] = (
                alpha * resized[:, :, c] +
                (1 - alpha) * frame[y1:y2, x1:x2, c]
            )
    
    cv2.imshow("Sunglasses Filter", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
