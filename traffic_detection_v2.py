import cv2
import requests
import numpy as np

# Try different indices if necessary (0, 1, 2, etc.)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Error: Could not open video device")

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Define the classes of objects you want to detect
classes = ['car', 'motorbike', 'bus', 'truck']

# URL for the API to check traffic conditions
TRAFFIC_API_URL = 'http://localhost/FinalProject/ci3/traffic_api.php'

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    vehicle_count = 0

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Ensure class_id is within valid range and confidence threshold
            if class_id < len(classes) and confidence > 0.5 and classes[class_id] in classes:
                vehicle_count += 1

    # Send request to API to check traffic conditions
    response = requests.get(TRAFFIC_API_URL, params={'vehicle_count': vehicle_count})

    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'success':
            print(data['message'])
            print(data['video_link'])
            # Logic to display ads if it's a good time
        else:
            print(data['message'])
            # Logic to handle not displaying ads
    else:
        print(f"API call failed with status code: {response.status_code}")

    # Display the frame with vehicle count
    cv2.putText(frame, f'Vehicle Count: {vehicle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
