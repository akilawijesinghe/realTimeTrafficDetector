import cv2
import requests
import numpy as np
import uuid

# Initialize the video capture from the default camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Error: Could not open video device")

# Load YOLO model weights and configuration
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Define the full COCO classes that YOLOv3 recognizes
classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
           "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
           "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
           "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
           "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
           "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
           "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
           "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
           "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
           "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
           "toothbrush"]

# URL for the API to check traffic conditions
TRAFFIC_API_URL = 'https://cqutrbas.au/api/check_traffic'


# Get the MAC address of the current machine
def get_mac_address():
    mac_num = hex(uuid.getnode()).replace('0x', '').upper()
    mac = ':'.join(mac_num[i:i+2] for i in range(0, 12, 2))
    return mac


def display_ad(video_url):
    # Release the video capture before playing the ad
    cap.release()
    cv2.destroyAllWindows()

    # Open the video from the URL
    ad_cap = cv2.VideoCapture(video_url)

    if not ad_cap.isOpened():
        print("Error: Could not open video URL")
        return

    while ad_cap.isOpened():
        ret, frame = ad_cap.read()
        if not ret:
            break

        cv2.imshow("Ad", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):  # Allow manual exit by pressing 'q'
            break

    # Close the video and destroy all windows
    ad_cap.release()
    cv2.destroyAllWindows()

    # Reopen the video capture for traffic detection
    cap.open(0)


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    height, width, channels = frame.shape

    # Preprocess the frame for YOLO detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    vehicle_count = 0

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Ensure that the class_id is within the range of the classes list
            if confidence > 0.3 and class_id < len(classes):
                detected_class = classes[class_id]
                if detected_class in ['car', 'motorbike', 'bus', 'truck']:
                    vehicle_count += 1
                    # Optional: Draw bounding box on detected vehicles for visualization
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f'{detected_class}: {round(confidence, 2)}', (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Get the MAC address
    mac_address = get_mac_address()

    # print(mac_address)

    # Send request to API to check traffic conditions
    response = requests.get(TRAFFIC_API_URL, params={'vehicle_count': vehicle_count,'mac_address': mac_address})

    if response.status_code == 200:
        data = response.json()
        print(data['message'])
        print(mac_address)
        if data['status'] == 'success':
            print(data['video_link'])
            requests.post(TRAFFIC_API_URL, data={'ad_displaying': 'true'})

            # Display the ad video from the provided URL
            display_ad(data['video_link'])

            requests.post(TRAFFIC_API_URL, data={'ad_displaying': 'false'})
    else:
        print(f"API call failed with status code: {response.status_code}")

    # Display the frame with vehicle count and bounding boxes
    cv2.putText(frame, f'Vehicle Count: {vehicle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                cv2.LINE_AA)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
