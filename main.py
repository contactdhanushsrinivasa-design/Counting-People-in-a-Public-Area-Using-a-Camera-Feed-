import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from imutils.video import FPS
import numpy as np
import dlib
import logging
import time
import threading
import os
import firebase_admin
from firebase_admin import credentials, firestore


# execution start time
start_time = time.time()

# setup logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Firebase
basedir = os.path.dirname(os.path.abspath(__file__))
key_path = os.path.join(basedir, "serviceAccountKey.json")
try:
    cred = credentials.Certificate(key_path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    logger.info("Firebase initialized successfully.")
except FileNotFoundError:
    logger.error("serviceAccountKey.json not found. Please download your Firebase service account key from the Firebase console and place it in the project directory as 'serviceAccountKey.json'.")
    exit(1)
except Exception as e:
    logger.error(f"Failed to initialize Firebase: {e}")
    exit(1)
model = YOLO('yolov8x.pt')


## Input Video
video_path = r"C:\Users\dhanu\Downloads\demopro\new.mp4"
logger.info("Starting the video..")
cap = cv2.VideoCapture(video_path)

##for camera ip
# camera_ip = "Camera Url"
# logger.info("Starting the live stream..")
# cap = cv2.VideoCapture(camera_ip)
# time.sleep(1.0)


basedir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(basedir, "coco.txt")
my_file = open(file_path, "r")
data = my_file.read()
class_list = data.split("\n")


#function for detect person coordinate
def update_firestore(entered, exited, current_inside, timestamp):
    """
    Updates the Firestore with current people count data.
    """
    try:
        # Update live data
        doc_ref = db.collection('people_counter').document('live')
        data = {
            'entered': entered,
            'exited': exited,
            'current_inside': current_inside,
            'last_updated': firestore.SERVER_TIMESTAMP
        }
        doc_ref.set(data)
        
        # Add to history
        db.collection('people_counter_history').add(data)
        
        logger.info("Firestore updated: Entered {}, Exited {}, Inside {}".format(entered, exited, current_inside))
    except Exception as e:
        logger.error("Error updating Firestore: {}".format(str(e)))


def get_person_coordinates(frame):
    """
    Extracts the coordinates of the person bounding boxes from the YOLO model predictions.

    Args:
        frame: Input frame for object detection.

    Returns:
        list: List of person bounding box coordinates in the format [x1, y1, x2, y2].
    """
    results = model.predict(frame, verbose=False)
    a = results[0].boxes.data.detach().cpu()
    px = pd.DataFrame(a).astype("float")

    list_corr = []
    for index, row in px.iterrows():
        x1 = row[0]
        y1 = row[1]
        x2 = row[2]
        y2 = row[3]
        d = int(row[5])
        c = class_list[d]
        if 'person' in c:
            list_corr.append([x1, y1, x2, y2])
    return list_corr


def people_counter():
    """
    Counts the number of people entering and exiting based on object tracking using DeepSORT.
    """
    count = 0

    writer = None
    # Initialize DeepSORT Tracker
    tracker = DeepSort(max_age=30)
    trackableObjects = {}
    unique_ids = set()

    # Initialize the total number of frames processed thus far, along
    # with the total number of objects that have moved either up or down
    totalFrames = 0
    totalDown = 0
    totalUp = 0

    # Initialize empty lists to store the counting data
    total = []
    move_out = []
    move_in = []

    # Initialize video writer
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter('Final_output.mp4', fourcc, 30, (W, H), True)

    fps = FPS().start()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 3 != 0:
            continue

        frame = cv2.resize(frame, (500, 280))

        # Get person coordinates using YOLO
        per_corr = get_person_coordinates(frame)

        # Convert YOLO detections to DeepSORT format
        detections = []
        confidences = []
        results = model.predict(frame, verbose=False)
        yolo_results = results[0]

        for box in yolo_results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Filter for person class (class 0) with confidence > 0.5
            if cls_id == 0 and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # DeepSORT format: ([x, y, width, height], confidence, class_name)
                detections.append(([x1, y1, x2-x1, y2-y1], conf, "person"))
                confidences.append(conf)

        # Calculate accuracy score (average confidence)
        accuracy_score = np.mean(confidences) if confidences else 0.0

        # Apply DeepSORT Tracking
        tracks = tracker.update_tracks(detections, frame=frame)

        cv2.line(frame, (0, H // 2 - 10), (W, H // 2 - 10), (0, 0, 0), 2)

        # Process tracks and count people
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            unique_ids.add(track_id)
            
            # Get bounding box coordinates
            l, t, r, b = map(int, track.to_ltrb())
            centroid = ((l + r) // 2, (t + b) // 2)

            # Track object movement for counting
            to = trackableObjects.get(track_id)

            if to is None:
                to = TrackableObject(track_id, centroid)
            else:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                if not to.counted:
                    if direction < 0 and centroid[1] < H // 2 - 20:
                        totalUp += 1
                        move_out.append(totalUp)
                        to.counted = True
                    elif 0 < direction < 1.1 and centroid[1] > 144:
                        totalDown += 1
                        move_in.append(totalDown)
                        to.counted = True

                        total = []
                        total.append(len(move_in) - len(move_out))

            trackableObjects[track_id] = to

            # Draw bounding box and ID
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            text = "ID {}".format(track_id)
            cv2.putText(frame, text, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
            cv2.circle(frame, centroid, 4, (255, 255, 255), -1)

        info_status = [
            ("Enter", totalUp),
            ("Exit ", totalDown),
        ]

        # info_total = [("Total people inside", ', '.join(map(str, total)))]
        
        for (i, (k, v)) in enumerate(info_status):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Display Total People Count (from DeepSORT)
        cv2.putText(frame, f"Total People: {len(unique_ids)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

        # Display Accuracy Score (average confidence)
        cv2.putText(frame, f"Accuracy: {accuracy_score:.2f}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 165, 0), 2)

        # Update Firestore every 100 frames
        if totalFrames % 100 == 0:
            current_inside = len(move_in) - len(move_out)
            update_firestore(totalUp, totalDown, current_inside, time.time())

        writer.write(frame)
        cv2.imshow("People Count", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        totalFrames += 1
        fps.update()

        end_time = time.time()
        num_seconds = (end_time - start_time)
        if num_seconds > 28800:
            break

    # Final update to Firestore with end-of-run summary
    final_inside = len(move_in) - len(move_out)
    update_firestore(totalUp, totalDown, final_inside, time.time())

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    fps.stop()
    logger.info("Elapsed time: {:.2f}".format(fps.elapsed()))
    logger.info("Approx. FPS: {:.2f}".format(fps.fps()))



if __name__ == "__main__":
    people_counter()
    

## Apply threading also

#def start_people_counter():
 #    t1 = threading.Thread(target=people_counter)
  #   t1.start()


#if __name__ == "__main__":
 #   start_people_counter()
