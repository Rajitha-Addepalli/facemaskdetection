import cv2
import numpy as np
import tensorflow as tf
import json
import os
import time
import csv
import winsound

from datetime import datetime
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "mask_detector.keras"
CLASS_NAMES_PATH = "class_names.json"

DETECTION_LOG = "detection_log.csv"
VIOLATION_LOG = "violation_log.csv"

IMG_SIZE = 128

CAMERA_INDEX = 0

CONFIDENCE_THRESHOLD = 0.50
UNCERTAIN_THRESHOLD = 0.65

DETECTION_SCALE = 1.1
DETECTION_NEIGHBORS = 5
MIN_FACE_SIZE = (50, 50)

# Detect faces every N frames
DETECT_EVERY_N_FRAMES = 3

# Tracking
MAX_TRACK_DISTANCE = 100
MAX_MISSED_FRAMES = 15

# Alert settings
ALERT_COOLDOWN = 5.0

last_alert_time = 0


# ============================================================
# START
# ============================================================

print("\n==========================================")
print("   REAL-TIME FACE MASK MONITORING")
print("==========================================\n")


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading trained model...")

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"{MODEL_PATH} not found.\n"
        "Please run train_model.py first."
    )

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

if os.path.exists(CLASS_NAMES_PATH):

    with open(
        CLASS_NAMES_PATH,
        "r"
    ) as file:

        class_names = json.load(file)

else:

    class_names = {
        "0": "with_mask",
        "1": "without_mask"
    }


print("\nClass mapping:")
print(class_names)


# ============================================================
# FACE DETECTOR
# ============================================================

print("\nLoading face detector...")

face_cascade = cv2.CascadeClassifier(

    cv2.data.haarcascades
    +
    "haarcascade_frontalface_default.xml"

)


if face_cascade.empty():

    raise RuntimeError(
        "Could not load Haar Cascade."
    )


print("Face detector loaded successfully.")


# ============================================================
# CREATE DETECTION LOG
# ============================================================

if not os.path.exists(DETECTION_LOG):

    with open(
        DETECTION_LOG,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([

            "timestamp",
            "faces_detected",
            "mask_count",
            "no_mask_count",
            "occluded_count",
            "uncertain_count",
            "compliance_percentage"

        ])


# ============================================================
# CREATE VIOLATION LOG
# ============================================================

if not os.path.exists(VIOLATION_LOG):

    with open(
        VIOLATION_LOG,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([

            "timestamp",
            "person_id",
            "confidence"

        ])


# ============================================================
# TRACKING
# ============================================================

tracks = {}


# ============================================================
# GET LOWEST AVAILABLE PERSON ID
# ============================================================

def get_available_person_id():

    """
    Return the smallest unused positive person ID.

    Example:
        Existing IDs = [1, 3]
        New ID       = 2

    This prevents IDs from continuously becoming
    Person 4, Person 5, Person 6... after people leave.
    """

    person_id = 1

    while person_id in tracks:

        person_id += 1

    return person_id


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def calculate_distance(point1, point2):

    return np.sqrt(

        (point1[0] - point2[0]) ** 2

        +

        (point1[1] - point2[1]) ** 2

    )


# ============================================================
# UPDATE TRACKS
# ============================================================

def update_tracks(detections):

    """
    Match newly detected faces with existing tracks.

    Person IDs are reused when tracks disappear.
    """

    global tracks


    # --------------------------------------------------------
    # IF NO FACES ARE DETECTED
    # --------------------------------------------------------

    if len(detections) == 0:

        for person_id in list(tracks.keys()):

            tracks[person_id]["missed"] += 1


        # Remove old tracks

        for person_id in list(tracks.keys()):

            if (

                tracks[person_id]["missed"]
                >
                MAX_MISSED_FRAMES

            ):

                del tracks[person_id]


        return {}


    # --------------------------------------------------------
    # CALCULATE DETECTION CENTERS
    # --------------------------------------------------------

    centers = []

    for box in detections:

        x, y, w, h = box

        center = (

            int(x + w / 2),

            int(y + h / 2)

        )

        centers.append(center)


    # --------------------------------------------------------
    # MATCHING
    # --------------------------------------------------------

    matched_tracks = set()

    detection_to_track = {}


    # --------------------------------------------------------
    # SORT EXISTING TRACKS BY ID
    #
    # This makes ID assignment deterministic.
    # --------------------------------------------------------

    existing_tracks = sorted(
        tracks.items(),
        key=lambda item: item[0]
    )


    # --------------------------------------------------------
    # MATCH EACH DETECTION TO CLOSEST TRACK
    # --------------------------------------------------------

    for detection_index, center in enumerate(centers):

        best_track_id = None

        best_distance = MAX_TRACK_DISTANCE


        for track_id, track in existing_tracks:

            if track_id in matched_tracks:

                continue


            dist = calculate_distance(

                center,

                track["center"]

            )


            if dist < best_distance:

                best_distance = dist

                best_track_id = track_id


        if best_track_id is not None:

            matched_tracks.add(
                best_track_id
            )

            detection_to_track[
                detection_index
            ] = best_track_id


    # --------------------------------------------------------
    # CREATE NEW TRACKS
    # --------------------------------------------------------

    for detection_index, center in enumerate(centers):

        if detection_index in detection_to_track:

            continue


        # Get smallest available ID

        person_id = get_available_person_id()


        tracks[person_id] = {

            "person_id": person_id,

            "center": center,

            "box": detections[detection_index],

            "missed": 0,

            "status": "uncertain",

            "confidence": 0.0

        }


        detection_to_track[
            detection_index
        ] = person_id


    # --------------------------------------------------------
    # UPDATE MATCHED TRACKS
    # --------------------------------------------------------

    for detection_index, person_id in detection_to_track.items():

        tracks[person_id]["center"] = centers[
            detection_index
        ]

        tracks[person_id]["box"] = detections[
            detection_index
        ]

        tracks[person_id]["missed"] = 0


    # --------------------------------------------------------
    # INCREASE MISSED COUNT
    # --------------------------------------------------------

    for person_id in list(tracks.keys()):

        if person_id not in detection_to_track.values():

            tracks[person_id]["missed"] += 1


    # --------------------------------------------------------
    # REMOVE OLD TRACKS
    # --------------------------------------------------------

    for person_id in list(tracks.keys()):

        if (

            tracks[person_id]["missed"]
            >
            MAX_MISSED_FRAMES

        ):

            del tracks[person_id]


    return detection_to_track


# ============================================================
# VIOLATION ALERT
# ============================================================

def trigger_violation_alert(
    person_id,
    confidence
):

    global last_alert_time


    current_time = time.time()


    # Prevent continuous alerts

    if (

        current_time
        -
        last_alert_time

        <
        ALERT_COOLDOWN

    ):

        return


    last_alert_time = current_time


    print(
        "\n=========================================="
    )

    print(
        "MASK VIOLATION ALERT"
    )

    print(
        f"Person ID   : {person_id}"
    )

    print(
        f"Confidence  : {confidence:.1f}%"
    )

    print(
        f"Time        : "
        f"{datetime.now().strftime('%H:%M:%S')}"
    )

    print(
        "=========================================="
    )


    # --------------------------------------------------------
    # WINDOWS SOUND
    # --------------------------------------------------------

    try:

        winsound.PlaySound(

            "SystemExclamation",

            winsound.SND_ALIAS
            |
            winsound.SND_ASYNC

        )

    except Exception:

        print(
            "Audio alert unavailable."
        )


    # --------------------------------------------------------
    # SAVE VIOLATION
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(

        "%Y-%m-%d %H:%M:%S"

    )


    with open(

        VIOLATION_LOG,

        "a",

        newline=""

    ) as file:

        writer = csv.writer(file)

        writer.writerow([

            timestamp,

            person_id,

            f"{confidence:.1f}"

        ])


# ============================================================
# OPEN WEBCAM
# ============================================================

print("\nOpening webcam...")


cap = cv2.VideoCapture(
    CAMERA_INDEX
)


if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam."
    )


# ============================================================
# CAMERA RESOLUTION
# ============================================================

cap.set(

    cv2.CAP_PROP_FRAME_WIDTH,

    1280

)

cap.set(

    cv2.CAP_PROP_FRAME_HEIGHT,

    720

)


print(
    "Webcam started successfully."
)


print("\nControls:")

print(
    "Q = Quit"
)

print(
    "R = Reset tracking"
)

print(
    "C = Clear detection log"
)

print()


# ============================================================
# FPS
# ============================================================

previous_time = time.time()

fps = 0.0

frame_count = 0

last_detections = []


# ============================================================
# MAIN LOOP
# ============================================================

while True:


    # ========================================================
    # READ FRAME
    # ========================================================

    ret, frame = cap.read()


    if not ret:

        print(
            "Could not read webcam frame."
        )

        break


    # ========================================================
    # MIRROR
    # ========================================================

    frame = cv2.flip(

        frame,

        1

    )


    frame_count += 1


    # ========================================================
    # FPS
    # ========================================================

    current_time = time.time()

    elapsed = (

        current_time
        -
        previous_time

    )


    if elapsed > 0:

        instant_fps = 1.0 / elapsed

        fps = (

            0.9 * fps
            +
            0.1 * instant_fps

        )


    previous_time = current_time


    # ========================================================
    # FACE DETECTION
    # ========================================================

    if (

        frame_count
        %
        DETECT_EVERY_N_FRAMES

        ==
        0

    ):

        gray = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2GRAY

        )


        detected_faces = face_cascade.detectMultiScale(

            gray,

            scaleFactor=DETECTION_SCALE,

            minNeighbors=DETECTION_NEIGHBORS,

            minSize=MIN_FACE_SIZE

        )


        last_detections = list(
            detected_faces
        )


        update_tracks(
            last_detections
        )


    # ========================================================
    # BATCH FACE PREPARATION
    # ========================================================

    face_images = []

    prediction_track_ids = []


    for person_id, track in tracks.items():

        if track["missed"] > 0:

            continue


        x, y, w, h = track["box"]


        x1 = max(
            0,
            x
        )

        y1 = max(
            0,
            y
        )

        x2 = min(

            frame.shape[1],

            x + w

        )

        y2 = min(

            frame.shape[0],

            y + h

        )


        face = frame[

            y1:y2,

            x1:x2

        ]


        if face.size == 0:

            continue


        face = cv2.resize(

            face,

            (IMG_SIZE, IMG_SIZE)

        )


        # OpenCV BGR -> RGB

        face = cv2.cvtColor(

            face,

            cv2.COLOR_BGR2RGB

        )


        face = face.astype(
            np.float32
        )


        face_images.append(
            face
        )


        prediction_track_ids.append(
            person_id
        )


    # ========================================================
    # BATCH PREDICTION
    # ========================================================

    if len(face_images) > 0:

        batch = np.array(

            face_images,

            dtype=np.float32

        )


        batch = preprocess_input(
            batch
        )


        predictions = model.predict(

            batch,

            verbose=0

        ).flatten()


        for i, prediction in enumerate(predictions):


            person_id = prediction_track_ids[i]


            prediction = float(
                prediction
            )


            # ------------------------------------------------
            # CLASS
            # ------------------------------------------------

            if prediction >= CONFIDENCE_THRESHOLD:

                predicted_class = 1

                confidence = prediction

            else:

                predicted_class = 0

                confidence = 1.0 - prediction


            confidence_percentage = (

                confidence * 100

            )


            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            if confidence < UNCERTAIN_THRESHOLD:

                status = "uncertain"

            elif predicted_class == 0:

                status = "with_mask"

            else:

                status = "without_mask"


            tracks[person_id][
                "status"
            ] = status


            tracks[person_id][
                "confidence"
            ] = confidence_percentage


    # ========================================================
    # COUNTERS
    # ========================================================

    total_people = 0

    mask_count = 0

    no_mask_count = 0

    uncertain_count = 0

    occluded_count = 0


    # ========================================================
    # DRAW TRACKS
    # ========================================================

    for person_id, track in list(
        tracks.items()
    ):


        if track["missed"] > 0:

            continue


        total_people += 1


        x, y, w, h = track["box"]

        status = track["status"]

        confidence = track["confidence"]


        # ----------------------------------------------------
        # SMALL FACE / POSSIBLE OCCLUSION
        # ----------------------------------------------------

        face_area = w * h

        frame_area = (

            frame.shape[0]
            *
            frame.shape[1]

        )


        relative_size = (

            face_area
            /
            frame_area

        )


        if relative_size < 0.001:

            display_status = "occluded"

            occluded_count += 1


        else:

            display_status = status


        # ----------------------------------------------------
        # COLORS AND LABEL
        # ----------------------------------------------------

        if display_status == "with_mask":

            color = (

                0,
                255,
                0

            )

            label = (

                f"MASK "
                f"{confidence:.1f}%"

            )

            mask_count += 1


        elif display_status == "without_mask":

            color = (

                0,
                0,
                255

            )

            label = (

                f"NO MASK "
                f"{confidence:.1f}%"

            )

            no_mask_count += 1


        elif display_status == "occluded":

            color = (

                0,
                165,
                255

            )

            label = "OCCLUDED"


        else:

            color = (

                0,
                165,
                255

            )

            label = (

                f"UNCERTAIN "
                f"{confidence:.1f}%"

            )

            uncertain_count += 1


        # ====================================================
        # DRAW FACE BOX
        # ====================================================

        cv2.rectangle(

            frame,

            (x, y),

            (x + w, y + h),

            color,

            2

        )


        # ====================================================
        # PERSON ID
        # ====================================================

        cv2.putText(

            frame,

            f"Person {person_id}",

            (

                x,

                max(
                    25,
                    y - 35
                )

            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            color,

            2

        )


        # ====================================================
        # MASK STATUS
        # ====================================================

        cv2.putText(

            frame,

            label,

            (

                x,

                max(
                    45,
                    y - 10
                )

            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            color,

            2

        )


        # ====================================================
        # VIOLATION
        # ====================================================

        if display_status == "without_mask":

            # Strong red border

            cv2.rectangle(

                frame,

                (
                    x - 3,
                    y - 3
                ),

                (
                    x + w + 3,
                    y + h + 3
                ),

                (
                    0,
                    0,
                    255
                ),

                3

            )


            cv2.putText(

                frame,

                "!! VIOLATION !!",

                (

                    x,

                    y + h + 28

                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.60,

                (
                    0,
                    0,
                    255
                ),

                2

            )


            # Trigger audio alert

            trigger_violation_alert(

                person_id,

                confidence

            )


    # ========================================================
    # COMPLIANCE
    # ========================================================

    classified_people = (

        mask_count
        +
        no_mask_count

    )


    if classified_people > 0:

        compliance = (

            mask_count
            /
            classified_people

        ) * 100

    else:

        compliance = 0.0


    # ========================================================
    # INFORMATION PANEL
    # ========================================================

    panel_width = 500

    panel_height = 200


    cv2.rectangle(

        frame,

        (0, 0),

        (
            panel_width,
            panel_height
        ),

        (
            0,
            0,
            0
        ),

        -1

    )


    # ========================================================
    # PEOPLE
    # ========================================================

    cv2.putText(

        frame,

        f"People: {total_people}",

        (15, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (
            255,
            255,
            255
        ),

        2

    )


    # ========================================================
    # MASK
    # ========================================================

    cv2.putText(

        frame,

        f"Mask: {mask_count}",

        (15, 60),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (
            0,
            255,
            0
        ),

        2

    )


    # ========================================================
    # NO MASK
    # ========================================================

    cv2.putText(

        frame,

        f"No Mask: {no_mask_count}",

        (15, 90),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (
            0,
            0,
            255
        ),

        2

    )


    # ========================================================
    # OCCLUDED
    # ========================================================

    cv2.putText(

        frame,

        f"Occluded: {occluded_count}",

        (15, 120),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (
            0,
            165,
            255
        ),

        2

    )


    # ========================================================
    # UNCERTAIN
    # ========================================================

    cv2.putText(

        frame,

        f"Uncertain: {uncertain_count}",

        (15, 150),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (
            0,
            165,
            255
        ),

        2

    )


    # ========================================================
    # COMPLIANCE
    # ========================================================

    cv2.putText(

        frame,

        f"Compliance: {compliance:.1f}%",

        (300, 65),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.60,

        (
            255,
            255,
            255
        ),

        2

    )


    # ========================================================
    # FPS
    # ========================================================

    cv2.putText(

        frame,

        f"FPS: {fps:.1f}",

        (450, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (
            255,
            255,
            255
        ),

        2

    )


    # ========================================================
    # WARNING MESSAGE
    # ========================================================

    if no_mask_count > 0:

        cv2.rectangle(

            frame,

            (
                frame.shape[1] - 350,
                0
            ),

            (
                frame.shape[1],
                55
            ),

            (
                0,
                0,
                255
            ),

            -1

        )


        cv2.putText(

            frame,

            "MASK VIOLATION DETECTED",

            (
                frame.shape[1] - 335,
                35
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (
                255,
                255,
                255
            ),

            2

        )


    # ========================================================
    # LOG DATA
    # ========================================================

    timestamp = datetime.now().strftime(

        "%Y-%m-%d %H:%M:%S"

    )


    with open(

        DETECTION_LOG,

        "a",

        newline=""

    ) as file:

        writer = csv.writer(file)

        writer.writerow([

            timestamp,

            total_people,

            mask_count,

            no_mask_count,

            occluded_count,

            uncertain_count,

            f"{compliance:.2f}"

        ])


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(

        "Real-Time Face Mask Monitoring",

        frame

    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # Q = QUIT
    # --------------------------------------------------------

    if key == ord("q"):

        break


    # --------------------------------------------------------
    # R = RESET TRACKING
    # --------------------------------------------------------

    elif key == ord("r"):

        tracks.clear()

        print(
            "Tracking reset. Next detected person = Person 1."
        )


    # --------------------------------------------------------
    # C = CLEAR DETECTION LOG
    # --------------------------------------------------------

    elif key == ord("c"):

        with open(

            DETECTION_LOG,

            "w",

            newline=""

        ) as file:

            writer = csv.writer(file)

            writer.writerow([

                "timestamp",
                "faces_detected",
                "mask_count",
                "no_mask_count",
                "occluded_count",
                "uncertain_count",
                "compliance_percentage"

            ])


        print(
            "Detection log cleared."
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


print("\n==========================================")
print("   MONITORING STOPPED")
print("==========================================")

print(
    f"Detection log : {DETECTION_LOG}"
)

print(
    f"Violation log : {VIOLATION_LOG}"
)

print("==========================================\n")