from flask import Flask, render_template, Response, jsonify
import cv2 as cv
import numpy as np
import mediapipe as mp
import time
from collections import deque
from AngleBuffer import AngleBuffer
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)

# Initialization variables
USER_FACE_WIDTH = 140
NOSE_TO_CAMERA_DISTANCE = 600
PRINT_DATA = True
SHOW_ALL_FEATURES = True
LOG_DATA = True
ENABLE_HEAD_POSE = True
SHOW_ON_SCREEN_DATA = True
TOTAL_BLINKS = 0
EYES_BLINK_FRAME_COUNTER = 0
BLINK_THRESHOLD = 0.51
EYE_AR_CONSEC_FRAMES = 2
LEFT_EYE_IRIS = [474, 475, 476, 477]
RIGHT_EYE_IRIS = [469, 470, 471, 472]
LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]
RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
NOSE_TIP_INDEX = 4
CHIN_INDEX = 152
LEFT_EYE_LEFT_CORNER_INDEX = 33
RIGHT_EYE_RIGHT_CORNER_INDEX = 263
LEFT_MOUTH_CORNER_INDEX = 61
RIGHT_MOUTH_CORNER_INDEX = 291
MOVING_AVERAGE_WINDOW = 10
DIRECTION_WINDOW = 15  # Number of frames to consider for smoothing direction
DIRECTION_THRESHOLD = 10  # Minimum frames to change direction

# Threshold values for yaw angle (these can be fine-tuned)
YAW_LEFT_THRESHOLD = -15  # Yaw value to consider looking left
YAW_RIGHT_THRESHOLD = 15   # Yaw value to consider looking right
YAW_FORWARD_BUFFER = 5     # Buffer range for considering forward (prevents rapid switching)

initial_pitch, initial_yaw, initial_roll = None, None, None
calibrated = False
current_direction = "Forward"  # To store and display the current direction
direction_stability_counter = 0  # Counts the number of stable frames in a particular direction
yaw_values = deque(maxlen=DIRECTION_WINDOW)  # Stores the last 'DIRECTION_WINDOW' yaw values

# Face mesh detector
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
)

# Initialize camera
cap = cv.VideoCapture(0)

# Email alert function
def send_email_alert(subject, message, to_email="tariroclaudia@gmail.com"):
    from_email = "claudiamhiribidi@gmail.com"
    password = "skydefxdtpsnroxh"  # Replace with app-specific password if using 2FA

    message += f"\n\nPlease follow this link to monitor the student:\nhttp://127.0.0.1:5000"
    
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))
    
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(from_email, password)
            server.sendmail(from_email, to_email, msg.as_string())
        print("Email alert sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Function to estimate head pose
def estimate_head_pose(landmarks, image_size):
    scale_factor = USER_FACE_WIDTH / 150.0
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0 * scale_factor, -65.0 * scale_factor),  # Chin
        (-225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),  # Left eye left corner
        (225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),  # Right eye right corner
        (-150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor),  # Left Mouth corner
        (150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor)  # Right mouth corner
    ])
  
    focal_length = image_size[1]
    center = (image_size[1] / 2, image_size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))

    image_points = np.array([
        landmarks[NOSE_TIP_INDEX],  # Nose tip
        landmarks[CHIN_INDEX],  # Chin
        landmarks[LEFT_EYE_LEFT_CORNER_INDEX],  # Left eye left corner
        landmarks[RIGHT_EYE_RIGHT_CORNER_INDEX],  # Right eye right corner
        landmarks[LEFT_MOUTH_CORNER_INDEX],  # Left mouth corner
        landmarks[RIGHT_MOUTH_CORNER_INDEX]  # Right mouth corner
    ], dtype="double")

    success, rotation_vector, translation_vector = cv.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE
    )

    rotation_matrix, _ = cv.Rodrigues(rotation_vector)
    projection_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))
    _, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(projection_matrix)

    pitch, yaw, roll = euler_angles.flatten()[:3]
    return pitch, yaw, roll

# Function to determine head direction with refined thresholds and stability checks
def get_head_direction(yaw):
    global current_direction, direction_stability_counter, yaw_values

    # Use the rolling average to anchor head direction more precisely
    yaw_values.append(yaw)  # Add the new yaw value
    avg_yaw = np.mean(yaw_values)  # Use a rolling average for smooth transitions

    # Check if the person is looking left, right, or forward based on yaw angle
    if avg_yaw < YAW_LEFT_THRESHOLD:
        new_direction = "Left"
    elif avg_yaw > YAW_RIGHT_THRESHOLD:
        new_direction = "Right"
    else:
        new_direction = "Forward"

    # Stability threshold check to reduce false direction changes
    if new_direction == current_direction:
        direction_stability_counter = 0  # Reset stability counter if direction hasn't changed
    else:
        direction_stability_counter += 1  # Increment stability counter

    # Commit to direction change if stability threshold met
    if direction_stability_counter >= DIRECTION_THRESHOLD:
        current_direction = new_direction  # Change direction if stable
        direction_stability_counter = 0  # Reset the stability counter

        # Trigger alert when looking Left or Right
        if current_direction in ["Left", "Right"]:
            send_email_alert("Alert: User Looking Away", f"The user is looking {current_direction}.")  # Send email alert

    return current_direction, current_direction in ["Left", "Right"]

# Blink detection logic
def euclidean_distance_3D(points):
    P0, P3, P4, P5, P8, P11, P12, P13 = points
    numerator = (
        np.linalg.norm(P3 - P13) ** 3
        + np.linalg.norm(P4 - P12) ** 3
        + np.linalg.norm(P5 - P11) ** 3
    )
    denominator = 3 * np.linalg.norm(P0 - P8) ** 3
    distance = numerator / denominator
    return distance

def blinking_ratio(landmarks):
    right_eye_ratio = euclidean_distance_3D(landmarks[RIGHT_EYE_POINTS])
    left_eye_ratio = euclidean_distance_3D(landmarks[LEFT_EYE_POINTS])
    ratio = (right_eye_ratio + left_eye_ratio + 1) / 2
    return ratio

# Main generator function for streaming video and processing
def generate_frames():
    global TOTAL_BLINKS, EYES_BLINK_FRAME_COUNTER, initial_pitch, initial_yaw, initial_roll, calibrated

    angle_buffer = AngleBuffer(size=MOVING_AVERAGE_WINDOW)

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        time.sleep(0.03)  # Small delay to ensure frame order
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = mp_face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array([
                np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                for p in results.multi_face_landmarks[0].landmark
            ])
            mesh_points_3D = np.array([[p.x, p.y, p.z] for p in results.multi_face_landmarks[0].landmark])

            # Head pose estimation
            pitch, yaw, roll = estimate_head_pose(mesh_points, (img_h, img_w))
            angle_buffer.add([pitch, yaw, roll])
            pitch, yaw, roll = angle_buffer.get_average()

            if initial_pitch is None:
                initial_pitch, initial_yaw, initial_roll = pitch, yaw, roll
                calibrated = True

            if calibrated:
                pitch -= initial_pitch
                yaw -= initial_yaw
                roll -= initial_roll

            # Get head direction based on yaw and apply smoothing
            head_direction, alert = get_head_direction(yaw)

            # Trigger alert for looking away
            if alert:
                print(f"Alert! Person is looking {head_direction}")

            # Blink detection
            eyes_aspect_ratio = blinking_ratio(mesh_points_3D)
            if eyes_aspect_ratio <= BLINK_THRESHOLD:
                EYES_BLINK_FRAME_COUNTER += 1
            else:
                if EYES_BLINK_FRAME_COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL_BLINKS += 1
                    if PRINT_DATA:
                        print(f"Blinked! Head direction: {head_direction}")
                    EYES_BLINK_FRAME_COUNTER = 0

            # Drawing features and displaying data
            for point in mesh_points:
                cv.circle(frame, tuple(point), 1, (0, 255, 0), -1)

            # Drawing iris
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_EYE_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_EYE_IRIS])
            cv.circle(frame, (int(l_cx), int(l_cy)), int(l_radius), (255, 0, 255), 2)
            cv.circle(frame, (int(r_cx), int(r_cy)), int(r_radius), (255, 0, 255), 2)

            # Displaying blink and head pose data on the screen
            if SHOW_ON_SCREEN_DATA:
                cv.putText(frame, f"Blinks: {TOTAL_BLINKS}", (30, 80), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
                cv.putText(frame, f"Pitch: {int(pitch)}", (30, 110), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
                cv.putText(frame, f"Yaw: {int(yaw)}", (30, 140), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
                cv.putText(frame, f"Roll: {int(roll)}", (30, 170), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
                cv.putText(frame, f"Looking: {head_direction}", (30, 200), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

        # Encode the frame to JPEG for streaming
        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame for video streaming
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Flask routes for rendering and video feed
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Flask route to send alert status to frontend
@app.route('/check_direction')
def check_direction():
    global current_direction
    return jsonify({'direction': current_direction, 'alert': current_direction in ["Left", "Right"]})

# Start Flask app
if __name__ == '__main__':
    app.run(debug=True)
