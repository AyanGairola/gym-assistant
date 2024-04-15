# Import necessary libraries
from flask import Flask, render_template, Response
import cv2
import math
import mediapipe as mp

# Initialize Flask app
app = Flask(__name__)

# Initialize variables
Counter = 0
Flags = [False, False, False, False, False, False]
w = 0
ALLDONE = False
Workouts = [["SHOULDER_PRESS", 6, "Not Done"], ["BICEPS_CURL", 5, "Not Done"], ["PUSH_UPS", 5, "Not Done"], ["Squats", 5, "Not Done"]]

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open video stream
cap = cv2.VideoCapture(0)

# Define functions
def calculate_angle(point1, point2, point3):
    # Calculate the angle between three points
    vector1 = (point1[0] - point2[0], point1[1] - point2[1])
    vector2 = (point3[0] - point2[0], point3[1] - point2[1])

    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    cosine_angle = dot_product / (magnitude1 * magnitude2)
    angle = math.acos(cosine_angle)

    return math.degrees(angle)

def gen_frames():
    global frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            
            if results.pose_landmarks:
                Shoulders = DrawShoulders(frame, results)
                Workout(frame, results, Shoulders)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def DrawShoulders(frame, results):
    L = []
    R = []
    if results.pose_landmarks:
        h, w, c = frame.shape
        
        p1 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        p2 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        p3 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        
        cx, cy = int(p1.x * w), int(p1.y * h)
        p1 = (cx,cy)
        cx, cy = int(p2.x * w), int(p2.y * h)
        p2 = (cx,cy)
        cx, cy = int(p3.x * w), int(p3.y * h)
        p3 = (cx,cy)
        
        L = [p1,p2,p3]
        
        p1 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        p2 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        p3 = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        
        cx, cy = int(p1.x * w), int(p1.y * h)
        p1 = (cx,cy)
        cx, cy = int(p2.x * w), int(p2.y * h)
        p2 = (cx,cy)
        cx, cy = int(p3.x * w), int(p3.y * h)
        p3 = (cx,cy)
        
        R = [p1,p2,p3]
        
    return [L,R]

def Workout(frame, results, Shoulders):
    global w, Counter, Workouts, ALLDONE, Flags
    items = []
    for workout in Workouts:
        items.append(str(workout[0]) + " " + str(workout[1]) + " reps " + workout[2])
    
    draw_todo_list(frame, items)
    
    workout = Workouts[w]
    Shoulders = DrawShoulders(frame, results)
    
    if Shoulders[0] or Shoulders[1]:
        Techniques(workout[0], Shoulders)
        
    if Counter == workout[1]:
        print(workout[0], " Done")
        Workouts[w][2] = "Done"
        Flags = [False] * 6
        if w + 1 < len(Workouts):
            w += 1
        else:
            print("All Done")
            ALLDONE = True
            
        Counter = 0

def Techniques(Technique, Shoulders):
    if Technique == "BICEPS_CURL":
        BicepsCurl(Shoulders)
    elif Technique == "SHOULDER_PRESS":
        ShouldersPress(Shoulders)
    elif Technique == "PUSH_UPS":
        PushUp(Shoulders)
    elif Technique == "SQUATS":
        Squat(Shoulders)

def BicepsCurl(Shoulders):
    global Flags, Counter, frame

    p1, p2, p3 = Shoulders[0]
    angle = calculate_angle(p1, p2, p3)
    if angle < 50:
        Color = (0, 0, 255)
        Flags[0] = True
    elif angle > 130:
        Color = (255, 0, 0)
        Flags[2] = True
    else:
        Color = (0, 255, 0)
    cv2.line(frame, p1, p2, Color, 4)
    cv2.line(frame, p3, p2, Color, 4)

    p1, p2, p3 = Shoulders[1]
    angle = calculate_angle(p1, p2, p3)
    if angle < 50:
        Color = (0, 0, 255)
        Flags[1] = True
    elif angle > 130:
        Color = (255, 0, 0)
        Flags[3] = True
    else:
        Color = (0, 255, 0)
    cv2.line(frame, p1, p2, Color, 4)
    cv2.line(frame, p3, p2, Color, 4)

    if Flags[0] and Flags[1]:
        Flags[4] = True

    if Flags[2] and Flags[3]:
        Flags[5] = True

    if Flags[4] and Flags[5]:
        Flags = [False] * 6
        Counter += 0.5

def ShouldersPress(Shoulders):
    global Flags, Counter, frame

    p1, p2, p3 = Shoulders[0]
    angle = calculate_angle(p1, p2, p3)

    if p1[1] < p3[1]:
        Color = (0, 0, 255)
    else:
        Color = (0, 255, 0)
    cv2.line(frame, p1, p2, Color, 4)
    cv2.line(frame, p3, p2, Color, 4)

    if 70 <= angle <= 100:
        Flags[0] = True

    if 150 <= angle <= 180:
        Flags[1] = True

    p1, p2, p3 = Shoulders[1]
    angle = calculate_angle(p1, p2, p3)

    if p1[1] < p3[1]:
        Color = (0, 0, 255)
    else:
        Color = (0, 255, 0)

    cv2.line(frame, p1, p2, Color, 4)
    cv2.line(frame, p3, p2, Color, 4)

    if 70 <= angle <= 100:
        Flags[2] = True

    if 150 <= angle <= 180:
        Flags[3] = True

    if Flags[0] and Flags[1]:
        Flags[4] = True

    if Flags[2] and Flags[3]:
        Flags[5] = True

    if Flags[4] and Flags[5]:
        Flags = [False] * 6
        Counter += 0.5

def PushUp(Shoulders):
    global Flags, Counter, frame

    p1, p2, p3 = Shoulders[0]  # Left shoulder landmarks
    angle_left = calculate_angle(p1, p2, p3)

    p1, p2, p3 = Shoulders[1]  # Right shoulder landmarks
    angle_right = calculate_angle(p1, p2, p3)

    if 70 <= angle_left <= 100 and 70 <= angle_right <= 100:
        Flags[0] = True
        Flags[1] = True

    if all(Flags[0:2]):
        Flags[4] = True

    if Flags[4]:
        Flags = [False] * 6
        Counter += 1

def Squat(Shoulders):
    global Flags, Counter, frame

    p1, p2, p3 = Shoulders[0]  # Left hip, knee, ankle landmarks
    angle_left = calculate_angle(p1, p2, p3)

    p1, p2, p3 = Shoulders[1]  # Right hip, knee, ankle landmarks
    angle_right = calculate_angle(p1, p2, p3)

    if 100 <= angle_left <= 160 and 100 <= angle_right <= 160:
        Flags[2] = True
        Flags[3] = True

    if all(Flags[2:4]):
        Flags[5] = True

    if Flags[5]:
        Flags = [False] * 6
        Counter += 1

def draw_todo_list(frame, items):
    global Counter
    # Define some parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    line_spacing = 30
    padding = 20

    # Draw a rectangle around the list
    cv2.rectangle(frame, (padding, padding), (250, 20 + (len(items)) * line_spacing), (255, 255, 255), -1)
    cv2.rectangle(frame, (padding, padding), (250, 20 + (len(items)) * line_spacing), (0, 0, 0), 2)

    # Write the items of the list
    for i, item in enumerate(items):
        cv2.putText(frame, f"{i + 1}. {item}", (30, 40 + i * line_spacing), font, font_scale, (0, 0, 0), font_thickness)
    if not ALLDONE:
        cv2.putText(frame, "Count =" + str(int(Counter)), (20, 60 + (len(items)) * line_spacing), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    else:
        cv2.rectangle(frame, (50, 190), (590, 290), (255, 255, 255), -1)
        cv2.putText(frame, "All Done - Good Work", (55, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)

# Define Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
