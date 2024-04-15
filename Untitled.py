import cv2
import math
import mediapipe as mp

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

def calculate_angle(point1, point2, point3):
    # Calculate the vectors between the points
    vector1 = (point1[0] - point2[0], point1[1] - point2[1])
    vector2 = (point3[0] - point2[0], point3[1] - point2[1])

    # Calculate the dot product of the vectors
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    # Calculate the magnitudes of the vectors
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    # Calculate the cosine of the angle using the dot product and magnitudes
    cosine_angle = dot_product / (magnitude1 * magnitude2)

    # Calculate the angle in radians using the arccosine function
    angle = math.acos(cosine_angle)

    return math.degrees(angle)

def DrawFullBody(frame, results):
    if results.pose_landmarks:
        h, w, c = frame.shape

        # Draw lines for the full body
        draw_body_lines(frame, results.pose_landmarks, w, h)

def draw_body_lines(frame, landmarks, w, h):
    # Define the connections between landmarks for the full body
    body_connections = [
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
        (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
        (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_KNEE),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_KNEE),
        (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
        (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
    ]

    # Draw lines for each body connection
    for connection in body_connections:
        start_landmark = landmarks.landmark[connection[0]]
        end_landmark = landmarks.landmark[connection[1]]
        start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
        end_point = (int(end_landmark.x * w), int(end_landmark.y * h))
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

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
        
        cv2.circle(frame, p1, 10, (0, 0, 255), 2)
        cv2.circle(frame, p1, 3, (0, 0, 255), -1)
        cv2.circle(frame, p2, 10, (0, 0, 255), 2)
        cv2.circle(frame, p2, 3, (0, 0, 255), -1)
        cv2.circle(frame, p3, 10, (0, 0, 255), 2)
        cv2.circle(frame, p3, 3, (0, 0, 255), -1)
        
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
        
        cv2.circle(frame, p1, 10, (0, 0, 255), 2)
        cv2.circle(frame, p1, 3, (0, 0, 255), -1)
        cv2.circle(frame, p2, 10, (0, 0, 255), 2)
        cv2.circle(frame, p2, 3, (0, 0, 255), -1)
        cv2.circle(frame, p3, 10, (0, 0, 255), 2)
        cv2.circle(frame, p3, 3, (0, 0, 255), -1)
        
        R = [p1,p2,p3]
        
    return [L,R]

def Workout():
    global w, Counter, Workouts, ALLDONE, Flags
    items = []
    for workout in Workouts:
        items.append(str(workout[0]) + " " + str(workout[1]) + " reps " + workout[2])
    
    draw_todo_list(items)
    
    workout = Workouts[w]
    Shoulders = DrawShoulders(frame, results)
    DrawFullBody(frame, results)
    
    if (Shoulders[0] or Shoulders[1]):
        Techniques(workout[0], Shoulders)
    if (Counter == workout[1]):
        print(workout[0], " Done")
        Workouts[w][2] = "Done"
        Flags = [False] * 6
        if (w + 1 < len(Workouts)):
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
    global Flags, Counter

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

    if (Flags[0] == True and Flags[1] == True):
        Flags[4] = True

    if (Flags[2] == True and Flags[3] == True):
        Flags[5] = True

    if (Flags[4] == True and Flags[5] == True):
        Flags = [False] * 6
        Counter += 0.5

def ShouldersPress(Shoulders):
    global Flags, Counter

    p1, p2, p3 = Shoulders[0]
    angle = calculate_angle(p1, p2, p3)

    if (p1[1] < p3[1]):
        Color = (0, 0, 255)

    else:
        Color = (0, 255, 0)
    cv2.line(frame, p1, p2, Color, 4)
    cv2.line(frame, p3, p2, Color, 4)

    if (70 <= angle <= 100):
        Flags[0] = True

    if (150 <= angle <= 180):
        Flags[1] = True

    p1, p2, p3 = Shoulders[1]
    angle = calculate_angle(p1, p2, p3)

    if (p1[1] < p3[1]):
        Color = (0, 0, 255)
    else:
        Color = (0, 255, 0)

    cv2.line(frame, p1, p2, Color, 4)
    cv2.line(frame, p3, p2, Color, 4)

    padding = 100

    if (70 <= angle <= 100):
        Flags[2] = True

    if (150 <= angle <= 180):
        Flags[3] = True

    if (Flags[0] == True and Flags[1] == True):
        Flags[4] = True

    if (Flags[2] == True and Flags[3] == True):
        Flags[5] = True

    if (Flags[4] == True and Flags[5] == True):
        Flags = [False] * 6
        Counter += 0.5

def PushUp(Shoulders):
    global Flags, Counter

    # Implement push-up technique
    # Check if the body is in the correct position for a push-up
    # For example, you can use the angle between the shoulders and hips
    p1, p2, p3 = Shoulders[0]  # Left shoulder landmarks
    angle_left = calculate_angle(p1, p2, p3)

    p1, p2, p3 = Shoulders[1]  # Right shoulder landmarks
    angle_right = calculate_angle(p1, p2, p3)

    # Assuming correct position if angles are within a range (e.g., 70-100 degrees)
    if 70 <= angle_left <= 100 and 70 <= angle_right <= 100:
        Flags[0] = True
        Flags[1] = True

    # Check if both shoulders are in the correct position
    if all(Flags[0:2]):
        Flags[4] = True  # Both shoulders are in the correct position

    # Check if the push-up is completed
    if Flags[4]:
        Flags = [False] * 6  # Reset flags
        Counter += 1  # Increment the counter

def Squat(Shoulders):
    global Flags, Counter

    # Implement squat technique
    # Check if the body is in the correct position for a squat
    # For example, you can use the angle between the hips and knees
    p1, p2, p3 = Shoulders[0]  # Left hip, knee, ankle landmarks
    angle_left = calculate_angle(p1, p2, p3)

    p1, p2, p3 = Shoulders[1]  # Right hip, knee, ankle landmarks
    angle_right = calculate_angle(p1, p2, p3)

    # Assuming correct position if angles are within a range (e.g., 100-160 degrees)
    if 100 <= angle_left <= 160 and 100 <= angle_right <= 160:
        Flags[2] = True
        Flags[3] = True

    # Check if both hips are in the correct position
    if all(Flags[2:4]):
        Flags[5] = True  # Both hips are in the correct position

    # Check if the squat is completed
    if Flags[5]:
        Flags = [False] * 6  # Reset flags
        Counter += 1  # Increment the counter

def draw_todo_list(items):
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
    if (not ALLDONE):
        cv2.putText(frame, "Count =" + str(int(Counter)), (20, 60 + (len(items)) * line_spacing), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    else:
        cv2.rectangle(frame, (50, 190), (590, 290), (255, 255, 255), -1)
        cv2.putText(frame, "All Done - Good Work", (55, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Holistic
    results = holistic.process(frame_rgb)

    Workout()

    # Display the frame
    cv2.imshow('MediaPipe Holistic', frame)

    # Check for exit key
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
