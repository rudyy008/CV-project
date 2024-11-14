import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        fingerCount = 0  # Reset finger count

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get hand index to check label (left or right)
                handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                handLabel = results.multi_handedness[handIndex].classification[0].label

                # Set variable to keep landmarks positions (x and y)
                handLandmarks = []
                for landmarks in hand_landmarks.landmark:
                    handLandmarks.append([landmarks.x, landmarks.y])

                # Thumb (needs a specific check based on hand label)
                if handLabel == "Left":
                    if handLandmarks[4][0] > handLandmarks[3][0]:
                        fingerCount += 1
                elif handLabel == "Right":
                    if handLandmarks[4][0] < handLandmarks[3][0]:
                        fingerCount += 1

                # Other fingers (Index, Middle, Ring, Pinky)
                if handLandmarks[8][1] < handLandmarks[6][1]:  # Index
                    fingerCount += 1
                if handLandmarks[12][1] < handLandmarks[10][1]:  # Middle
                    fingerCount += 1
                if handLandmarks[16][1] < handLandmarks[14][1]:  # Ring
                    fingerCount += 1
                if handLandmarks[20][1] < handLandmarks[18][1]:  # Pinky
                    fingerCount += 1

                # Draw hand landmarks and connections
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # Display finger count (on the screen)
        cv2.putText(image, str(fingerCount), (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)

        # Display image
        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

cap.release()
cv2.destroyAllWindows()
