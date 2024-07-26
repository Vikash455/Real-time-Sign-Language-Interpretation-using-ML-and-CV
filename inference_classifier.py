import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize the video capture
cap = cv2.VideoCapture(0)  # Ensure this is the correct index for your camera

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

while True:
    data_aux = []
    x_ = []
    y_ = []
    z_ = []

    ret, frame = cap.read()

    if not ret or frame is None:
        print("Failed to capture frame from camera. Check camera index and connection.")
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z

                x_.append(x)
                y_.append(y)
                z_.append(z)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z

                # Normalize x, y, z coordinates
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
                data_aux.append(z - min(z_))

        # Ensure the feature vector has the correct number of elements
        if len(data_aux) != model.n_features_in_:
            # Padding or trimming the data_aux to the expected length
            if len(data_aux) < model.n_features_in_:
                data_aux.extend([0] * (model.n_features_in_ - len(data_aux)))
            else:
                data_aux = data_aux[:model.n_features_in_]

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Predict gesture
        prediction = model.predict([np.asarray(data_aux)])
        predicted_label = labels_dict[int(prediction[0])]

        #for confidence score
        confidence_score = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([np.asarray(data_aux)])
            confidence_score = np.max(probabilities)

        if confidence_score is not None:
            text = f"{predicted_label} ({confidence_score * 100:.2f})"
        else:
            text = f"{predicted_label}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()

# for accuracy score 
a = int(input("Enter number of Correct Interpretations: "))
b = int(input("Enter Total number of Correct Interpretations: "))

if b != 0:
    percentage = (a / b) * 100
    print(f"{a} is {percentage:.2f}% of {b}.")
else:
    print("Cannot divide by zero. Please enter a non-zero value for b.")


