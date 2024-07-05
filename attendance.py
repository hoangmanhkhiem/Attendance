import cv2
import face_recognition
import os
import datetime

# Load the known faces and their names from the faces directory
known_faces = []
known_names = []
faces_dir = "faces"  # Directory containing the face images
for name in os.listdir(faces_dir):
    known_names.append(name.split('.')[0])
    face_image = face_recognition.load_image_file(os.path.join(faces_dir, name))
    face_encoding = face_recognition.face_encodings(face_image)[0]
    known_faces.append(face_encoding)

# Initialize variables for attendance
present_names = []
absent_names = []

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Capture frame from camera
    if not ret:
        print("Failed to capture frame from camera")
        break

    # Convert the frame from BGR to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through detected faces
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the detected face encoding with known face encodings
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        # Calculate face recognition percentage
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        min_distance = min(face_distances)
        percentage = (1 - min_distance) * 100

        if any(matches):
            # Assign the name of the known face with the highest percentage
            best_match_index = face_distances.argmin()
            if percentage >= 50:
                name = known_names[best_match_index]
                if name not in present_names:
                    present_names.append(name)
            else:
                name = f"Maybe {known_names[best_match_index]}"

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display the name and percentage of the detected face
        cv2.putText(frame, f"{name} ({percentage:.2f}%)", (left, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame with detected faces
    cv2.imshow('Face Recognition', frame)

    # Check if it's after 5pm
    now = datetime.datetime.now()
    if now.hour >= 17:
        # Write absent students to file
        for name in known_names:
            if name not in present_names:
                absent_names.append(name)
        if len(absent_names) > 0:
            filename = f"vang_{now.day}_{now.month}_{now.year}.txt"
            with open(filename, 'w') as f:
                for name in absent_names:
                    f.write(name + '\n')
            print(f"Absent students written to {filename}")
        else:
            print("All students present")

        # Reset attendance for next day
        present_names = []
        absent_names = []

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
