import cv2
import numpy as np
import streamlit as st

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_spots_wrinkles_texture_pores(frame):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    # Loop over the faces detected in the frame
    for (x, y, w, h) in faces:
        # Draw a bounding box around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the face from the frame
        face_image = frame[y:y + h, x:x + w]

        # Apply a Gaussian blur to the face image to smoothen it
        blurred_face_image = cv2.GaussianBlur(face_image, (5, 5), 0)

        # Convert the face image to single channel
        gray_face_image = cv2.cvtColor(blurred_face_image, cv2.COLOR_BGR2GRAY)

        # Detect spots in the face image
        spots = cv2.threshold(gray_face_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Increment the spot counter for each spot detected
        spot_count = np.count_nonzero(spots)

        # Detect wrinkles in the face image
        wrinkles = cv2.HoughLinesP(gray_frame, 1, np.pi / 180, 100, minLineLength=10, maxLineGap=10)

        # Increment the wrinkle counter for each wrinkle detected
        if wrinkles is not None:
            wrinkle_count = len(wrinkles)
        else:
            wrinkle_count = 0

        # Detect texture in the face image
        texture = cv2.threshold(gray_face_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Increment the texture counter for each texture detected
        texture_count = np.count_nonzero(texture)

        # Detect pores in the face image
        pores = cv2.threshold(gray_face_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Increment the pore counter for each pore detected
        pore_count = np.count_nonzero(pores)

        # Add text labels for counts within the bounding box
        label = f"Spots: {spot_count}"
        cv2.putText(frame, label, (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        label = f"Wrinkles: {wrinkle_count}"
        cv2.putText(frame, label, (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        label = f"Texture: {texture_count}"
        cv2.putText(frame, label, (x + 10, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        label = f"Pores: {pore_count}"
        cv2.putText(frame, label, (x + 10, y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

def main():
    st.title("Real-time Spot, Wrinkle, Texture, and Pore Detector")

    # Start the camera
    camera = cv2.VideoCapture(0)

    #st.write("To stop the camera, press STOP, and to rerun it, from the right side hamburger button press RERUN to strat it again")

    video_stream = st.empty()

    while True:
        # Capture a frame from the camera
        success, frame = camera.read()

        # Detect the spots, wrinkles, texture, and pores in the frame
        frame = detect_spots_wrinkles_texture_pores(frame)

        # Display the frame with the face detections and spot/wrinkle/texture/pore count
        video_stream.image(frame, channels="BGR", use_column_width=True, output_format="JPEG")

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
