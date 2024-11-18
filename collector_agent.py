import cv2
import os
import time


if not os.path.exists("frames"):
    os.makedirs("frames")

def start_camera():
    cap = cv2.VideoCapture(0)  # 0 for the default camera
    
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Camera started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Preprocess: Resize for faster processing
        frame_resized = cv2.resize(frame, (640, 480))

        # Display the frame
        cv2.imshow("Live Feed", frame_resized)

        # Save the frame with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(f"frames/frame_{timestamp}.jpg", frame_resized)

        # Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting surveillance...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_camera()
