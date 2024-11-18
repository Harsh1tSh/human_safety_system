from flask import Flask,jsonify,  send_from_directory, Response
from detection_agent.detector import detect_objects, draw_detected_objects
import cv2
import os
import time

app = Flask(__name__)

@app.route("/")
def home():
    return "Surveillance Agent is Running"

@app.route("/get_latest_frame", methods=["GET"])
def get_latest_frame():
    frame_folder = "frames"
    if not os.path.exists(frame_folder):
        return jsonify({"error": "No frames found"}), 404

    files = sorted(os.listdir(frame_folder), reverse=True)
    if not files:
        return jsonify({"error": "No frames available"}), 404

    latest_frame = files[0]
    
    # Set custom headers to prevent caching
    from flask import Response
    response = send_from_directory(frame_folder, latest_frame)
    response.cache_control.no_cache = True
    response.cache_control.max_age = 0
    return response

@app.route("/live_feed", methods=["GET"])
def live_feed():
    frame_folder = "frames"
    
    def generate_frames():
        while True:
            files = sorted(os.listdir(frame_folder), reverse=True)
            if files:
                latest_frame = files[0]
                frame_path = os.path.join(frame_folder, latest_frame)
                frame = cv2.imread(frame_path)

                # Run object detection on the frame
                boxes, confidences, class_ids, indexes = detect_objects(frame)
                frame = draw_detected_objects(frame, boxes, confidences, class_ids, indexes)

                _, jpeg = cv2.imencode('.jpg', frame)
                frame_bytes = jpeg.tobytes()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(1)  # Add a slight delay to avoid overwhelming the browser

    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
