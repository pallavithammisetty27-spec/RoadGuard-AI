import cv2
import supervision as sv
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="vqXPJrPdpCdnUCm9TzTM"
)

# Camera index 0 badhulu 1 try chedam, leda DirectShow backend vadudam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Frame ni temporary ga save chesi workflow ki pampdam
    cv2.imwrite("temp_frame.jpg", frame)

    result = client.run_workflow(
        workspace_name="pallavi-esuec",
        workflow_id="find-pothole-vfind-pothole-5076e-3-yolo11s-t1-logic",
        images={
            "image": "temp_frame.jpg"
        },
        use_cache=True
    )

    predictions = result[0]["predictions"]
    detections = sv.Detections.from_inference(predictions)

    annotated_image = box_annotator.annotate(scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    cv2.imshow("Live Pothole Detection", annotated_image)

    # Keyboard lo 'q' press chesthe live video close avuthundi
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()