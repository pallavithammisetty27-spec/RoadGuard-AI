import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
from inference_sdk import InferenceHTTPClient

# Roboflow Client Setup
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="vqXPJrPdpCdnUCm9TzTM"
)

st.set_page_config(page_title="Smart Pothole Detection", layout="wide")

st.title("🚨 Smart Pothole Detection")
st.markdown("App is open ,View the road in the camera!")

class PotholeTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Detection
        result = client.infer(img, model_id="find-pothole-vfind-pothole-5076e-3-yolo11s-t1-logic")
        
        # Potholes count
        predictions = result.get("predictions", [])
        count = len(predictions)
        
        # Screen meeda Results
        cv2.putText(img, f"Potholes Detected: {count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        return img

# Direct Camera Stream
webrtc_streamer(
    key="example",
    video_transformer_factory=PotholeTransformer,
    media_stream_constraints={"video": True, "audio": False}
)