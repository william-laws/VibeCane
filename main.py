#!/usr/bin/env python3
"""
AI Tracking Client (ai.py)

Tracks an object (ex: Tumor) in your camera and sends movement commands to a robot.

How to use:
1. Press 'a' + Enter → turn AI tracking ON/OFF
2. Press 'q' + Enter → quit
3. Robot stops if you press X on PS4 controller.

the thing uses bluetooth in this case, so it has to sleep and make sure frame rate/computation speed isn't too fast.
"""

import os

# Suppress inference optional-model warnings (we only use OWLv2-finetuned / Instant)
os.environ.setdefault("QWEN_2_5_ENABLED", "False")
os.environ.setdefault("QWEN_3_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_SAM_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_SAM3_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_GAZE_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_YOLO_WORLD_ENABLED", "False")

import cv2
import numpy as np
import socket
import time
from datetime import datetime
try:
    from inference.models.utils import get_model
except ImportError:
    try:
        from inference import get_model
    except ImportError:
        raise ImportError(
            "Cannot import 'get_model'. Install Roboflow's inference package:\n"
            "  pip install 'inference[transformers]'\n"
            "For Instant/OWLv2-finetuned models (e.g. find-floors), [transformers] is required."
        ) from None


TRACK_CLASS       = ["trashcan", "wet-floor"]                      
AUTO_TRACK        = False                        
ROBOT_AI_SPEED    = 20                          
AI_SPEED_INC      = 4                           

CONFIDENCE_THRESH = 0.80                        
IOU_THRESH        = 0.65                        
TOLERANCE_PER     = 0.10                        

HORIZ_OFFSET_PX = 50                            
VERT_OFFSET_PX  = -30                           

RF_MODEL_ID       = "find-floors-v2/4"   # Roboflow model
RF_API_KEY        = "2rIoMmHR2ZDZE0btEav0"      # Roboflow API key

CAMERA_PORT       = 0                          
RECORD_VIDEO      = False                       
CAMERA_VFOV_DEG   = 60.0                        
CAMERA_PITCH_DEG  = 15.0                        
CAMERA_HEIGHT     = 0.3                        
ARROW_LENGTH_M    = 1                        

ROBOT_HOST        = "cane.local"           
ROBOT_ON          = False                        

#setup
model = get_model(model_id=RF_MODEL_ID, api_key=RF_API_KEY)

def predictions_from_inference(result):
    """Convert native inference result to our preds format: list of {class, points}."""
    preds = []
    out = result[0] if isinstance(result, list) else result
    for p in (out.predictions if hasattr(out, "predictions") else []):
        cname = getattr(p, "class_name", None) or getattr(p, "class", "Unknown")
        x, y = float(getattr(p, "x", 0)), float(getattr(p, "y", 0))
        w, h = float(getattr(p, "width", 0)), float(getattr(p, "height", 0))
        x0, y0 = x - w / 2, y - h / 2
        pts = [{"x": x0, "y": y0}, {"x": x0 + w, "y": y0}, {"x": x0 + w, "y": y0 + h}, {"x": x0, "y": y0 + h}]
        preds.append({"class": cname, "points": pts})
    return preds

cap = cv2.VideoCapture(CAMERA_PORT)
FW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if RECORD_VIDEO:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = cv2.VideoWriter(
        f"record_{ts}.mp4",
        cv2.VideoWriter_fourcc(*"XVID"),
        7.0,
        (FW, FH)
    )
else:
    out = None

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ROBOT_ADDR = (ROBOT_HOST, 5005)

if ROBOT_ON and AUTO_TRACK:
    sock.sendto(f"ai_speed {ROBOT_AI_SPEED}".encode(), ROBOT_ADDR)
    sock.sendto(f"ai_speed_inc {AI_SPEED_INC/100}".encode(), ROBOT_ADDR)

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

#arrow func
def _camera_params(fw, fh, vfov_deg):
    """Focal length and principal point from image size and vertical FOV."""
    vfov_rad = np.deg2rad(vfov_deg)
    fy = fh / (2.0 * np.tan(vfov_rad / 2.0))
    hfov_rad = 2.0 * np.arctan((fw / fh) * np.tan(vfov_rad / 2.0))
    fx = fw / (2.0 * np.tan(hfov_rad / 2.0))
    cx, cy = fw / 2.0, fh / 2.0
    return fx, fy, cx, cy

def ground_point_to_image(x, z, pitch_rad, h, fx, fy, cx, cy):
    """Project a 3D point on the ground (world: x right, z forward, y=0) into image coords.
    Camera at (0, h, 0), looking forward with pitch down from horizontal."""
    cam_x = x
    cam_y = h * np.cos(pitch_rad) - z * np.sin(pitch_rad)
    cam_z = h * np.sin(pitch_rad) + z * np.cos(pitch_rad)
    if cam_z <= 0.01:
        return None
    u = fx * cam_x / cam_z + cx
    v = fy * cam_y / cam_z + cy
    return (u, v)

def estimate_pitch_from_frame(frame, fy, cy):
    """Estimate camera pitch from frame by finding horizon-like horizontal lines (adaptive to scene)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=60, maxLineGap=20)
    if lines is None or len(lines) == 0:
        return None
    horizon_candidates = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx * dx + dy * dy)
        if length < 40:
            continue
        angle = np.abs(np.arctan2(dy, dx))
        # Horizontal: angle near 0 or near pi
        if angle < np.deg2rad(25) or angle > np.deg2rad(155):
            # Use midpoint y as horizon candidate
            my = (y1 + y2) / 2.0
            horizon_candidates.append(my)
    if not horizon_candidates:
        return None
    v_horizon = float(np.median(horizon_candidates))
    # v_horizon = cy - fy*tan(pitch) => pitch = atan((cy - v_horizon) / fy)
    pitch_rad = np.arctan2(cy - v_horizon, fy)
    pitch_rad = np.clip(pitch_rad, np.deg2rad(5), np.deg2rad(70))
    return pitch_rad

def project_ground_arrow(frame, pitch_rad, fw, fh, vfov_deg, h_m, length_m):
    """Draw a 3D forward arrow on the ground, projected into the image."""
    fx, fy, cx, cy = _camera_params(fw, fh, vfov_deg)
    w_shaft = length_m * 0.15
    w_tip = length_m * 0.04
    head_len = length_m * 0.25
    # 3D points on ground: (x, z) in meters. Forward = +z.
    # Shaft: rectangle from (0,0) to (0, length_m - head_len) with width
    z_base = 0.05
    z_tip_shaft = length_m - head_len
    z_tip_head = length_m
    pts_3d_shaft = [
        (-w_shaft, z_base), (w_shaft, z_base),
        (w_tip, z_tip_shaft), (-w_tip, z_tip_shaft)
    ]
    pts_3d_head = [(0, z_tip_head), (-w_tip * 1.5, z_tip_shaft), (w_tip * 1.5, z_tip_shaft)]
    color_fill = (0, 255, 255)
    color_edge = (0, 180, 255)
    for pts_3d in (pts_3d_shaft, pts_3d_head):
        pts_2d = []
        valid = True
        for (x, z) in pts_3d:
            uv = ground_point_to_image(x, z, pitch_rad, h_m, fx, fy, cx, cy)
            if uv is None:
                valid = False
                break
            pts_2d.append(uv)
        if not valid or len(pts_2d) < 3:
            continue
        pts_2d = np.array(pts_2d, dtype=np.int32)
        cv2.fillPoly(frame, [pts_2d], color_fill)
        cv2.polylines(frame, [pts_2d], True, color_edge, 1)

# loop
ai_mode = False          
running = True
lock = False
last_ai_mode_sent = False  
show_forward_arrow = False  

def ensure_ai_on():
    global last_ai_mode_sent
    if ROBOT_ON and AUTO_TRACK and not last_ai_mode_sent:
        sock.sendto(b"ai_mode on", ROBOT_ADDR)
        last_ai_mode_sent = True

def ensure_ai_off():
    global last_ai_mode_sent
    if ROBOT_ON and last_ai_mode_sent:
        sock.sendto(b"move 0 0", ROBOT_ADDR)
        sock.sendto(b"ai_mode off", ROBOT_ADDR)
        last_ai_mode_sent = False

print("Press 'a' to toggle AI mode, 'p' for forward arrow, 'q' to quit.")

while running:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    if ai_mode:
        res = model.infer(frame, confidence=CONFIDENCE_THRESH, iou_threshold=IOU_THRESH)
        preds = predictions_from_inference(res)

        for det in preds:
            pts = det["points"]
            contour = np.array(
                [[int(p["x"]), int(p["y"])] for p in pts],
                dtype=np.int32
            ).reshape((-1,1,2))
            cv2.polylines(frame, [contour], True, (0,255,0), 2)
            lx, ly = int(pts[0]["x"]), int(pts[0]["y"]) - 10
            cv2.putText(
                frame,
                det.get("class","Unknown"),
                (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,0,0),
                2
            )

        candidates = [d for d in preds if d.get("class") == TRACK_CLASS]
        if candidates:
            all_pts = []
            for det in candidates:
                all_pts.extend(det["points"])
            xs = [p["x"] for p in all_pts]
            ys = [p["y"] for p in all_pts]
            cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)

            cx_adj = cx + HORIZ_OFFSET_PX
            cy_adj = cy + VERT_OFFSET_PX

            dx = (cx_adj - FW/2) / (FW/2)
            dy = (cy_adj - FH/2) / (FH/2)

            if abs(dx) <= TOLERANCE_PER:
                dx = 0
            if abs(dy) <= TOLERANCE_PER:
                dy = 0

            if dx == 0 and dy == 0:
                if not lock:
                    print("Target Locked")
                    lock = True
                cmd = "move 0 0"
            else:
                lock = False
                cmd = f"move {dx} {dy}"
                print(f"dx: {dx*100:.2f}%, dy: {dy*100:.2f}%")

            if ROBOT_ON and AUTO_TRACK:
                ensure_ai_on()
                sock.sendto(cmd.encode(), ROBOT_ADDR)

        # deleted for now to avoid exiting ai mode
        # else:
        #     # No detections
        #     if AUTO_TRACK:
        #         ensure_ai_off()
        #     ai_mode = False
        #     print("AI mode: OFF (no detections)")

        time.sleep(0.05)

    if show_forward_arrow:
        _, fy_est, _, cy_est = _camera_params(FW, FH, CAMERA_VFOV_DEG)
        pitch_rad = estimate_pitch_from_frame(frame, fy_est, cy_est)
        if pitch_rad is None:
            pitch_rad = np.deg2rad(CAMERA_PITCH_DEG)
        project_ground_arrow(
            frame, pitch_rad, FW, FH,
            CAMERA_VFOV_DEG, CAMERA_HEIGHT, ARROW_LENGTH_M
        )

    if RECORD_VIDEO and out:
        out.write(frame)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        running = False
        ensure_ai_off()
    elif key == ord('a'):
        ai_mode = not ai_mode
        print("AI mode:", "ON" if ai_mode else "OFF")
        if not ai_mode:
            ensure_ai_off()
        else:
            if ROBOT_ON and AUTO_TRACK:
                sock.sendto(f"ai_speed {ROBOT_AI_SPEED}".encode(), ROBOT_ADDR)
                sock.sendto(f"ai_speed_inc {AI_SPEED_INC/100}".encode(), ROBOT_ADDR)
                ensure_ai_on()
    elif key == ord('p'):
        show_forward_arrow = not show_forward_arrow
        print("Forward arrow:", "ON" if show_forward_arrow else "OFF")

#cleanup
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
sock.close()
