#!/usr/bin/env python3

import time
import cv2
import numpy as np
import libcamera
from libcamera import controls
from libcamera.utils import MappedFrameBuffer
from ultralytics import YOLO
import torch


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "/home/vardhan/yolo11n.pt"

WIDTH = 800
HEIGHT = 450

YOLO_SIZE = 320
CONFIDENCE = 0.15
IOU = 0.45

FRAME_DURATION = 33333

LEFT_LIMIT = 0.35
RIGHT_LIMIT = 0.65

MOVEMENT_THRESHOLD = 18
AREA_CHANGE_THRESHOLD = 0.12


# ============================================================
# CPU SETTINGS
# ============================================================

torch.set_num_threads(4)

try:
    torch.set_num_interop_threads(1)
except Exception:
    pass


# ============================================================
# LOAD YOLO
# ============================================================

print()
print("========================================")
print("       AMR YOLO VISION SYSTEM")
print("========================================")
print()

print("Loading YOLO11n...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully")
print("Classes:", model.names)
print()


# ============================================================
# CAMERA MANAGER
# ============================================================

cm = libcamera.CameraManager.singleton()

cameras = cm.cameras

if not cameras:
    raise RuntimeError("No camera detected")

camera = cameras[0]

print("Camera:", camera.id)


# ============================================================
# ACQUIRE CAMERA
# ============================================================

camera.acquire()


# ============================================================
# CONFIGURE CAMERA
# ============================================================

config = camera.generate_configuration(
    [libcamera.StreamRole.Viewfinder]
)

stream_config = config.at(0)

stream_config.size = libcamera.Size(
    WIDTH,
    HEIGHT
)

stream_config.pixel_format = libcamera.formats.XRGB8888

config.validate()

camera.configure(config)

stream = stream_config.stream

print(
    f"Configured: {WIDTH}x{HEIGHT}-XRGB8888"
)


# ============================================================
# BUFFER ALLOCATION
# ============================================================

allocator = libcamera.FrameBufferAllocator(camera)

allocator.allocate(stream)

buffers = allocator.buffers(stream)

print("Camera buffers:", len(buffers))


# ============================================================
# CREATE REQUESTS
# ============================================================

requests = []

for buffer in buffers:

    request = camera.create_request()

    if request is None:
        raise RuntimeError(
            "camera.create_request() failed"
        )

    request.add_buffer(
        stream,
        buffer
    )

    try:
        request.set_control(
            controls.FrameDurationLimits,
            [
                FRAME_DURATION,
                FRAME_DURATION
            ]
        )
    except Exception:
        pass

    requests.append(request)


print("Camera requests:", len(requests))


# ============================================================
# START CAMERA
# ============================================================

camera.start()

for request in requests:
    camera.queue_request(request)

print()
print("========================================")
print("CAMERA STARTED")
print("Resolution :", WIDTH, "x", HEIGHT)
print("Target FPS :", 30)
print("YOLO size  :", YOLO_SIZE)
print("Confidence :", CONFIDENCE)
print("========================================")
print()


# ============================================================
# TRACKING
# ============================================================

previous_cx = None
previous_area = None
previous_label = None

yolo_count = 0

camera_frame_count = 0
fps_count = 0
fps_start = time.time()


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # GET COMPLETED CAMERA REQUEST
        # ----------------------------------------------------

        ready_requests = cm.get_ready_requests()

        if not ready_requests:

            time.sleep(0.001)
            continue

        request = ready_requests[-1]


        # ----------------------------------------------------
        # GET IMAGE
        # ----------------------------------------------------

        buffer = request.buffers[stream]

        with MappedFrameBuffer(buffer) as mapped:

            data = mapped.planes[0]

            frame = np.frombuffer(
                data,
                dtype=np.uint8
            ).reshape(
                (HEIGHT, WIDTH, 4)
            )

            # XRGB8888 -> BGR
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGRA2BGR
            )

            frame = frame.copy()


        camera_frame_count += 1
        fps_count += 1


        # ----------------------------------------------------
        # REUSE REQUEST
        # ----------------------------------------------------

        request.reuse()

        camera.queue_request(request)


        # ====================================================
        # YOLO INFERENCE
        # ====================================================

        start = time.time()

        results = model.predict(
            frame,
            imgsz=YOLO_SIZE,
            conf=CONFIDENCE,
            iou=IOU,
            verbose=False,
            device="cpu"
        )

        yolo_time = time.time() - start

        yolo_count += 1


        # ====================================================
        # DETECTIONS
        # ====================================================

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                confidence = float(
                    box.conf[0]
                )

                class_id = int(
                    box.cls[0]
                )

                label = model.names[class_id]

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                width = x2 - x1
                height = y2 - y1

                area = max(
                    width * height,
                    1
                )

                cx = (x1 + x2) / 2

                detections.append({
                    "label": label,
                    "confidence": confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": cx,
                    "area": area
                })


        # ====================================================
        # DRAW DETECTIONS
        # ====================================================

        for d in detections:

            cv2.rectangle(
                frame,
                (d["x1"], d["y1"]),
                (d["x2"], d["y2"]),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f'{d["label"]} {d["confidence"]:.2f}',
                (
                    d["x1"],
                    max(d["y1"] - 10, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 0),
                2
            )


        # ====================================================
        # SELECT MAIN OBJECT
        # ====================================================

        selected = None

        if detections:

            persons = [
                d for d in detections
                if d["label"] == "person"
            ]

            if persons:

                selected = max(
                    persons,
                    key=lambda d: d["area"]
                )

            else:

                selected = max(
                    detections,
                    key=lambda d: d["area"]
                )


        # ====================================================
        # DEFAULT STATE
        # ====================================================

        decision = "FORWARD"
        position = "NONE"
        movement = "STATIONARY"
        trend = "UNKNOWN"


        # ====================================================
        # PROCESS OBJECT
        # ====================================================

        if selected is not None:

            label = selected["label"]
            cx = selected["cx"]
            area = selected["area"]


            # ------------------------------------------------
            # POSITION
            # ------------------------------------------------

            x_ratio = cx / WIDTH

            if x_ratio < LEFT_LIMIT:

                position = "LEFT"

            elif x_ratio > RIGHT_LIMIT:

                position = "RIGHT"

            else:

                position = "CENTER"


            # ------------------------------------------------
            # MOVEMENT
            # ------------------------------------------------

            if (
                previous_cx is not None
                and previous_label == label
            ):

                dx = cx - previous_cx

                if dx > MOVEMENT_THRESHOLD:

                    movement = "MOVING RIGHT"

                elif dx < -MOVEMENT_THRESHOLD:

                    movement = "MOVING LEFT"

                else:

                    movement = "STATIONARY"


            # ------------------------------------------------
            # APPROACHING / MOVING AWAY
            # ------------------------------------------------

            if (
                previous_area is not None
                and previous_label == label
            ):

                change = (
                    area - previous_area
                ) / max(
                    previous_area,
                    1
                )

                if change > AREA_CHANGE_THRESHOLD:

                    trend = "APPROACHING"

                elif change < -AREA_CHANGE_THRESHOLD:

                    trend = "MOVING AWAY"

                else:

                    trend = "STABLE"


            # ------------------------------------------------
            # ROBOT DECISION
            # ------------------------------------------------

            if trend == "APPROACHING":

                decision = "STOP"

            elif position == "CENTER":

                decision = "STOP"

            elif position == "LEFT":

                decision = "MOVE RIGHT"

            elif position == "RIGHT":

                decision = "MOVE LEFT"

            else:

                decision = "FORWARD"


            # ------------------------------------------------
            # UPDATE TRACKING
            # ------------------------------------------------

            previous_cx = cx
            previous_area = area
            previous_label = label


        else:

            previous_cx = None
            previous_area = None
            previous_label = None


        # ====================================================
        # DECISION COLOR
        # ====================================================

        if decision == "STOP":

            decision_color = (
                0,
                0,
                255
            )

        elif decision in [
            "MOVE LEFT",
            "MOVE RIGHT"
        ]:

            decision_color = (
                0,
                165,
                255
            )

        else:

            decision_color = (
                0,
                255,
                0
            )


        # ====================================================
        # INFORMATION PANEL
        # ====================================================

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (410, 225),
            (0, 0, 0),
            -1
        )

        frame = cv2.addWeighted(
            overlay,
            0.55,
            frame,
            0.45,
            0
        )


        cv2.putText(
            frame,
            f"DECISION: {decision}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            decision_color,
            3
        )


        if selected is not None:

            object_text = (
                f"OBJECT: {selected['label']}"
            )

            confidence_text = (
                f"CONF: {selected['confidence']:.2f}"
            )

        else:

            object_text = "OBJECT: NONE"
            confidence_text = "CONF: --"


        cv2.putText(
            frame,
            object_text,
            (15, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            confidence_text,
            (15, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"POSITION: {position}",
            (15, 126),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"MOVEMENT: {movement}",
            (15, 154),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"TREND: {trend}",
            (15, 182),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"YOLO: {yolo_time:.2f}s",
            (15, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        # ====================================================
        # LEFT / CENTER / RIGHT GUIDE LINES
        # ====================================================

        left_line = int(
            WIDTH * LEFT_LIMIT
        )

        right_line = int(
            WIDTH * RIGHT_LIMIT
        )

        cv2.line(
            frame,
            (left_line, 0),
            (left_line, HEIGHT),
            (255, 255, 0),
            1
        )

        cv2.line(
            frame,
            (right_line, 0),
            (right_line, HEIGHT),
            (255, 255, 0),
            1
        )


        cv2.putText(
            frame,
            "LEFT",
            (25, HEIGHT - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "CENTER",
            (
                WIDTH // 2 - 45,
                HEIGHT - 15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "RIGHT",
            (
                WIDTH - 100,
                HEIGHT - 15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 0),
            2
        )


        # ====================================================
        # YOLO COUNT
        # ====================================================

        cv2.putText(
            frame,
            f"YOLO #{yolo_count}",
            (
                WIDTH - 130,
                30
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        # ====================================================
        # PROCESSING FPS
        # ====================================================

        now = time.time()

        if now - fps_start >= 2.0:

            fps = (
                fps_count /
                (now - fps_start)
            )

            print(
                f"PROCESSING FPS: {fps:.2f}"
            )

            fps_count = 0
            fps_start = now


        # ====================================================
        # TERMINAL OUTPUT
        # ====================================================

        if selected is not None:

            print(
                f"YOLO #{yolo_count}: "
                f"{yolo_time:.2f}s | "
                f"{selected['label']} "
                f"{selected['confidence']:.2f} | "
                f"{position} | "
                f"{movement} | "
                f"{trend} | "
                f"{decision}"
            )

        else:

            print(
                f"YOLO #{yolo_count}: "
                f"{yolo_time:.2f}s | "
                f"NO DETECTION | "
                f"{decision}"
            )


        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "IMX477 + YOLO11n - AMR VISION",
            frame
        )


        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


# ============================================================
# CLEANUP
# ============================================================

except KeyboardInterrupt:

    print("\nStopping...")


finally:

    try:
        camera.stop()
    except Exception:
        pass

    try:
        camera.release()
    except Exception:
        pass

    cv2.destroyAllWindows()

    print("Camera stopped")
    print("YOLO stopped")
