# Holistic Diagram Using AI - System Design

## What It Does
A collection of real-time body analysis demos using Google MediaPipe. Each script
demonstrates a different computer vision capability: face detection, face mesh, pose
estimation, hand tracking, iris tracking, selfie segmentation, head posture, facial
expression, and the full holistic model -- all live on webcam.

---

## Architecture

```
Webcam (live video stream)
        |
        v
+--------------------------------------------------+
|  OpenCV VideoCapture (video.py / videosource.py) |
|  Frame-by-frame capture at camera FPS            |
+--------------------------------------------------+
        |
        v
+--------------------------------------------------+
|  MediaPipe Model (per script)                    |
|                                                  |
|  holistic.py       -> face + hands + pose        |
|  face_detection.py -> face bounding box          |
|  face_mesh.py      -> 468 face landmarks         |
|  hands.py          -> 21 hand landmarks          |
|  handstrack.py     -> hand tracking + overlay    |
|  pose.py           -> 33 body pose landmarks     |
|  iris.py           -> iris landmarks + depth     |
|  head_posture.py   -> pitch / yaw / roll angles  |
|  facial_expression -> expression classification  |
|  selfie_segmentation -> background removal       |
|  objectron.py      -> 3D object bounding box     |
+--------------------------------------------------+
        |
  Landmark data (x, y, z coordinates)
        |
+--------------------------------------------------+
|  custom/ (geometry utilities)                    |
|  custom/core.py          -> math helpers         |
|  custom/face_geometry.py -> 3D face model        |
|  custom/iris_lm_depth.py -> iris depth estimate  |
+--------------------------------------------------+
        |
  Annotated frame -> cv2.imshow() -> display
```

---

## What Each Script Detects

| Script                | Model               | Output                              |
|-----------------------|---------------------|-------------------------------------|
| face_detection.py     | BlazeFace           | Bounding box + 6 keypoints          |
| face_mesh.py          | Face Mesh           | 468 3D face landmarks               |
| hands.py              | MediaPipe Hands     | 21 landmarks per hand               |
| pose.py               | BlazePose           | 33 body landmarks                   |
| iris.py               | Iris model          | 5 iris landmarks + depth            |
| holistic.py           | Holistic (combined) | Face + hands + pose simultaneously  |
| head_posture.py       | Face Mesh + solvePnP| Pitch / Yaw / Roll in degrees       |
| selfie_segmentation.py| Selfie Segmentation | Binary mask: person vs background   |
| objectron.py          | Objectron           | 3D bounding box of objects          |

---

## Data Flow (holistic.py as example)

```
Webcam frame (BGR)
        |
  Convert BGR -> RGB  (MediaPipe requires RGB)
        |
  mp.solutions.holistic.Holistic.process(frame)
        |
  Returns:
    result.face_landmarks       (468 points)
    result.left_hand_landmarks  (21 points)
    result.right_hand_landmarks (21 points)
    result.pose_landmarks       (33 points)
        |
  mp_drawing.draw_landmarks():
    Draws skeleton connections between landmarks
        |
  Convert RGB -> BGR -> cv2.imshow() -> display
```

---

## Key Design Decisions

| Decision                       | Reason                                            |
|--------------------------------|---------------------------------------------------|
| MediaPipe pre-trained models   | State-of-art accuracy, CPU-only, single pip install|
| One script per capability      | Easy to study each model in isolation; modular    |
| custom/ geometry module        | Iris depth and 3D face geometry need custom math  |
| solvePnP for head posture      | Lifts 2D landmarks to 3D using known face geometry|

---

## Interview Conclusion

This repository systematically explores MediaPipe's full capability surface. The holistic
model is architecturally interesting: it runs three neural networks (face, hands, pose)
in a coordinated pipeline sharing a single body detection step, reducing latency versus
three independent detectors. The head posture script uses OpenCV's solvePnP to lift 2D
landmark coordinates to 3D by solving the Perspective-n-Point problem against a known
3D face model -- a standard computer vision technique for estimating camera pose. The
iris depth script in custom/iris_lm_depth.py derives eye-to-camera distance from the
apparent iris size, since the average human iris diameter (11.7mm) is a known constant.
Production extension: add a Kalman filter temporal smoother to reduce landmark jitter,
and action recognition on top of the pose landmark sequence.
