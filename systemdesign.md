# Holistic Diagram Using AI — System Design

## What It Does
A collection of real-time body analysis demos using Google MediaPipe. Each script demonstrates a different computer vision capability: face detection, face mesh, pose estimation, hand tracking, iris tracking, selfie segmentation, head posture analysis, facial expression, and the full holistic model — all live on webcam.

---

## Architecture

```
Webcam (live video stream)
        |
        v
+--------------------------------------------------+
|  OpenCV VideoCapture (videosource.py / video.py) |
|  Frame-by-frame capture                          |
+--------------------------------------------------+
        |
        v
+-------------------------------------------+
|  MediaPipe Model (per script)             |
|                                           |
|  holistic.py      -> full body pipeline   |
|  face_detection.py-> face bounding box    |
|  face_mesh.py     -> 468 face landmarks   |
|  hands.py         -> 21 hand landmarks    |
|  handstrack.py    -> hand tracking + draw |
|  pose.py          -> 33 body pose pts     |
|  iris.py          -> iris landmarks       |
|  head_posture.py  -> 3D head rotation     |
|  facial_expression-> expression classify  |
|  selfie_segmentation -> background remove |
|  objectron.py     -> 3D object detection  |
+-------------------------------------------+
        |
   Landmark data (x, y, z coordinates)
        |
        v
+--------------------------------------------------+
|  custom/ (geometry utilities)                    |
|  custom/core.py         -> math helpers          |
|  custom/face_geometry.py-> 3D face model         |
|  custom/iris_lm_depth.py-> iris depth estimation |
+--------------------------------------------------+
        |
        v
  Annotated frame -> cv2.imshow() -> display
```

---

## Individual Scripts and What They Detect

| Script | MediaPipe Model | Output |
|---|---|---|
| face_detection.py | BlazeFace | Bounding box + 6 keypoints |
| face_mesh.py | Face Mesh | 468 3D face landmarks |
| hands.py / handstrack.py | MediaPipe Hands | 21 landmarks per hand (wrist, knuckles, tips) |
| pose.py | BlazePose | 33 body landmarks (shoulder, elbow, knee, etc.) |
| iris.py | Iris model | 5 iris landmarks + depth estimation |
| holistic.py | Holistic | Face + hands + pose simultaneously |
| head_posture.py | Face Mesh + solvePnP | Pitch / Yaw / Roll angles |
| facial_expression.py | Face Mesh + classifier | Emotion label (happy, neutral, etc.) |
| selfie_segmentation.py | Selfie Segmentation | Binary mask: person vs background |
| objectron.py | Objectron | 3D bounding box of objects |

---

## Data Flow (holistic.py as example)

```
Webcam frame (BGR)
        |
  Convert: BGR -> RGB
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
    Draw connections between landmarks as skeleton overlay
        |
  Convert: RGB -> BGR for OpenCV display
        |
  cv2.imshow() -> annotated frame on screen
```

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| MediaPipe pre-trained models | State-of-art accuracy, CPU-only, no GPU needed, single pip install |
| One script per capability | Easy to study each model in isolation; modular and educational |
| custom/ geometry module | Iris depth and face 3D geometry require custom math beyond MediaPipe output |
| solvePnP for head posture | Uses 6 known 3D face points + 2D landmarks to solve camera pose |

---

## Interview Conclusion

This repository is a systematic exploration of MediaPipe's full capability surface — each script is both a working demo and a reference implementation. The holistic model is the most architecturally interesting: it runs three separate neural networks (face, hands, pose) in a coordinated pipeline that shares a single body detection step to reduce latency, rather than running three independent detectors. The head posture script (head_posture.py) demonstrates an important computer vision technique: using OpenCV's solvePnP to lift 2D landmark coordinates into 3D space by solving the Perspective-n-Point problem against a known 3D face model. The iris depth estimation in custom/iris_lm_depth.py derives eye-to-camera distance from the apparent size of the iris — a clever trick since the average human iris diameter is a known physical constant (11.7mm). If I were building a product on top of this, I would combine the holistic model with a temporal smoother (Kalman filter) to reduce landmark jitter, and add action recognition on top of the pose landmarks.
