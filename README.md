Here’s a clean, practical roadmap you can start implementing **right now in Python**, with each step broken into the specific components you’ll need. Since you’ll manually supply a tutorial video, we can base everything on a single input file and expand later.

I’ll keep it concise, as you prefer, while giving you enough structure to begin coding.

---

# **🧩 Phase 1 — Base Project Structure**

### **Folder layout**

```
clay-hand-pipeline/
    data/
        raw/
            tutorial.mp4
        segments/
        landmarks/
    src/
        shot_segmentation.py
        hand_tracking.py
        normalization.py
        clay_filter.py
        feature_extraction.py
        cluster.py
    notebooks/
        exploration.ipynb
    requirements.txt
```

We’ll fill these in step-by-step.

---

# **🪚 Phase 2 — Shot Segmentation (Hands-based)**

**Goal:** Break the video into segments containing actual hand activity.

We’ll use Mediapipe just to detect when hands are present—no landmarks yet.

### **Step 2.1 — Build a shot segmentation script**

`shot_segmentation.py` outline:

```python
import cv2
import mediapipe as mp
from pathlib import Path

def segment_by_hand_presence(video_path, min_active_frames=15):
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    segments = []
    active_start = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hands_present = results.multi_hand_landmarks is not None

        if hands_present and active_start is None:
            active_start = frame_idx

        if not hands_present and active_start is not None:
            if frame_idx - active_start >= min_active_frames:
                segments.append((active_start, frame_idx))
            active_start = None

        frame_idx += 1

    cap.release()
    return segments, fps
```

### **Step 2.2 — Export segments**

```python
def save_segments(video_path, segments, fps, out_dir="data/segments"):
    Path(out_dir).mkdir(exist_ok=True)
    import ffmpeg

    for i, (start, end) in enumerate(segments):
        ss = start / fps
        t = (end - start) / fps

        out_path = f"{out_dir}/segment_{i:03d}.mp4"

        (
            ffmpeg
            .input(video_path, ss=ss, t=t)
            .output(out_path, c="copy")
            .run(overwrite_output=True)
        )
```

---

# **🖐️ Phase 3 — Extract Hand Landmarks for Each Segment**

`hand_tracking.py`:

```python
def extract_landmarks(video_path):
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    landmarks = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            hands = []
            for hand in results.multi_hand_landmarks:
                hands.append([(lm.x, lm.y, lm.z) for lm in hand.landmark])
            landmarks.append((frame_idx, hands))

        frame_idx += 1

    cap.release()
    return landmarks
```

We’ll save these as JSON or numpy arrays.

---

# **🧭 Phase 4 — Normalize Pose to a Canonical Hand Orientation**

`normalization.py`:

Normalization involves:

* Centering on wrist
* Scaling based on palm size
* Rotation-aligning using PCA on the palm

```python
import numpy as np

def normalize_landmarks(landmarks):
    pts = np.array(landmarks)  # shape (21, 3)

    # 1. Center
    wrist = pts[0]
    pts -= wrist

    # 2. Scale
    scale = np.linalg.norm(pts[9] - pts[0])  # wrist to middle finger MCP
    pts /= scale

    # 3. Rotation normalize using PCA
    pca = PCA(n_components=3)
    pts = pca.fit_transform(pts)

    return pts
```

This puts profile vs top-down shots into roughly the same orientation space.

---

# **🟫 Phase 5 — Filter Segments Where Hands Actually Touch Clay**

We can get a simple first-pass filter using **motion + region of interest**.

`clay_filter.py`:

```python
def is_clay_interaction(landmarks_seq, wheel_center, wheel_radius):
    touching = []

    for pts in landmarks_seq:
        fingertips = pts[[8, 12, 16, 20]]  # index, middle, ring, pinky
        for tip in fingertips:
            dist = np.linalg.norm(tip[:2] - wheel_center)
            if dist < wheel_radius * 1.2:
                touching.append(True)
                break
        else:
            touching.append(False)

    return sum(touching) / len(touching) > 0.3
```

Later you can replace ROI with a real wheel detector (Hough circle).

---

# **⭐ Phase 6 — Extract Features for Clustering**

A simple approach uses:

* normalized (21×3) → flatten → 63-dim vector
* OR distance matrix (21×21) → flatten → 441-dim vector

`feature_extraction.py`:

```python
def frame_to_feature(norm_pts):
    # distance matrix
    D = np.linalg.norm(norm_pts[:, None, :] - norm_pts[None, :, :], axis=-1)
    return D.flatten()
```

Distance matrices give you view-invariance almost “for free.”

---

# **🌌 Phase 7 — Clustering**

Start with HDBSCAN:

`cluster.py`:

```python
import hdbscan
from sklearn.decomposition import PCA

def cluster_features(X):
    X_reduced = PCA(n_components=20).fit_transform(X)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=50)
    labels = clusterer.fit_predict(X_reduced)
    return labels
```

Later you can try:

* SimCLR
* BYOL
* Temporal autoencoders

---

# **🎯 Phase 8 — Evaluate + Visualization**

Open a notebook:

```python
import seaborn as sns
sns.scatterplot(x=emb[:,0], y=emb[:,1], hue=labels)
```

And map clusters back to timecodes to inspect behavior clusters.

---

# **If you want, we can now start coding each piece live.**

Just send the video tutorial and we’ll start implementing segmentation and landmark extraction step-by-step.
