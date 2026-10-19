# Hand Gesture Controlled Presentation

Control a slideshow with hand gestures using only a webcam. A pre-trained model (MediaPipe) finds your hand and its 21 landmark points. A classifier we trained on those points recognises the gesture, and a stability filter turns it into a slide action (next or previous) without accidental triggers.

## Contents

1. [What it does](#1-what-it-does)
2. [The gestures](#2-the-gestures)
3. [Before you start](#3-before-you-start)
4. [Step 1: Install Python 3.11](#4-step-1-install-python-311)
5. [Step 2: Get the project](#5-step-2-get-the-project)
6. [Step 3: Create a virtual environment](#6-step-3-create-a-virtual-environment)
7. [Step 4: Activate it](#7-step-4-activate-it)
8. [Step 5: Install the libraries](#8-step-5-install-the-libraries)
9. [Step 6: Run the project](#9-step-6-run-the-project)
10. [Using it with PowerPoint](#10-using-it-with-powerpoint)
11. [Recording your own data and retraining](#11-recording-your-own-data-and-retraining)
12. [How it works](#12-how-it-works)
13. [Settings you can tune](#13-settings-you-can-tune)
14. [Project structure](#14-project-structure)
15. [Troubleshooting](#15-troubleshooting)
16. [Limitations](#16-limitations)

---

## 1. What it does

- Reads your webcam and detects one hand.
- Classifies the hand shape into one of four classes: `NEXT`, `PREVIOUS`, `ARM`, `NEUTRAL`.
- Presses the **Right** or **Left** arrow key on the active window, so a slideshow moves forward or backward.
- Stays **locked (DISARMED)** until you show the arm gesture, so normal hand movement while talking does nothing.

## 2. The gestures

| Gesture | Hand shape | What happens | Hold time |
|---|---|---|---|
| **ARM / DISARM** | **L sign**: index finger up, thumb stuck out about 90 degrees, other fingers curled | Locks or unlocks the system | 0.8 s |
| **NEXT** | **Peace sign**: index and middle fingers up, ring and pinky down | Next slide (Right arrow) | 0.5 s |
| **PREVIOUS** | **Open palm**: all five fingers out | Previous slide (Left arrow) | 0.5 s |
| **NEUTRAL** | Anything else: fist, relaxed hand, pointing, a hand moving between gestures | Nothing | n/a |

Rules to know:

- The system **starts DISARMED**. Hold the L sign for about a second to arm it. The screen shows `ARMED (Live)`.
- Each gesture fires **once per hold**. Lower your hand or change shape before repeating it.
- After a slide changes there is a 1 second cooldown.
- Use your **right hand**, facing the camera. The camera image is mirrored, like a selfie.

## 3. Before you start

You need:

- A computer with a **webcam**, running Windows 10/11 (the main instructions), macOS, or Linux (short notes are included).
- An internet connection for the downloads.
- **No GPU needed.**
- About 1 GB of free disk space.

Every command below goes in a terminal:

- **Windows:** open **Command Prompt** (press the Windows key, type `cmd`, press Enter) or PowerShell.
- **macOS / Linux:** open **Terminal**.

---

## 4. Step 1: Install Python 3.11

> **Why exactly 3.11?** The pinned `mediapipe==0.10.9` has no installable files for Python 3.12 or newer, and the install fails there. Python 3.10 also works. Use **3.11**.

### Windows

1. Go to <https://www.python.org/downloads/windows/> and find a **Python 3.11.x** release. If the newest 3.11.x only offers source code, use **3.11.9**, the last 3.11 release with a normal Windows installer. Download **Windows installer (64-bit)**.
2. Run the installer. On the **first screen**, tick **"Add python.exe to PATH"** at the bottom. This is the most commonly missed step.
3. Click **Install Now** and wait for "Setup was successful".
4. Open a **new** Command Prompt and check:

   ```
   py -3.11 --version
   ```

   You should see `Python 3.11.x`. If you do, you're done.

   If `py` isn't recognised, try `python --version`. If neither works, run the installer again, choose **Modify**, and make sure "Add Python to environment variables" is ticked.

### macOS

1. Download the **macOS 64-bit universal2 installer** for 3.11.x from <https://www.python.org/downloads/macos/> and run it.
2. Check in Terminal:

   ```
   python3.11 --version
   ```

### Linux (Ubuntu/Debian)

```
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-tk
python3.11 --version
```

If your distribution doesn't have 3.11, install it with `pyenv` or from the "deadsnakes" package source.

---

## 5. Step 2: Get the project

**Option A: download the ZIP.** Download the project ZIP from the repository page and extract it. You'll get a folder, for example `hand-gesture-driven-ppt`.

**Option B: use git.**

```
git clone <your-repo-url>
```

Then move into the project folder. Replace the path with where *your* folder is:

```
cd path\to\hand-gesture-driven-ppt
```

(On macOS/Linux use forward slashes: `cd path/to/hand-gesture-driven-ppt`.)

**Run every later command from inside this project folder.** The scripts look for `data/`, `models/` and `src/` relative to it. Check you're in the right place:

- **Windows:** `dir` should list `main.py`, `train.py`, `collect.py`, `requirements.txt`.
- **macOS/Linux:** `ls` should list the same.

---

## 6. Step 3: Create a virtual environment

A virtual environment ("venv") is a private folder of Python libraries just for this project, so it can't clash with other projects on your computer.

### Windows

```
py -3.11 -m venv .venv
```

### macOS / Linux

```
python3.11 -m venv .venv
```

This creates a new folder called `.venv` inside the project. It takes a few seconds and prints nothing when it succeeds.

## 7. Step 4: Activate it

You must do this **every time you open a new terminal** before running the project.

### Windows, Command Prompt

```
.venv\Scripts\activate
```

### Windows, PowerShell

```
.\.venv\Scripts\Activate.ps1
```

If PowerShell says *"running scripts is disabled on this system"*, run this once and then try again:

```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

(Or just use Command Prompt instead.)

### macOS / Linux

```
source .venv/bin/activate
```

### How to know it worked

Your prompt now starts with `(.venv)`:

```
(.venv) C:\path\to\hand-gesture-driven-ppt>
```

Confirm you're using the right Python:

```
python --version
```

It must say `Python 3.11.x`. If it shows another version, the venv was created with the wrong Python: close the terminal, delete the `.venv` folder, and repeat Step 3 with `py -3.11`.

To leave the environment later, type `deactivate`.

---

## 8. Step 5: Install the libraries

With `(.venv)` showing, run:

```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs the exact versions the project was built with:

| Library | What it's used for |
|---|---|
| `opencv-python` | Reading the webcam and drawing the on-screen overlay |
| `mediapipe` | Finding the hand and its 21 landmark points |
| `numpy`, `pandas` | Handling the data |
| `scikit-learn`, `joblib` | Training, comparing, saving and loading the classifier |
| `pyautogui` | Pressing the arrow keys |
| `matplotlib` | Saving the confusion-matrix picture |

It takes a few minutes. Check that it worked:

```
python -c "import cv2, mediapipe, sklearn, pyautogui; print('All good')"
```

You should see `All good`.

---

## 9. Step 6: Run the project

Run these in order, from the project folder, with `(.venv)` active.

### 6a. Do you already have a trained model?

Look for the file `models/gesture_model.pkl`.

- **It exists:** skip to 6c.
- **It doesn't exist:** do 6b first. (The model file is created by training, so a fresh download may not include it.)

### 6b. Get training data, then train

You need `data/landmarks.csv`, which contains recorded hand landmarks.

- If the project came with `data/landmarks.csv`, skip straight to training.
- Otherwise record your own data first (see [section 11](#11-recording-your-own-data-and-retraining)).

Then train:

```
python train.py
```

It prints a comparison of four models (Random Forest, SVM, kNN, MLP), picks the best, and saves it as `models/gesture_model.pkl`. It also writes `results/model_comparison.csv`, `results/confusion_matrix.csv` and `results/confusion_matrix.png`.

If the data has **two or more different people**, it tests by training on some people and testing on a person it has never seen (this is the honest number). With only one person it prints a warning and uses a weaker test.

### 6c. Check the system with no risk (dry run)

```
python main.py --dry-run
```

A window opens showing your webcam. **No keys are pressed** in this mode; actions are only printed and counted. Use it to practice the gestures and to check for accidental triggers by talking naturally for a few minutes with the system armed.

What the screen shows:

| On-screen text | Meaning |
|---|---|
| `Raw: NEXT (0.93)` | What the model thinks this single frame is, and how sure it is |
| `ARMED (Live)` / `DISARMED (Locked)` | Whether actions will fire |
| Yellow bar | How close the current hold is to firing |
| `SLIDE COOLDOWN` / `TOGGLE COOLDOWN` | Short pause after an action |
| `DRY RUN - triggers: N` | How many actions would have fired |

Press **`q`** with the camera window selected to quit.

### 6d. Use it for real

```
python main.py
```

Same window, but now the arrow keys are actually pressed. Read the next section before using it with slides.

---

## 10. Using it with PowerPoint

Key presses go to **whichever window is in front**, so the slideshow needs to be the active window.

1. Open your presentation and start the slideshow (**F5**, or the Slide Show tab).
2. In a terminal, start `python main.py`. The camera window opens.
3. **Click on the slideshow** to make it the active window. (If the camera window stays active, the keys go to it and slides won't change.)
4. Stand about 0.5 to 1.5 m from the camera, with the hand clearly visible.
5. Hold the **L sign** until the camera window says `ARMED`, then use the **peace sign** (next) and **open palm** (previous).
6. Show the L sign again to lock the system.

Tips:

- You can keep the camera window small and in a corner, as long as it's running.
- Face a light source if you can. A bright window behind you makes the hand dark and hard to detect.
- **Emergency stop:** move your mouse to any screen corner (PyAutoGUI's fail-safe) or press `q` in the camera window.

Works with Google Slides in a browser, PDF viewers, and other slideshow programs too, since it only presses Left and Right.

---

## 11. Recording your own data and retraining

Do this to add more people, improve accuracy, or rebuild the model from scratch.

### Record

Each person records separately, with their own name:

```
python collect.py --person alice
```

A window opens. Press a number key to record that gesture, move/tilt your hand slowly, and press **`s`** to stop that gesture:

| Key | Records | How to do it |
|---|---|---|
| `0` | `NEXT` | Peace sign |
| `1` | `PREVIOUS` | Open palm, fingers spread |
| `2` | `ARM` | **Exaggerated L**: thumb clearly out, about 90 degrees from the index finger |
| `3` | `NEUTRAL` | Fist, relaxed open hand, **pointing with the thumb tucked in**, half-formed gestures, a hand moving between gestures |
| `s` | Stop | |
| `q` | Quit and save | |

There's a 2 second "get ready" countdown after pressing a key, so you have time to get into the pose.

Good practice:

- Record about **1 to 2 minutes per gesture**.
- Vary **distance** (near and far), **angle** (tilt the hand a little), and **lighting**.
- `NEUTRAL` is the hardest and most important. Include the in-between poses, because those are what cause accidental triggers.
- Record **at least 3 to 4 different people**. Models trained on one person usually work worse on others.
- Run the command once per person with a different `--person` name. All rows go into `data/landmarks.csv`.

> If `collect.py` says the file has an old header, rename the old file (for example to `data/landmarks_old.csv`) and run again. Newer files include `person_id` and `session` columns.

### Check the hand-coded baseline (optional but useful)

```
python test_baseline.py
```

This scores a simple rule-based classifier (finger counting, no machine learning) on your data, per person. Comparing it with the trained model's results is the evidence that the ML model adds something.

### Retrain

```
python train.py
```

Look at `results/confusion_matrix.png` to see which gestures get mixed up. Then run `main.py` again.

To test the pipeline with the original single-person data:

```
python train.py --data data/landmarks_old_format.csv
```

---

## 12. How it works

```
Webcam frame
   -> MediaPipe Hands -> 21 landmark points (x, y, z)
   -> Feature extraction (translate to wrist, scale by hand size, plus 8 shape features)
   -> Classifier -> gesture + confidence
   -> Stability filter -> action (NEXT / PREVIOUS / arm toggle)
   -> PyAutoGUI presses the arrow key
```

**Why normalise the landmarks?** Raw coordinates change if you move or stand closer. Subtracting the wrist position and dividing by hand size makes the features depend on the *shape* of the hand, not where it is or how far away.

**Why a trained model instead of rules?** Counting raised fingers works for simple cases, but fails when a relaxed open hand looks like the "previous" gesture, or when pointing looks like the L. The trained model learns from examples to tell those apart.

**The stability filter** is what makes it usable. A gesture fires only if all of these are true:

1. The model is at least 75% confident.
2. It wins a majority vote over the last 0.4 seconds of frames, so one flickering frame doesn't matter.
3. It has been held for its hold time (0.5 s for slides, 0.8 s for arm).
4. Its cooldown is over.
5. It hasn't already fired during this same hold. You release it before it can fire again.
6. For slide gestures: the system is ARMED.

If the hand leaves the frame, the filter resets, so an old, half-finished gesture can't fire later.

---

## 13. Settings you can tune

All in `main.py`, where `GestureFilter(...)` is created:

| Setting | Default | Effect |
|---|---|---|
| `confidence_threshold` | 0.75 | Higher means fewer false triggers but more missed gestures |
| `slide_hold_time` | 0.5 s | How long to hold NEXT/PREVIOUS |
| `toggle_hold_time` | 0.8 s | How long to hold the L to arm or disarm |
| `slide_cooldown` | 1.0 s | Pause after a slide action |
| `toggle_cooldown` | 2.0 s | Pause after arming or disarming |

The L detection in the hand-coded baseline uses a threshold in `src/rules.py` (`thumb_far`, default 0.40). This only affects `test_baseline.py`, not the live app.

---

## 14. Project structure

```
hand-gesture-driven-ppt/
├── collect.py            Record landmark data for one person
├── train.py              Compare models, evaluate, save the best one
├── test_baseline.py      Score the rule-based baseline on recorded data
├── main.py               Run the live gesture controller
├── requirements.txt      Library versions
├── CHANGES.md            Log of bugs found and fixed
├── src/
│   ├── features.py       Feature extraction + CSV loading (used everywhere)
│   ├── rules.py          Rule-based baseline classifier
│   ├── stability.py      Smoothing, hold time, cooldown, arm/disarm logic
│   └── actions.py        Presses the keys
├── data/
│   └── landmarks.csv     Recorded data (created by collect.py)
├── models/
│   └── gesture_model.pkl The trained model (created by train.py)
└── results/              Comparison table and confusion matrix (created by train.py)
```

---

## 15. Troubleshooting

| Problem | Cause and fix |
|---|---|
| `python` or `py` is not recognised | Python isn't installed or not on PATH. Reinstall and tick "Add python.exe to PATH". |
| `ERROR: Could not find a version that satisfies the requirement mediapipe==0.10.9` | You're on Python 3.12 or newer. Install Python 3.11, delete `.venv`, and redo Steps 3 to 5. |
| PowerShell: "running scripts is disabled" | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, or use Command Prompt. |
| `ModuleNotFoundError: No module named 'cv2'` (or `mediapipe`, `sklearn`...) | The venv isn't active, or libraries aren't installed. Check for `(.venv)` in your prompt, then run `pip install -r requirements.txt`. |
| `ModuleNotFoundError: No module named 'src'` | You're not in the project folder. `cd` into it first. |
| `FileNotFoundError: models/gesture_model.pkl` | Nothing has been trained yet. Run `python train.py` (you need `data/landmarks.csv` first). |
| `FileNotFoundError: data/landmarks.csv` | Record data first with `python collect.py --person yourname`. |
| "Model was trained with different features. Run: python train.py" | The model is from an older version. Retrain. |
| A scikit-learn version note, or odd model errors | The model was saved on another scikit-learn version. Retrain it on your machine. |
| `Could not open the webcam` | Another app (Zoom, Teams, browser) may be using it. Close them. For a second camera use `python main.py --camera 1`. Check your OS camera permissions. |
| Camera works but slides don't move | The slideshow isn't the active window. Click it after starting `main.py`. Also confirm you're not in `--dry-run`. |
| macOS: keys aren't pressed | Give your terminal permission under **System Settings, Privacy & Security, Accessibility** and **Camera**. |
| Linux: PyAutoGUI errors | It needs an X11 session (not Wayland) and `python3.11-tk`. |
| Gestures flicker or are not detected | Improve lighting (face a light, avoid a bright window behind you), keep the hand 0.5 to 1.5 m away, and keep it fully inside the frame. |
| It triggers by accident | Raise `confidence_threshold` or the hold times, stay DISARMED when you're not presenting, and record more `NEUTRAL` data. |
| It misses the arm (L) gesture | Open the thumb wider, about 90 degrees from the index finger, and hold for about a second. |
| Low FPS | Close other apps. Lower the camera resolution in your camera settings. |

---

## 16. Limitations

- It tracks **one hand** and is designed for the **right hand**.
- Gestures are **static poses**. Swipes and moving gestures aren't supported.
- It works best at about **0.5 to 2 m** from the camera. Further away the hand is too small to track reliably.
- Accuracy drops in **dim light or strong backlight**.
- A model trained on few people may be less accurate for new users. Recording more people improves this.
- A normal presentation clicker is simpler and more reliable for plain slide changes. This project is a demonstration of real-time hand-gesture recognition and its stability handling.
- Only camera frames are processed locally to compute landmarks. **No images or video are saved**, only landmark numbers when you record data.
