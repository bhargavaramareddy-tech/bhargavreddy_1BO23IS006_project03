import cv2
import mediapipe as mp
import sounddevice as sd
import numpy as np
import speech_recognition as sr
import pyttsx3
from scipy.io.wavfile import write
import tkinter as tk
from tkinter import messagebox
import threading
import queue
import time

# ----------------- TTS QUEUE (non-blocking) -----------------
tts_queue = queue.Queue()
engine = pyttsx3.init()
engine.setProperty("rate", 160)

def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("TTS error:", e)
        tts_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def speak_async(text):
    tts_queue.put(text)

# ----------------- SPEECH TO TEXT -----------------
def record_audio(duration=4, fs=44100):
    print("Recording audio...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    return np.squeeze(audio)

def save_audio(data, filename="speech.wav", fs=44100):
    # ensure int16
    data16 = np.asarray(data, dtype=np.int16)
    write(filename, fs, data16)
    return filename

def recognize_speech(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Speech service error: {e}"

def do_speech_to_text():
    def process():
        btn_stt.config(state="disabled")
        lbl_stt_result.config(text="Recording...")
        try:
            data = record_audio(duration=4)
            filename = save_audio(data)
            lbl_stt_result.config(text="Recognizing...")
            text = recognize_speech(filename)
            lbl_stt_result.config(text=text)
        except Exception as e:
            lbl_stt_result.config(text=f"Error: {e}")
            print("STT error:", e)
        finally:
            btn_stt.config(state="normal")
    threading.Thread(target=process, daemon=True).start()

# ----------------- GESTURE RECOGNITION -----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def fingers_up(hand_landmarks, handedness_label):
    # returns list of booleans for [thumb, index, middle, ring, pinky]
    tips = [4, 8, 12, 16, 20]
    pip = [3, 6, 10, 14, 18]
    res = []
    lm = hand_landmarks.landmark
    # Thumb: compare x (because thumb points sideways). handedness helps.
    if handedness_label == "Right":
        res.append(lm[tips[0]].x < lm[pip[0]].x)
    else:
        res.append(lm[tips[0]].x > lm[pip[0]].x)
    # Other fingers: tip y < pip y means finger is up (camera y increases downward)
    for i in range(1,5):
        res.append(lm[tips[i]].y < lm[pip[i]].y)
    return res

def classify_gesture(finger_states):
    # finger_states: [thumb, index, middle, ring, pinky] booleans
    t, i, m, r, p = finger_states
    # Define gestures:
    if all([i, m, r, p]) and not t:
        return "OPEN_PALM"   # hello / stop depending on mapping
    if not any([t, i, m, r, p]):
        return "FIST"
    if i and not any([m, r, p, t]):
        return "INDEX"
    if t and not any([i, m, r, p]):
        return "THUMB_UP"
    if i and m and not any([r, p, t]):
        return "PEACE"
    return "UNKNOWN"

def detect_gesture():
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_spoken = ""
    last_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks and result.multi_handedness:
                hand_landmarks = result.multi_hand_landmarks[0]
                hand_label = result.multi_handedness[0].classification[0].label  # "Right" or "Left"
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                states = fingers_up(hand_landmarks, hand_label)
                gesture = classify_gesture(states)

                # Map detected gesture to text
                if gesture == "THUMB_UP":
                    spoken = "YES"
                elif gesture == "INDEX":
                    spoken = "HELLO"
                elif gesture == "OPEN_PALM":
                    spoken = "STOP"
                elif gesture == "FIST":
                    spoken = "I'M HERE"
                elif gesture == "PEACE":
                    spoken = "NO"
                else:
                    spoken = ""

                # Debounce so it doesn't repeat too fast
                if spoken and (spoken != last_spoken or time.time() - last_time > 2.0):
                    last_spoken = spoken
                    last_time = time.time()
                    lbl_gesture.config(text=f"Gesture: {spoken}")
                    speak_async(spoken)

            cv2.imshow("Gesture Camera (press Q to close)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print("Gesture loop error:", e)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

def start_gesture():
    threading.Thread(target=detect_gesture, daemon=True).start()

# ----------------- UI -----------------
root = tk.Tk()
root.title("Smart Communication Assistant - Neon Dark Mode")
root.geometry("1000x550")
root.configure(bg="#0A0F1F")

title = tk.Label(root, text="Smart Communication Assistant", fg="#00FFA8", bg="#0A0F1F", font=("Arial Rounded MT Bold", 28))
title.pack(pady=10)

frame_main = tk.Frame(root, bg="#0A0F1F")
frame_main.pack(fill="both", expand=True)

# Column 1 - Speech to Text
col1 = tk.Frame(frame_main, bg="#11182E", bd=3, relief="groove")
col1.place(relx=0.03, rely=0.18, relwidth=0.30, relheight=0.75)

tk.Label(col1, text="Speech → Text", fg="#00FFA8", bg="#11182E", font=("Arial", 18, "bold")).pack(pady=10)
btn_stt = tk.Button(col1, text="Start Recording (4s)", command=do_speech_to_text, bg="#00FFA8", fg="black", font=("Arial", 14))
btn_stt.pack(pady=15)
lbl_stt_result = tk.Label(col1, text="Speak something...", fg="white", bg="#11182E", font=("Arial", 14), wraplength=250)
lbl_stt_result.pack(pady=10)

# Column 2 - Text to Speech
col2 = tk.Frame(frame_main, bg="#11182E", bd=3, relief="groove")
col2.place(relx=0.35, rely=0.18, relwidth=0.30, relheight=0.75)

tk.Label(col2, text="Text → Speech", fg="#00FFA8", bg="#11182E", font=("Arial", 18, "bold")).pack(pady=10)
entry_tts = tk.Entry(col2, font=("Arial", 16), width=22)
entry_tts.pack(pady=15)
def on_speak_btn():
    text = entry_tts.get().strip()
    if not text:
        messagebox.showwarning("Warning", "Enter text.")
        return
    speak_async(text)
tk.Button(col2, text="Speak", command=on_speak_btn, bg="#00FFA8", fg="black", font=("Arial", 14)).pack(pady=10)

# Column 3 - Gesture
col3 = tk.Frame(frame_main, bg="#11182E", bd=3, relief="groove")
col3.place(relx=0.67, rely=0.18, relwidth=0.30, relheight=0.75)

tk.Label(col3, text="Gesture Recognition", fg="#00FFA8", bg="#11182E", font=("Arial", 18, "bold")).pack(pady=10)
tk.Button(col3, text="Start Gesture", command=start_gesture, bg="#00FFA8", fg="black", font=("Arial", 14)).pack(pady=15)
lbl_gesture = tk.Label(col3, text="Gesture: None", fg="white", bg="#11182E", font=("Arial", 14))
lbl_gesture.pack(pady=10)

root.mainloop()
# quit TTS worker when GUI closes:
tts_queue.put(None)
