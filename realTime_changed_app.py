import os
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for, Response, session
from flask_session import Session
import cv2
from werkzeug.utils import secure_filename
import tensorflow as tf
from collections import deque, Counter
import pickle
import mediapipe as mp
from gtts import gTTS
import time
import speech_recognition as sr
import psycopg2
from flask_bcrypt import Bcrypt
from flask_jwt_extended import create_access_token, JWTManager
from datetime import datetime


app = Flask(__name__)


app.config["JWT_SECRET_KEY"] = "your_secret_key" 
app.config['SESSION_TYPE'] = "filesystem"
Session(app)

# Hashing the passwords and accessing the token
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# Constants
IMG_SIZE = (224, 224)
SEQUENCE_LENGTH = 20  # Number of frames required for prediction
CLASS_MAP = {0: 'Accident', 1: 'Call', 2: 'Doctor', 3: 'Help', 4: 'Hot', 5: 'Lose', 6: 'Pain', 7: 'Thief'}

# File Paths
VIDEO_FOLDER = "static/upload_media/videos"
AUDIO_FOLDER = "static/upload_media/recorded_speech"
EMERGENCY_SIGNS_PATH = "static/emergency_words_gif"
PROCESSED_AUDIO_TRANSLATION = "static/translated_audio"

# Creating the folders for uploads
for folder in [VIDEO_FOLDER, AUDIO_FOLDER,PROCESSED_AUDIO_TRANSLATION]:
    os.makedirs(folder, exist_ok=True)


# Preload available emergency words
emergency_words = {f.split('.')[0].lower(): f for f in os.listdir(EMERGENCY_SIGNS_PATH)}


# Emergency words model loading
try:
    video_model = tf.keras.models.load_model("Gesture_cnn_lstm_model.h5")
    base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    cnn_out = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    cnn_model = tf.keras.Model(inputs=base_model.input, outputs=cnn_out)
except Exception as e:
    print(f"Error loading the model: {e}")
    video_model, cnn_model = None, None

# ================== Initialize Mediapipe Hands ==================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils


# ================== Alphabets and Numbers Model Loading ==================  
try:
    with open('MODEL_LATEST_TUNING_augment_islrtc.p','rb') as f:
        model_dict = pickle.load(f)
        alpha_model = model_dict['model']     
except Exception as e:
    print(f"Error loading the model: {e}")
    alpha_model = None


# ================== Real-Time Constants ==================
prev_prediction = ""
hold_counter = 0
hold_threshold = 15
last_hand_time = time.time()
space_added = False

# Better buffer using deque
prediction_buffer = deque(maxlen=15)  # Buffer size for smoothing
confidence_threshold = 10  # Min occurrences in buffer to consider stable

# Sentence buffers
current_sentence = ""
final_sentence = ""
final_sentences_history = []

# Constants for duplicate letter control
last_letter_added = ""
last_added_time = 0
letter_repeat_cooldown = 1.0  # in seconds



# ================== Hand Landmark Extraction ==================
def extract_hand_landmarks(img):
    """Extract hand landmarks from an image/frame and normalize coordinates."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    data_aux = []
    hand_landmarks_list = results.multi_hand_landmarks or []

    if len(hand_landmarks_list) == 1:
        hand_landmarks_list.append(None) 

    for hand_landmarks in hand_landmarks_list:
        x_vals, y_vals = [], []
        if hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x_vals.append(landmark.x)
                y_vals.append(landmark.y)

            min_x, max_x = min(x_vals), max(x_vals)
            min_y, max_y = min(y_vals), max(y_vals)

            for landmark in hand_landmarks.landmark:
                norm_x = (landmark.x - min_x) / (max_x - min_x + 1e-6)
                norm_y = (landmark.y - min_y) / (max_y - min_y + 1e-6)
                data_aux.extend([norm_x, norm_y])
        else:
            data_aux.extend([0] * 42)

    return np.array([data_aux]) if len(data_aux) == 84 else None

# ================== Saving the translated sentence in database ==================
def save_translation_to_db(sentence):
    user_id = session.get("user_id")
    try :
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            
            cur.execute(
                """ INSERT INTO Gesture_translations (user_id, translated_sentence) VALUES (%s, %s)
            ON CONFLICT (user_id, translated_sentence) DO NOTHING;""", (user_id, sentence)
            )
            conn.commit()
            cur.close()
            conn.close()
            print(f"[DB] Saved: {sentence}")
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500



# ================== Real-Time Frame Generation ==================
def generate_frames():
    global current_sentence, final_sentence, prev_prediction
    global hold_counter, last_hand_time, space_added, prediction_buffer
    global final_sentences_history
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while cap and cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Extract hand landmarks
        data_np = extract_hand_landmarks(frame)

        gesture_text = "Waiting for gesture..."
        space_text = ""
        display_current = current_sentence
        display_final = final_sentence

        if data_np is not None:
            last_hand_time = time.time()
            space_added = False

            # Prediction and buffering
            prediction = alpha_model.predict(data_np)[0]
            prediction_buffer.append(prediction)

            # Get most common prediction and its frequency
            most_common_pred = Counter(prediction_buffer).most_common(1)[0]
            stable_prediction, freq = most_common_pred

            # Confidence filtering
            if freq >= confidence_threshold:
                if stable_prediction == prev_prediction:
                    hold_counter += 1
                else:
                    hold_counter = 1
                    prev_prediction = stable_prediction

                # Confirm gesture
                if hold_counter >= hold_threshold:
                    current_time = time.time()
                    
                    # Check if this is a new letter or enough time has passed since same letter was added
                    if stable_prediction != last_letter_added or (current_time - last_added_time) > letter_repeat_cooldown:
                        current_sentence += stable_prediction
                        print(f"[Letter Added] Current Sentence: {current_sentence}")
                        
                        last_letter_added = stable_prediction
                        last_added_time = current_time
                        
                    # Reset the tracking
                    hold_counter = 0
                    prediction_buffer.clear()

                gesture_text = f"Gesture: {stable_prediction}"
        
        else:
            # No hand detected
            elapsed = time.time() - last_hand_time
            
            if elapsed > 3 and not space_added:
                current_sentence += " "
                space_text = "Space Added"
                space_added = True
                print(f"[Space] Current Sentence: {current_sentence}")
                
                # Reset duplicate letter tracking 

            elif elapsed > 7:
                if current_sentence.strip():
                    final_sentence = current_sentence.strip()
                    final_sentences_history.append(final_sentence)
                    
                    # Store the final sentence in the db
                    save_translation_to_db(final_sentence)
                    print(f"[Final] Final Sentence: {final_sentence}")

                    # Reset for next sentence
                    current_sentence = ""
                    prediction_buffer.clear()
                    hold_counter = 0
                    space_added = False
                    last_letter_added = ""

                last_hand_time = time.time()

        # Draw landmarks
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display text overlays
        cv2.putText(frame, gesture_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.putText(frame, space_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, f"Current: {display_current}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, f"Final: {display_final}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
    cap.release()
    last_letter_added = ""
    print("[Camera Stopped] Sentence history:", final_sentences_history)
   
# Upload file handler
def get_uploaded_file(file_key):
    if file_key not in request.files:
        return None, jsonify({"error": f"No {file_key} uploaded"}), 400
    
    file = request.files[file_key]
    if file.filename == '':
        return None, jsonify({"error": f"No selected {file_key} file"}), 400
    
    filename = secure_filename(file.filename)
    # file_path = os.path.join(VIDEO_FOLDER if file_key == "video" else UPLOAD_FOLDER, filename)
    if file_key == "video":
        file_path = os.path.join(VIDEO_FOLDER, filename)
    else:
        return "Plaese upload valid file"
    
    file.save(file_path)
    
    return file_path, None, 200 
    
# Helper function to process video and make predictions
def run_model_on_video(video_path):
    if not video_model or not cnn_model:
        return "Model is not loaded."
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count < SEQUENCE_LENGTH:
        return "Video has less than 20 frames"
    
    frame_indices = np.linspace(0, frame_count - 1, SEQUENCE_LENGTH, dtype=int)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, IMG_SIZE) / 255.0
        frames.append(frame)
    
    cap.release()
    
    if len(frames) != SEQUENCE_LENGTH:
        return "Could not extract 20 frames from the video"
    
    frames = np.array(frames).reshape(-1, 224, 224, 3)
    features = cnn_model.predict(frames, batch_size=16, verbose=0)
    features = features.reshape(1, SEQUENCE_LENGTH, features.shape[-1])
    
    prediction = video_model.predict(features)
    predicted_class = np.argmax(prediction)
    
    return CLASS_MAP.get(predicted_class, "Unknown") 

# Database Connection
def get_db_connection():
    try:
        return psycopg2.connect(
            dbname="sign_language_db",
            user="postgres",
            password="Namrata",
            host="localhost",
            port="5432"
        )
    except Exception as e:
        print(f"Database connection error: {e}")
        return None   
    
# Fetching the media from the database:
def fetch_media_from_db(media_type, category=None, char=None):
    conn = get_db_connection()
    if not conn:
        print("Unable to connect to the database")
        return None
    
    cur = conn.cursor()
    
    if media_type == "image":
        query = "SELECT file_path FROM Media_dataset WHERE media_type='image' AND category=%s AND file_path LIKE %s"
        cur.execute(query, (category, f"%/{char.upper()}.%"))
    elif media_type == "video":
        query = "SELECT file_path FROM Media_dataset WHERE media_type='video' AND category=%s AND file_path ILIKE %s"
        cur.execute(query, (category, f"%{char.lower()}%"))  # Match the specific word

    result = cur.fetchone()
    cur.close()
    conn.close()
    
    if result:
        return result[0].replace("\\", "/")  # Ensure Flask-compatible path
    else:
        print("Result is none")
        return None
    

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

# About Page route
@app.route('/about')
def about():
    return render_template('about.html')

# Service Page route
@app.route('/services')
def services():
    return render_template('services.html')

# Feedback Page route 
@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

# Processing the video API
@app.route('/process_video', methods=['POST'])
def process_video():
    user_id = session.get("user_id")  
    print("\nUser id is : ", user_id,"\n")
    video_path, error_response, status = get_uploaded_file("video")
    if error_response:
        return error_response, status
    
    result_text = run_model_on_video(video_path)
    processed_status = bool(result_text)
    
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            
            # Check if the file already exists for this user
            cur.execute("SELECT id FROM User_uploads WHERE user_id = %s AND file_path = %s;", (user_id, video_path))
            existing_file = cur.fetchone()

            if existing_file:
                # If file exists, still return the result but don't insert it again
                cur.close()
                conn.close()
                return jsonify({"warning": "You have already uploaded this file.", "result": result_text})
            
            cur.execute(
                "INSERT INTO User_uploads (user_id, media_type, file_path, processed_status, prediction_result) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id;",
                (user_id, "video", video_path, processed_status, result_text)
            )
            conn.commit()
            cur.close()
            conn.close()
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500
      
    return jsonify({"result": result_text})

# Signup Route
@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    username = data["username"]
    email = data["email"]
    password = bcrypt.generate_password_hash(data["password"]).decode("utf-8")

    try:
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("SELECT user_id FROM users WHERE email = %s;", (email,))
        existing_user = cur.fetchone()

        if existing_user:
            return jsonify({"error": "Email already exists"}), 400  # Custom error message
        
        cur.execute(
            "INSERT INTO users (username, email, password) VALUES (%s, %s, %s) RETURNING user_id;",
            (username, email, password)
        )
        conn.commit()
        return jsonify({"message": "User registered successfully!"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
# Login Route
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT user_id, username, password FROM users WHERE email = %s;", (email,))
            user = cur.fetchone()

        if user and bcrypt.check_password_hash(user[2], password):
            access_token = create_access_token(identity=user[0])
            session["user_id"] = user[0]
            return jsonify({
                "token": access_token,
                "username": user[1],
                "message": "Login successful!"
            })

        return jsonify({"error": "Invalid credentials"}), 401

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Changed to 500 for server errors

    finally:
        conn.close()  # Ensure the database connection is closed

# Text to Sign processing
@app.route('/process_text', methods=['POST'])
def process_text():
    user_id = session.get("user_id")
    data = request.json
    text = data.get("text", "").strip().lower()
    input_type = data.get("input_type", "text")
    audio_path = data.get("audio_path", None) # By default is None, when no audio file is found
    
    # Clean the text by removing special characters except space 
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    
    print(f"\nuser_id: {user_id},\nText: {text},\nInput_type: {input_type}, \naudio_path: {audio_path}")

    if not text or text == "undefined":
        return jsonify({"error": "No text provided"}), 400

    words = text.split()
    result = []

    for word in words:
        if word in emergency_words:
            gif_path = fetch_media_from_db("video", category="Emergency Sign", char=word)
            if gif_path:
                result.append({"type": "gif", "path": gif_path, "label": word.capitalize()})
        else:
            word_data = {"type": "images", "paths": [], "label": word.capitalize()}
            for char in word:
                if char.isalnum():
                    if char == "0":
                        char = "o"  # Treat '0' as 'O'
                    img_path = fetch_media_from_db("image", category="Alphabet" if char.isalpha() else "Number", char=char)
                    if img_path:
                        word_data["paths"].append(img_path)
            result.append(word_data)
    
    status = "processed" if result else "failed"
            
    try:
        conn = get_db_connection()
        if conn: 
            cur = conn.cursor()
            
            # Check if the user has entered the same sentence more than once
            cur.execute("SELECT id from Text_sign_mapping WHERE user_id = %s AND sentence = %s", (user_id, text))
            existing_file = cur.fetchone()
            
            if existing_file:
                # If the user has entered the same sentence more than once, give the warning , and procced as usual
                cur.close()
                conn.close()
                return jsonify({"status": "processed", "result": result, "warning": "You have translated the same sentence again"})
            
            cur.execute(
                "INSERT INTO Text_sign_mapping (user_id, input_type, audio_path, sentence, list_of_words, status) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;",
                (user_id, input_type, audio_path, text, words, status)
            )
            conn.commit()
            conn.close()
            cur.close()
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    
    finally:
        cur.close()
        conn.close()  

    return jsonify({"result": result})

# Speech to text handling
@app.route('/speech-to-text',methods=['POST'])
def speech_to_text():
    user_id = session.get("user_id")
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        # print("Listening....")
        recognizer.adjust_for_ambient_noise(source)
        
        try:
            audio = recognizer.listen(source, timeout=5)
            
            filename = f"speech_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp3"
            file_path = os.path.join(AUDIO_FOLDER, filename)
            
            with open (file_path, "wb") as f:
                f.write(audio.get_wav_data())
                
            text = recognizer.recognize_google(audio,language="en-IN")
            
            return jsonify({
                "text": text,
                "input_type": "speech",
                "audio_path": file_path  # Return audio file path
            })
        
        except sr.UnknownValueError:
            return jsonify({"error:":"Could not understand the audio"}), 400
        
        except sr.RequestError:
            return jsonify({"error:":"Speech Recognition Service error."}), 500
        
# Text to speech conversion
@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    data = request.get_json()
    
    # Convert to lower case to avoid duplicates
    text = data.get("text", "").strip().lower()
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    audio_filename = f"{text}.mp3"
    audio_path = os.path.join(PROCESSED_AUDIO_TRANSLATION,audio_filename)
    
    if os.path.exists(audio_path):
        return jsonify({"audio_url": audio_path})

    try:
        tts = gTTS(text=text, lang='en')  # Convert text to speech
        tts.save(audio_path)  # Save the audio file
        
        return jsonify({"audio_url": audio_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500        
  
# Feedback submission
@app.route("/submit-feedback", methods=["POST"])
def submit_feedback():
    data = request.get_json()

    user_id = session.get("user_id") 
    if not user_id:
        return jsonify({"error": "User not logged in"}), 401
    
    name=data["name"]
    email=data.get("email")
    category=data.get("category")
    rating=int(data.get("rating"))
    message=data.get("message")
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute(
            "INSERT INTO feedback(user_id, email, name, category, rating, message) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;",
            (user_id, email, name, category, rating, message)
        )
        conn.commit()
        
        return jsonify({"message": "Feedback submitted successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
 
# Video Response from camera to frontend           
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Sending the translated sentence in real time to frontend for display
@app.route('/get_translation')
def get_translation():
    return jsonify({
        "gesture": prev_prediction,
        "current_text": current_sentence,
        "final_sentence": final_sentence})

@app.route('/get_all_sentences')
def get_all_sentences():
    return jsonify({
        "all_sentences": final_sentences_history
    })  
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
