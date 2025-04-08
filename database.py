import psycopg2
import os

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="sign_language_db",
    user="postgres",
    password="Namrata",
    host="localhost",
    port="5432"
)

cur = conn.cursor()


#------------------------------------ Creating the tables in the db -------------------------------------
#  Users Table 
cur.execute("""
CREATE TABLE IF NOT EXISTS Users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

#  Image/Video Dataset Table
cur.execute("""
CREATE TABLE IF NOT EXISTS Media_dataset (
    id SERIAL PRIMARY KEY,
    media_type VARCHAR(10) CHECK (media_type IN ('image','video')) NOT NULL,
    file_path TEXT NOT NULL UNIQUE,
    category VARCHAR(100),
    description TEXT
);
""")

#  Uploaded Media Table
cur.execute("""
CREATE TABLE IF NOT EXISTS User_uploads (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(user_id) ON DELETE CASCADE,
    media_type VARCHAR(10) CHECK (media_type IN ('video')) NOT NULL,
    file_path TEXT NOT NULL,
    processed_status BOOLEAN DEFAULT FALSE,
    prediction_result TEXT,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, file_path)
);
""")

#  Real-Time Hand Gesture Recognition Table
cur.execute("""
CREATE TABLE IF NOT EXISTS Gesture_translations (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(user_id) ON DELETE CASCADE,
    translated_sentence TEXT NOT NULL,
    translated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, translated_sentence)  
);
""")

#Text to Sign Table 
cur.execute("""
CREATE TABLE IF NOT EXISTS Text_sign_mapping (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(user_id) ON DELETE CASCADE,
    input_type TEXT CHECK(input_type IN ('text','speech')) NOT NULL,
    audio_path TEXT,
    sentence TEXT NOT NULL,
    list_of_words TEXT[] NOT NULL,
    word_count INT GENERATED ALWAYS AS (array_length(list_of_words, 1)) STORED,
    status TEXT DEFAULT 'pending',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, sentence)
);
""")

cur.execute("""
CREATE TABLE feedback (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    category TEXT NOT NULL,
    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
    message TEXT NOT NULL,
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);         
""")

print("\nTables created successfully!")

            

#------------------------------------ Adding the data in tables in the db -------------------------------------

# Folder paths (ensure these exist)
alphabet_folder = "static/alphabets_numbers"  # A-Z & 0-9 images
emergency_gif_folder = "static/emergency_words_gif"  # Emergency sign videos/gifs

# Debugging: Print the absolute path of the alphabet folder
# print("Alphabet Folder Absolute Path:", os.path.abspath(alphabet_folder))
# print("Files in Alphabet Folder:", os.listdir(alphabet_folder))

# Insert Alphabet & Number Sign Images (A-Z, 0-9)
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
    file_path = None
    for ext in ["png", "PNG", "jpg", "jpeg"]:
        temp_path = os.path.join(alphabet_folder, f"{letter if letter != '0' else 'O'}.{ext}")
        temp_path = os.path.normpath(temp_path).replace("\\", "/")  # Normalize path
        
        if os.path.exists(temp_path):  # Check if file exists
            file_path = temp_path
            break  # Stop checking if file is found

    if file_path:
        category = "Alphabet" if letter.isalpha() else "Number"
        cur.execute("""
            INSERT INTO Media_dataset (media_type, file_path, category, description)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (file_path) DO NOTHING
        """, ("image", file_path, category, f"Sign language representation of '{letter}'"))
    else:
        print(f"❌ File not found for {letter}")

# Insert Emergency Sign GIFs
emergency_gestures = ["call", "doctor", "help", "hot", "lose", "pain", "accident", "thief"]
for gesture in emergency_gestures:
    file_path = os.path.join(emergency_gif_folder, f"{gesture}.gif")
    file_path = os.path.normpath(file_path).replace("\\", "/")  # Normalize path

    if os.path.exists(file_path):  # Check if file exists before inserting
        cur.execute("""
            INSERT INTO Media_dataset (media_type, file_path, category, description)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (file_path) DO NOTHING
        """, ("video", file_path, "Emergency Sign", f"Emergency gesture for '{gesture}'"))

        
print("\nAll images and GIFs added successfully!")

# Commit and close
conn.commit()
cur.close()
conn.close()