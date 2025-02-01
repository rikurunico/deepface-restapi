from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from deepface import DeepFace
import sqlite3
import os
import shutil
from datetime import datetime
from pydantic import BaseModel
import uuid

app = FastAPI(title="Face Recognition API")

# Constants
IMAGE_STORAGE = "./face_storage"
TEMP_STORAGE = "./temp_storage"
DATABASE_PATH = "face_db.sqlite"

# Ensure directories exist
os.makedirs(IMAGE_STORAGE, exist_ok=True)
os.makedirs(TEMP_STORAGE, exist_ok=True)


# Database connection
def get_db():
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# Initialize database
def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS persons (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id TEXT PRIMARY KEY,
            person_id TEXT NOT NULL,
            image_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (person_id) REFERENCES persons (id)
        )
    """
    )
    conn.commit()
    conn.close()


init_db()


def detect_faces(image_path: str) -> bool:
    """Detect if image contains faces."""
    try:
        faces = DeepFace.extract_faces(
            img_path=image_path, detector_backend="opencv", enforce_detection=False
        )
        return len(faces) > 0
    except Exception as e:
        print(f"Face detection error: {str(e)}")
        return False


@app.post("/api/v1/persons", status_code=201)
async def create_person(name: str = Form(...), image: UploadFile = File(...)):
    """Register a new person with their face(s)."""
    conn = get_db()
    cursor = conn.cursor()

    temp_path = None
    saved_path = None

    try:
        # Save uploaded image temporarily
        temp_path = os.path.join(TEMP_STORAGE, f"temp_{uuid.uuid4()}.jpg")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Check if image contains faces
        if not detect_faces(temp_path):
            raise HTTPException(
                status_code=400, detail="No faces detected in the image"
            )

        # Save the actual image
        saved_filename = f"{uuid.uuid4()}.jpg"
        saved_path = os.path.join(IMAGE_STORAGE, saved_filename)
        shutil.copy2(temp_path, saved_path)

        # Create person record
        person_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO persons (id, name) VALUES (?, ?)", (person_id, name)
        )

        # Store face embedding
        embedding_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO face_embeddings (id, person_id, image_path) VALUES (?, ?, ?)",
            (embedding_id, person_id, saved_path),
        )

        conn.commit()

        return {
            "status": "success",
            "message": "Successfully registered person",
            "data": {"person_id": person_id, "name": name},
        }

    except Exception as e:
        conn.rollback()
        if saved_path and os.path.exists(saved_path):
            os.remove(saved_path)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        conn.close()


@app.post("/api/v1/recognition")
async def recognize_faces(image: UploadFile = File(...)):
    """Recognize faces in an image."""
    conn = get_db()
    cursor = conn.cursor()

    temp_path = None

    try:
        # Save uploaded image temporarily
        temp_path = os.path.join(TEMP_STORAGE, f"temp_{uuid.uuid4()}.jpg")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Check if image contains faces
        if not detect_faces(temp_path):
            raise HTTPException(
                status_code=400, detail="No faces detected in the image"
            )

        # Get all stored face embeddings
        cursor.execute(
            """
            SELECT fe.image_path, p.name, p.id
            FROM face_embeddings fe
            JOIN persons p ON fe.person_id = p.id
        """
        )
        stored_faces = cursor.fetchall()

        recognized_faces = []

        # Compare with each stored face
        for stored_face in stored_faces:
            try:
                result = DeepFace.verify(
                    img1_path=temp_path,
                    img2_path=stored_face["image_path"],
                    model_name="Facenet",
                    detector_backend="opencv",
                    enforce_detection=False,
                )

                if result["verified"]:
                    recognized_faces.append(
                        {
                            "person_id": stored_face["id"],
                            "name": stored_face["name"],
                            "confidence": result["distance"],
                        }
                    )

            except Exception as e:
                print(f"Verification error: {str(e)}")
                continue

        if recognized_faces:
            return {
                "status": "success",
                "message": f"Found {len(recognized_faces)} matching face(s)",
                "data": {"recognized_faces": recognized_faces},
            }
        else:
            return {
                "status": "success",
                "message": "No matching faces found",
                "data": {"recognized_faces": []},
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        conn.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
