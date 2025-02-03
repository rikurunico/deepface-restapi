from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from deepface import DeepFace
import sqlite3
import os
import shutil
import uuid
import cv2
import numpy as np
import faiss
import traceback


app = FastAPI(title="Optimized Face Recognition API")

# Constants
IMAGE_STORAGE = "./face_storage"
TEMP_STORAGE = "./temp_storage"
DATABASE_PATH = "face_db.sqlite"
DETECTOR_BACKEND = "retinaface"
MODEL_NAME = "ArcFace"
EMBEDDING_SIZE = 512
SIMILARITY_THRESHOLD = 0.6

# FAISS Initialization
gpu_res = faiss.StandardGpuResources()
index = faiss.IndexFlatIP(EMBEDDING_SIZE)
index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
id_to_person = {}

# Ensure directories exist
os.makedirs(IMAGE_STORAGE, exist_ok=True)
os.makedirs(TEMP_STORAGE, exist_ok=True)


# Database connection
def get_db():
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


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
            embedding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (person_id) REFERENCES persons (id)
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


@app.on_event("startup")
async def load_embeddings():
    """Pre-load embeddings to FAISS index on startup"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, person_id, embedding FROM face_embeddings")

    embeddings = []
    for row in cursor:
        embedding = np.frombuffer(row["embedding"], dtype=np.float32)
        embeddings.append(embedding)
        id_to_person[index.ntotal] = row["person_id"]

    if embeddings:
        index.add(np.array(embeddings))

    conn.close()


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize embedding vector for cosine similarity"""
    return embedding / np.linalg.norm(embedding)


@app.post("/api/v1/persons", status_code=201)
async def create_person(name: str = Form(...), image: UploadFile = File(...)):
    """Register a new person with face embedding"""
    conn = get_db()
    cursor = conn.cursor()
    temp_path = os.path.join(TEMP_STORAGE, f"temp_{uuid.uuid4()}.jpg")
    person_id = str(uuid.uuid4())

    try:
        # Save temporary file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Detect and align face
        faces = DeepFace.extract_faces(
            img_path=temp_path,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=True,
        )

        if len(faces) != 1:
            raise HTTPException(400, "Image must contain exactly one face")

        # Save face image
        face_img = faces[0]["face"]
        face_filename = f"{uuid.uuid4()}.jpg"
        face_path = os.path.join(IMAGE_STORAGE, face_filename)
        cv2.imwrite(face_path, (face_img * 255).astype(np.uint8))

        # Calculate face embedding
        embedding_result = DeepFace.represent(
            img_path=face_path,  # Use the cropped face image
            model_name=MODEL_NAME,
            detector_backend="skip",  # Skip detection since we already have cropped face
            enforce_detection=False,
        )

        if not embedding_result:
            raise HTTPException(400, "Failed to generate face embedding")

        # Extract embedding vector
        embedding_vector = np.array(embedding_result[0]["embedding"], dtype=np.float32)
        normalized_embedding = normalize_embedding(embedding_vector)

        # Store in database
        cursor.execute(
            "INSERT INTO persons (id, name) VALUES (?, ?)", (person_id, name)
        )
        cursor.execute(
            "INSERT INTO face_embeddings (id, person_id, image_path, embedding) VALUES (?, ?, ?, ?)",
            (str(uuid.uuid4()), person_id, face_path, normalized_embedding.tobytes()),
        )
        conn.commit()

        # Update FAISS index
        index.add(normalized_embedding.reshape(1, -1))
        id_to_person[index.ntotal - 1] = person_id

        return {"message": "Person registered successfully"}

    except Exception as e:
        conn.rollback()
        traceback.print_exc()
        raise HTTPException(400, f"Registration failed: {str(e)}")
    finally:
        conn.close()
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/v1/recognize")
async def recognize_faces(
    image: UploadFile = File(...), threshold: float = SIMILARITY_THRESHOLD
):
    """Recognize faces in image using GPU-accelerated search"""
    temp_path = os.path.join(TEMP_STORAGE, f"temp_{uuid.uuid4()}.jpg")
    results = []

    try:
        # Save temporary file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Detect and align faces
        faces = DeepFace.extract_faces(
            img_path=temp_path,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=True,
        )

        conn = get_db()

        for face in faces:
            # Save each face to temporary file
            face_temp_path = os.path.join(TEMP_STORAGE, f"face_{uuid.uuid4()}.jpg")
            cv2.imwrite(face_temp_path, face["face"] * 255)  # Convert to uint8

            try:
                # Generate embedding using temp face file
                embedding_dict = DeepFace.represent(
                    img_path=face_temp_path,
                    model_name=MODEL_NAME,
                    detector_backend="skip",
                    enforce_detection=False,
                    align=True,
                )
                # Extract embedding array from dict
                embedding_array = embedding_dict[0]["embedding"]
                embedding_np = normalize_embedding(
                    np.array(embedding_array, dtype=np.float32)
                )

                # Search in FAISS index
                D, I = index.search(embedding_np.reshape(1, -1), 1)

                if D[0][0] > threshold:
                    person_id = id_to_person.get(I[0][0])
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM persons WHERE id = ?", (person_id,))
                    person = cursor.fetchone()

                    if person:
                        results.append(
                            {"confidence": float(D[0][0]), "person": dict(person)}
                        )
            finally:
                if os.path.exists(face_temp_path):
                    os.remove(face_temp_path)

        return {"results": results}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Recognition failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
