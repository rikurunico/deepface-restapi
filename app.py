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
from contextlib import contextmanager
import threading

app = FastAPI(title="Optimized Face Recognition API")

# Constants
IMAGE_STORAGE = "./face_storage"
TEMP_STORAGE = "./temp_storage"
DATABASE_PATH = "face_db.sqlite"
DETECTOR_BACKEND = "retinaface"
MODEL_NAME = "ArcFace"
EMBEDDING_SIZE = 512
SIMILARITY_THRESHOLD = 0.6
FAISS_INDEX_PATH = "./faiss_index.index"

# Global initialization
gpu_res = faiss.StandardGpuResources()
gpu_res.setTempMemory(1024 * 1024 * 512)  # 512MB
index_lock = threading.Lock()
id_to_person = {}


# Database setup
@contextmanager
def db_connection():
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS persons (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.execute(
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


init_db()


def initialize_faiss():
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
    else:
        index = faiss.IndexFlatIP(EMBEDDING_SIZE)
        index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
    return index


index = initialize_faiss()


@app.on_event("startup")
async def load_embeddings():
    """Load embeddings from database to FAISS index"""
    global index
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, person_id, embedding FROM face_embeddings")

        embeddings = []
        ids = []
        for row in cursor:
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            embeddings.append(embedding)
            ids.append(row["id"])
            id_to_person[len(embeddings) - 1] = row["person_id"]

        if embeddings:
            embeddings_array = np.array(embeddings)
            with index_lock:
                index.add(embeddings_array)
            faiss.write_index(faiss.index_gpu_to_cpu(index), FAISS_INDEX_PATH)


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize embedding vector with mixed precision"""
    return (embedding / np.linalg.norm(embedding)).astype(np.float32)


@app.post("/api/v1/persons", status_code=201)
async def create_person(name: str = Form(...), image: UploadFile = File(...)):
    """Register new person with memory-optimized processing"""
    temp_path = os.path.join(TEMP_STORAGE, f"temp_{uuid.uuid4()}.jpg")
    person_id = str(uuid.uuid4())

    try:
        # Save temporary file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Face detection
        try:
            faces = DeepFace.extract_faces(
                img_path=temp_path,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True,
                align=True,
            )
        except ValueError as e:
            raise HTTPException(400, "No face detected") from e

        if len(faces) != 1:
            raise HTTPException(400, "Image must contain exactly one face")

        # Process face
        face_img = faces[0]["face"]
        face_filename = f"{uuid.uuid4()}.jpg"
        face_path = os.path.join(IMAGE_STORAGE, face_filename)
        cv2.imwrite(face_path, (face_img * 255).astype(np.uint8))

        # Generate embedding
        try:
            embedding = DeepFace.represent(
                img_path=face_path,
                model_name=MODEL_NAME,
                detector_backend="skip",
                enforce_detection=False,
            )[0]["embedding"]
        except Exception as e:
            raise HTTPException(500, "Embedding generation failed") from e

        normalized_embedding = normalize_embedding(np.array(embedding))

        # Database operations
        with db_connection() as conn:
            conn.execute(
                "INSERT INTO persons (id, name) VALUES (?, ?)", (person_id, name)
            )
            conn.execute(
                "INSERT INTO face_embeddings (id, person_id, image_path, embedding) VALUES (?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    person_id,
                    face_path,
                    normalized_embedding.tobytes(),
                ),
            )
            conn.commit()

        # Update FAISS index
        with index_lock:
            index.add(normalized_embedding.reshape(1, -1))
            id_to_person[index.ntotal - 1] = person_id
            faiss.write_index(faiss.index_gpu_to_cpu(index), FAISS_INDEX_PATH)

        return {"message": "Person registered successfully"}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(400, f"Registration failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/v1/recognize")
async def recognize_faces(
    image: UploadFile = File(...), threshold: float = SIMILARITY_THRESHOLD
):
    """Memory-optimized face recognition"""
    temp_path = os.path.join(TEMP_STORAGE, f"temp_{uuid.uuid4()}.jpg")

    try:
        # Save temporary file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Detect faces
        try:
            faces = DeepFace.extract_faces(
                img_path=temp_path,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True,
                align=True,
            )
        except ValueError as e:
            return {"results": []}

        results = []

        for face in faces:
            try:
                # Generate embedding directly from numpy array
                embedding = DeepFace.represent(
                    img_path=face["face"],
                    model_name=MODEL_NAME,
                    detector_backend="skip",
                    enforce_detection=False,
                )[0]["embedding"]

                query_embedding = normalize_embedding(np.array(embedding))

                # FAISS search
                with index_lock:
                    D, I = index.search(query_embedding.reshape(1, -1), 1)

                if D[0][0] > threshold:
                    with db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT * FROM persons WHERE id = ?",
                            (id_to_person.get(I[0][0]),),
                        )
                        person = cursor.fetchone()
                        if person:
                            results.append(
                                {"confidence": float(D[0][0]), "person": dict(person)}
                            )
            except Exception as e:
                traceback.print_exc()
                continue

        return {"results": results}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Recognition failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
