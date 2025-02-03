from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
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
import pickle
from typing import Dict, List, Optional
import logging
from datetime import datetime
import json
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class FaceRecognitionAPI:
    def __init__(self):
        # Constants
        self.IMAGE_STORAGE = "./face_storage"
        self.TEMP_STORAGE = "./temp_storage"
        self.DATABASE_PATH = "face_db.sqlite"
        self.DETECTOR_BACKEND = "retinaface"
        self.MODEL_NAME = "ArcFace"
        self.EMBEDDING_SIZE = 512
        self.SIMILARITY_THRESHOLD = 0.6
        self.FAISS_INDEX_PATH = "./faiss_index.index"
        self.ID_MAPPING_PATH = "./id_mapping.pkl"
        self.BATCH_SIZE = 32

        # Initialize storage directories
        self._init_storage()

        # Initialize components
        self.gpu_res = self._init_gpu_resources()
        self.index_lock = threading.Lock()
        self.id_to_person: Dict[int, str] = {}
        self.index = self._init_faiss()

        # Initialize database
        self._init_database()

    def _init_storage(self):
        """Initialize storage directories"""
        os.makedirs(self.IMAGE_STORAGE, exist_ok=True)
        os.makedirs(self.TEMP_STORAGE, exist_ok=True)

    def _init_gpu_resources(self):
        """Initialize GPU resources for FAISS"""
        gpu_res = faiss.StandardGpuResources()
        gpu_res.setTempMemory(1024 * 1024 * 512)  # 512MB
        return gpu_res

    def _init_faiss(self):
        """Initialize FAISS index"""
        if os.path.exists(self.FAISS_INDEX_PATH):
            try:
                index = faiss.read_index(self.FAISS_INDEX_PATH)
                return faiss.index_cpu_to_gpu(self.gpu_res, 0, index)
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                return self._create_new_index()
        return self._create_new_index()

    def _create_new_index(self):
        """Create a new FAISS index"""
        index = faiss.IndexFlatIP(self.EMBEDDING_SIZE)
        return faiss.index_cpu_to_gpu(self.gpu_res, 0, index)

    @contextmanager
    def get_db_connection(self):
        """Database connection context manager"""
        conn = sqlite3.connect(self.DATABASE_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_database(self):
        """Initialize database tables"""
        with self.get_db_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id TEXT PRIMARY KEY,
                    person_id TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (person_id) REFERENCES persons (id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_person_id ON face_embeddings(person_id);
            """
            )

    def save_id_mapping(self):
        """Save id_to_person mapping to disk"""
        with open(self.ID_MAPPING_PATH, "wb") as f:
            pickle.dump(self.id_to_person, f)

    def load_id_mapping(self):
        """Load id_to_person mapping from disk"""
        if os.path.exists(self.ID_MAPPING_PATH):
            with open(self.ID_MAPPING_PATH, "rb") as f:
                self.id_to_person = pickle.load(f)

    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding vector"""
        return (embedding / np.linalg.norm(embedding)).astype(np.float32)

    async def load_embeddings(self):
        """Load embeddings from database to FAISS index"""
        logger.info("Loading embeddings from database...")
        self.id_to_person.clear()

        try:
            self.load_id_mapping()

            if not self.id_to_person:
                with self.get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT id, person_id, embedding FROM face_embeddings"
                    )

                    embeddings = []
                    batch = []

                    for idx, row in enumerate(cursor):
                        embedding = np.frombuffer(row["embedding"], dtype=np.float32)
                        batch.append(embedding)
                        self.id_to_person[idx] = row["person_id"]

                        if len(batch) >= self.BATCH_SIZE:
                            embeddings.extend(batch)
                            batch = []

                    if batch:
                        embeddings.extend(batch)

                    if embeddings:
                        embeddings_array = np.array(embeddings)
                        with self.index_lock:
                            self.index = self._create_new_index()
                            self.index.add(embeddings_array)

                        cpu_index = faiss.index_gpu_to_cpu(self.index)
                        faiss.write_index(cpu_index, self.FAISS_INDEX_PATH)
                        self.save_id_mapping()

                logger.info(f"Loaded {len(embeddings)} embeddings")
            else:
                logger.info("Using existing id mapping")

        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise

    async def create_person(self, name: str, image: UploadFile) -> Dict:
        """Register new person with face embedding"""
        temp_path = os.path.join(self.TEMP_STORAGE, f"temp_{uuid.uuid4()}.jpg")
        person_id = str(uuid.uuid4())

        try:
            # Save temporary file
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

            # Detect faces
            faces = DeepFace.extract_faces(
                img_path=temp_path,
                detector_backend=self.DETECTOR_BACKEND,
                enforce_detection=True,
                align=True,
            )

            if len(faces) != 1:
                raise HTTPException(400, "Image must contain exactly one face")

            # Process face
            face_img = faces[0]["face"]
            face_filename = f"{uuid.uuid4()}.jpg"
            face_path = os.path.join(self.IMAGE_STORAGE, face_filename)
            cv2.imwrite(face_path, (face_img * 255).astype(np.uint8))

            # Generate embedding
            embedding = DeepFace.represent(
                img_path=face_path,
                model_name=self.MODEL_NAME,
                detector_backend="skip",
                enforce_detection=False,
            )[0]["embedding"]

            normalized_embedding = self.normalize_embedding(np.array(embedding))

            # Database operations
            with self.get_db_connection() as conn:
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
            with self.index_lock:
                self.index.add(normalized_embedding.reshape(1, -1))
                self.id_to_person[self.index.ntotal - 1] = person_id
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, self.FAISS_INDEX_PATH)
                self.save_id_mapping()

            return {"message": "Person registered successfully", "person_id": person_id}

        except Exception as e:
            logger.error(f"Error creating person: {e}")
            raise HTTPException(400, f"Registration failed: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    async def recognize_faces(self, image: UploadFile, threshold: float = None) -> Dict:
        """Recognize faces in image"""
        if threshold is None:
            threshold = self.SIMILARITY_THRESHOLD

        temp_path = os.path.join(self.TEMP_STORAGE, f"temp_{uuid.uuid4()}.jpg")

        try:
            # Save temporary file
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

            # Detect faces
            faces = DeepFace.extract_faces(
                img_path=temp_path,
                detector_backend=self.DETECTOR_BACKEND,
                enforce_detection=True,
                align=True,
            )

            results = []

            for face in faces:
                try:
                    # Generate embedding
                    embedding = DeepFace.represent(
                        img_path=face["face"],
                        model_name=self.MODEL_NAME,
                        detector_backend="skip",
                        enforce_detection=False,
                    )[0]["embedding"]

                    query_embedding = self.normalize_embedding(np.array(embedding))

                    # FAISS search
                    with self.index_lock:
                        D, I = self.index.search(query_embedding.reshape(1, -1), 1)

                    if D[0][0] > threshold:
                        with self.get_db_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute(
                                "SELECT * FROM persons WHERE id = ?",
                                (self.id_to_person.get(I[0][0]),),
                            )
                            person = cursor.fetchone()
                            if person:
                                results.append(
                                    {
                                        "confidence": float(D[0][0]),
                                        "person": dict(person),
                                    }
                                )

                except Exception as e:
                    logger.error(f"Error processing face: {e}")
                    continue

            return {"results": results}

        except Exception as e:
            logger.error(f"Error recognizing faces: {e}")
            raise HTTPException(500, f"Recognition failed: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


# Initialize FastAPI app
app = FastAPI(title="Optimized Face Recognition API")
api = FaceRecognitionAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes
@app.on_event("startup")
async def startup_event():
    await api.load_embeddings()


@app.post("/api/v1/persons", status_code=201)
async def create_person(name: str = Form(...), image: UploadFile = File(...)):
    return await api.create_person(name, image)


@app.post("/api/v1/recognize")
async def recognize_faces(
    image: UploadFile = File(...), threshold: Optional[float] = None
):
    return await api.recognize_faces(image, threshold)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
