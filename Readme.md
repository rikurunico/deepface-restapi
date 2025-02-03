# Face Recognition API

A high-performance face recognition API built with **FastAPI** and optimized using **FAISS** for efficient similarity search. This system provides endpoints for registering individuals and recognizing faces in images.

## Tech Stack

### Core Components
- **FastAPI**: Modern web framework for building high-performance APIs with Python 3.7+.
- **DeepFace**: Face recognition framework handling detection and embedding extraction.
- **FAISS** (Facebook AI Similarity Search): Library for efficient similarity search of dense vectors.
- **SQLite**: Lightweight database for storing person records and embeddings.
- **OpenCV**: Image processing and face alignment.
- **UVicorn**: ASGI server for running the FastAPI application.

### Key Python Packages
- `fastapi`: Web framework.
- `deepface`: Face detection and recognition.
- `faiss-cpu`/`faiss-gpu`: Similarity search.
- `numpy`: Numerical operations.
- `opencv-python`: Image processing.
- `python-multipart`: File upload handling.
- `sqlite3`: Database integration.

### Optimization Tools
- **RetinaFace**: High-accuracy face detector.
- **ArcFace**: State-of-the-art face recognition model.
- **GPU Acceleration**: FAISS index operations (requires a CUDA-enabled GPU).
- **Batch Processing**: Efficient handling of multiple face embeddings.

## Environment Setup

### Prerequisites
- **Conda package manager**: Make sure Conda is installed to manage dependencies and virtual environments.
- **CUDA-enabled GPU** (optional for GPU acceleration): If you want to utilize GPU acceleration, ensure that your GPU supports CUDA. For a guide, refer to the following tutorial:
  - [TensorFlow GPU Installation on Windows](https://www.lavivienpost.com/install-tensorflow-gpu-on-windows-complete-guide/#7)

### Installation
1. Clone the repository:
```bash
git clone https://github.com/rikurunico/deepface-restapi.git
cd deepface-restapi
```

2. Create the Conda environment:
```bash
conda env create -f environment.yml
```

3. Activate the environment:
```bash
conda activate tf_env
```

## API Endpoints

### Register Person
`POST /api/v1/persons`
- Registers a new person with their face embedding.
- **Parameters**:
  - `name`: The person's name (form data).
  - `image`: The face image file (JPEG/PNG).

### Recognize Faces
`POST /api/v1/recognize`
- Recognizes faces in an image.
- **Parameters**:
  - `image`: The image file containing faces.
  - `threshold`: Similarity threshold (default: 0.6).

## Usage Examples

### Register a Person
```bash
curl -X POST -F "name=John Doe" -F "image=@john.jpg" http://localhost:8000/api/v1/persons
```

### Recognize Faces
```bash
curl -X POST -F "image=@group_photo.jpg" http://localhost:8000/api/v1/recognize
```

Sample Response:
```json
{
  "results": [
    {
      "confidence": 0.82,
      "person": {
        "id": "a1b2c3d4-...",
        "name": "John Doe",
        "created_at": "2023-07-15T10:00:00"
      }
    }
  ]
}
```

## Performance Considerations
- **GPU Utilization**: FAISS index operations are GPU-accelerated.
- **Batch Processing**: Efficient handling of multiple faces in a single image.
- **Cached Embeddings**: Pre-loaded embeddings in FAISS index at startup.
- **Face Alignment**: RetinaFace detector ensures optimal recognition accuracy.

## Important Notes
1. The first run will download DeepFace models (~500MB).
2. Minimum recommended image size: 640x480 pixels.
3. Optimal recognition threshold: 0.6-0.7.
4. The API automatically handles face alignment and normalization.


## License
This project is licensed under the terms set by the author. Please refer to the [LICENSE](LICENSE) file for details on usage and permissions.

