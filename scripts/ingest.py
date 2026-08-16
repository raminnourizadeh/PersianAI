"""Build the Persian dense + BM25 sparse Qdrant collection."""

from pathlib import Path
import json
import os
import time
import uuid

from pypdf import PdfReader
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import torch

from persian_rag.rag_utils import (
    build_bm25,
    normalize_persian,
    sparse_document_vector,
    structural_chunks,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333").rstrip("/")
EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "persian_documents")
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
BM25_PATH = PROJECT_ROOT / "data" / "indexes" / "bm25_index.json"
BATCH_SIZE = 8


print("\n" + "=" * 65)
cuda_has_room = False
if torch.cuda.is_available():
    free_memory, _ = torch.cuda.mem_get_info()
    cuda_has_room = free_memory >= 4 * 1024**3

embedding_device = "cuda" if cuda_has_room else "cpu"
print(f"Loading embedding model on {embedding_device}...")
print("=" * 65)
load_start = time.perf_counter()
embedding_model = SentenceTransformer(EMBED_MODEL, device=embedding_device)
embedding_model.max_seq_length = 1024
print(f"Embedding model loaded in {time.perf_counter() - load_start:.2f} sec")


def collect_chunks(pdf_files):
    records = []

    for pdf_path in pdf_files:
        print(f"\nدر حال پردازش ساختاری: {pdf_path.name}")
        reader = PdfReader(pdf_path)

        for page_number, page in enumerate(reader.pages, start=1):
            text = normalize_persian(page.extract_text() or "")
            if len(text) < 30:
                continue

            chunks = structural_chunks(text)
            print(
                f"Page {page_number}: {len(text)} chars -> "
                f"{len(chunks)} structural chunks"
            )

            for chunk_index, chunk in enumerate(chunks):
                records.append(
                    {
                        "text": chunk["text"],
                        "source": pdf_path.name,
                        "page": page_number,
                        "chunk": chunk_index,
                        "section": chunk["section"],
                    }
                )

    return records


def main():
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        print("هیچ PDF در پوشه documents پیدا نشد.")
        return

    records = collect_chunks(pdf_files)
    if not records:
        print("هیچ متن قابل استفاده‌ای استخراج نشد.")
        return

    texts = [record["text"] for record in records]
    print(f"\nBuilding Persian BM25 index for {len(texts)} chunks...")
    bm25 = build_bm25(texts)
    BM25_PATH.write_text(
        json.dumps(bm25, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Creating dense embeddings...")
    embedding_start = time.perf_counter()
    dense_vectors = embedding_model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embedding_time = time.perf_counter() - embedding_start
    vector_size = dense_vectors.shape[1]

    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(COLLECTION_NAME):
        print("Collection قبلی حذف می‌شود...")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams()
        },
    )

    points = []
    for record, dense_vector in zip(records, dense_vectors):
        indices, values = sparse_document_vector(record["text"], bm25)
        stable_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f'{record["source"]}:{record["page"]}:{record["chunk"]}',
        )
        points.append(
            models.PointStruct(
                id=str(stable_id),
                vector={
                    "dense": dense_vector.tolist(),
                    "bm25": models.SparseVector(indices=indices, values=values),
                },
                payload=record,
            )
        )

    for start in range(0, len(points), 64):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[start:start + 64],
            wait=True,
        )

    print("\n" + "=" * 65)
    print("✅ اسناد ساختاری با Dense + BM25 وارد Qdrant شدند.")
    print(f"Total chunks: {len(records)}")
    print(f"Vocabulary: {len(bm25['vocabulary'])}")
    print(f"Embedding time: {embedding_time:.2f} sec")
    print("=" * 65)


if __name__ == "__main__":
    main()
