"""BGE-M3 dense embeddings and exact cosine ranking."""

from .markdown import Chunk


MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
MAX_TOKENS = 512
BATCH_SIZE = 16


class BgeM3:
    def __init__(self, device: str, local_files_only: bool = False) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install the Dense dependencies from requirements-dense.txt") from exc

        self.model = SentenceTransformer(
            MODEL_ID,
            revision=MODEL_REVISION,
            device=device,
            local_files_only=local_files_only,
            model_kwargs={"use_safetensors": False},
        )
        self.model.max_seq_length = MAX_TOKENS

    def encode(self, texts: list[str]):
        return self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


def rank_dense(chunks: list[Chunk], document_vectors, query_vector, top_k: int):
    scores = document_vectors @ query_vector
    hits = zip(chunks, (float(score) for score in scores), strict=True)
    return sorted(hits, key=lambda hit: (-hit[1], hit[0].start_line))[:top_k]
