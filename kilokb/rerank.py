"""Cross-encoder and BGE-M3 token-vector reranking for frozen candidates."""

from .dense import BATCH_SIZE, MAX_TOKENS, MODEL_ID, MODEL_REVISION
from .frozen import FrozenCandidate


CROSS_MODEL = "BAAI/bge-reranker-v2-m3"
CROSS_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


def reorder(
    candidates: tuple[FrozenCandidate, ...], scores: list[float], top_k: int
) -> list[tuple[FrozenCandidate, float]]:
    paired = zip(candidates, scores, strict=True)
    return sorted(paired, key=lambda hit: (-hit[1], hit[0].input_rank))[:top_k]


def maxsim_score(query, document) -> float:
    return float((query @ document.T).amax(dim=1).mean().item())


class CrossEncoder:
    def __init__(self, device: str, local_files_only: bool) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        options = {"revision": CROSS_REVISION, "local_files_only": local_files_only}
        self.tokenizer = AutoTokenizer.from_pretrained(CROSS_MODEL, **options)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            CROSS_MODEL, use_safetensors=True, **options
        ).to(device).float().eval()
        self.device = device

    def lengths(self, query: str, passages: list[str]) -> list[int]:
        return [
            len(tokens) for tokens in self.tokenizer(
                [query] * len(passages), passages, truncation=False, verbose=False
            )["input_ids"]
        ]

    def score(self, query: str, passages: list[str]) -> list[float]:
        import torch

        inputs = self.tokenizer(
            [query] * len(passages), passages, truncation=True, max_length=MAX_TOKENS,
        )
        ordering = sorted(range(len(passages)), key=lambda index: len(inputs["input_ids"][index]))
        scores = [0.0] * len(passages)
        with torch.inference_mode():
            for start in range(0, len(ordering), BATCH_SIZE):
                batch_indexes = ordering[start:start + BATCH_SIZE]
                batch = self.tokenizer.pad(
                    [{key: values[index] for key, values in inputs.items()} for index in batch_indexes],
                    return_tensors="pt", verbose=False,
                ).to(self.device)
                logits = self.model(**batch).logits.reshape(-1).cpu().tolist()
                for index, score in zip(batch_indexes, logits, strict=True):
                    scores[index] = score
        return scores


class MaxSim:
    def __init__(self, device: str, local_files_only: bool) -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel, AutoTokenizer

        options = {"revision": MODEL_REVISION, "local_files_only": local_files_only}
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **options)
        self.model = AutoModel.from_pretrained(
            MODEL_ID, use_safetensors=False, **options
        ).to(device).float().eval()
        projection = hf_hub_download(MODEL_ID, "colbert_linear.pt", **options)
        state = torch.load(projection, map_location="cpu", weights_only=True)
        output_dim, input_dim = state["weight"].shape
        self.projection = torch.nn.Linear(input_dim, output_dim, bias="bias" in state)
        self.projection.load_state_dict(state)
        self.projection.to(device).float().eval()
        self.device = device

    def lengths(self, texts: list[str]) -> list[int]:
        return [
            len(tokens) for tokens in self.tokenizer(texts, truncation=False, verbose=False)["input_ids"]
        ]

    def encode(self, texts: list[str]):
        import torch

        vectors = []
        with torch.inference_mode():
            for start in range(0, len(texts), BATCH_SIZE):
                batch = self.tokenizer(
                    texts[start:start + BATCH_SIZE], padding=True, truncation=True,
                    max_length=MAX_TOKENS, return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**batch).last_hidden_state[:, 1:]
                projected = torch.nn.functional.normalize(self.projection(hidden), dim=-1)
                vectors.extend(
                    row[mask.bool()].contiguous()
                    for row, mask in zip(projected, batch["attention_mask"][:, 1:], strict=True)
                )
        return vectors
