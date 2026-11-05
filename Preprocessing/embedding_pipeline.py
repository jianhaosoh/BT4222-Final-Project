"""Generate regular-API Gemini text embeddings and join supplied EB-NeRD image vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from google import genai
from google.genai import errors


PROJECT_DIR = Path(__file__).resolve().parents[1]
MAX_TEXT_TOKENS = 8192


def make_text(article) -> str | None:
    """Combine the available Danish title, subtitle and body without translating."""
    fields = []
    for name in ("title", "subtitle", "body"):
        value = article.get(name)
        if isinstance(value, str) and value.strip():
            fields.append(f"{name}: {value.strip()}")
    return "\n".join(fields) or None


def input_hash(text: str, model: str, dimension: int) -> str:
    content = json.dumps([model, dimension, "article-text-v1", text],
                         ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def validate_vector(values, dimension: int):
    vector = np.asarray(values, dtype="float32")
    if vector.shape != (dimension,) or not np.isfinite(vector).all() or not np.any(vector):
        raise ValueError(f"Expected a finite, nonzero {dimension}-dimensional vector")
    return vector


def text_chunks(client, text: str, model: str) -> list[tuple[str, int]]:
    """Keep all text, splitting oversized inputs near the middle at natural breaks."""
    count = client.models.count_tokens(model=model, contents=text).total_tokens
    if not isinstance(count, int) or count < 1:
        raise ValueError("Gemini returned an invalid token count")
    if count <= MAX_TEXT_TOKENS:
        return [(text, count)]
    if len(text) < 2:
        raise ValueError("Cannot split input within the model's token limit")
    start, end = len(text) // 4, 3 * len(text) // 4
    boundary = text.rfind("\n", start, end)
    if boundary < 0:
        boundary = text.rfind(" ", start, end)
    boundary = boundary + 1 if boundary >= 0 else len(text) // 2
    return (text_chunks(client, text[:boundary], model)
            + text_chunks(client, text[boundary:], model))


def embed_text(client, text: str, model: str, dimension: int):
    """Embed whole text or pool long-text chunks; retry temporary API errors."""
    for attempt in range(5):
        try:
            chunks = text_chunks(client, text, model)
            vectors = []
            for chunk, _ in chunks:
                result = client.models.embed_content(
                    model=model, contents=chunk, config={"output_dimensionality": dimension},
                )
                if result.embeddings is None or len(result.embeddings) != 1:
                    raise ValueError("Expected exactly one text embedding")
                vectors.append(validate_vector(result.embeddings[0].values, dimension))
            if len(vectors) == 1:
                return vectors[0], "ok", None
            # Weight by chunk length so a short final chunk does not dominate.
            vector = np.average(np.asarray(vectors, dtype="float64"), axis=0,
                                weights=[count for _, count in chunks])
            norm = np.linalg.norm(vector)
            if not np.isfinite(norm) or norm == 0:
                raise ValueError("Cannot normalise pooled text embedding")
            return validate_vector(vector / norm, dimension), "ok", None
        except errors.APIError as error:
            if error.code in (400, 401, 403, 404):
                raise RuntimeError(f"Gemini HTTP {error.code}; check API key, model and input") from None
            if (error.code == 429 or error.code >= 500) and attempt < 4:
                time.sleep(min(2 ** (attempt + 1), 60))
                continue
            return None, "api_error", f"http_{error.code}"


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def save_text(rows: list[dict], path: Path) -> None:
    frame = pd.DataFrame(rows, columns=["article_id", "text_embedding", "has_text_embedding",
                                      "status", "error_code", "input_hash"])
    frame["article_id"] = frame["article_id"].astype("int32")
    save_frame(frame, path)


def save_cache(cached: dict, path: Path) -> None:
    if cached:
        save_frame(pd.DataFrame({"input_hash": list(cached),
                                 "text_embedding": list(cached.values())}), path)


def generate_embeddings(articles_path: Path | list[Path], image_path: Path, output_dir: Path, *,
                        limit: int | None = None, dry_run: bool = False, client=None,
                        model: str = "gemini-embedding-2", dimension: int = 768):
    if dimension not in (768, 1536, 3072) or (limit is not None and limit < 1):
        raise ValueError("Use dimension 768, 1536 or 3072, and a positive limit")
    paths = [articles_path] if isinstance(articles_path, Path) else articles_path
    if output_dir.resolve() in {path.resolve().parent for path in [*paths, image_path]}:
        raise ValueError("Save generated vectors separately from source data")
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        if frame["article_id"].isna().any() or frame["article_id"].duplicated().any():
            raise ValueError("Each catalogue must have non-null, unique article IDs")
        frames.append(frame)
    # Overlapping IDs keep the first supplied catalogue snapshot.
    articles = pd.concat(frames, ignore_index=True).drop_duplicates("article_id", keep="first").sort_values("article_id")
    if limit is not None:
        articles = articles.head(limit)
    images = pd.read_parquet(image_path, columns=["article_id", "image_embedding"])
    images = articles[["article_id"]].merge(images, on="article_id", how="left", validate="one_to_one")
    images["has_image_embedding"] = images["image_embedding"].map(lambda value: isinstance(value, (list, np.ndarray)))
    images["image_embedding"] = [validate_vector(value, 1024) if present else None
                                 for value, present in zip(images["image_embedding"], images["has_image_embedding"])]
    images["status"] = np.where(images["has_image_embedding"], "ok", "missing_input")
    images["error_code"] = None
    if dry_run:
        print(f"Articles: {len(articles)}; nonempty text: {articles.apply(make_text, axis=1).notna().sum()}; "
              f"image vectors: {images['has_image_embedding'].sum()}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    images.to_parquet(output_dir / "image_embeddings.parquet", index=False)
    text_path = output_dir / "text_embeddings.parquet"
    cache_path = output_dir / "text_cache.parquet"
    cached = {}
    for path in (cache_path, text_path):
        if path.exists():
            previous = pd.read_parquet(path)
            if "input_hash" in previous:
                for _, row in previous.iterrows():
                    if row.get("status", "ok") == "ok":
                        values = row["text_embedding"]
                        cached[row["input_hash"]] = validate_vector(values, len(values))
    # Preserve the full cache even when this run selects only a small sample.
    save_cache(cached, cache_path)
    rows, owned_client = [], False
    try:
        for _, article in articles.iterrows():
            text = make_text(article)
            key = input_hash(text, model, dimension) if text else None
            vector, status, error = None, "missing_input", None
            if text:
                vector = cached.get(key)
                if vector is None:
                    if client is None:
                        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"), vertexai=False)
                        owned_client = True
                    vector, status, error = embed_text(client, text, model, dimension)
                    if vector is not None:
                        cached[key] = vector
                else:
                    vector = validate_vector(vector, dimension)
                    status = "ok"
            rows.append({"article_id": int(article["article_id"]), "text_embedding": vector,
                         "has_text_embedding": vector is not None, "status": status,
                         "error_code": error, "input_hash": key})
            if len(rows) % 100 == 0:
                save_cache(cached, cache_path)
                save_text(rows, text_path)
                print(f"Processed {len(rows)}/{len(articles)} articles", flush=True)
    finally:
        if rows:
            save_cache(cached, cache_path)
            save_text(rows, text_path)
        if owned_client:
            client.close()
    metadata = {
        "source_catalogues": [str(path) for path in paths],
        "articles": len(articles),
        "text_model": model, "text_dimension": dimension,
        "image_source": "ebnerd_supplied", "image_dimension": 1024,
        "counts": {
            "text": pd.Series([row["status"] for row in rows]).value_counts().to_dict(),
            "image": images["status"].value_counts().to_dict(),
        },
        "long_text_processing": {
            "max_chunk_tokens": MAX_TEXT_TOKENS,
            "pooling": "token-count-weighted mean followed by L2 normalisation",
        },
    }
    if len(paths) == 1:
        with paths[0].open("rb") as source:
            metadata["catalogue_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved separate text and image vectors in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", required=True, type=Path, nargs="+",
                        help="Article catalogues, with train/validation before test")
    parser.add_argument("--image-embeddings", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_DIR / ".env", override=False)
    generate_embeddings(args.articles, args.image_embeddings,
                        args.output_dir or PROJECT_DIR / "data/embeddings" / args.articles[0].parent.name,
                        limit=args.limit, dry_run=args.dry_run,
                        model=os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2"),
                        dimension=int(os.environ.get("GEMINI_EMBEDDING_DIMENSION", "768")))


if __name__ == "__main__":
    main()
