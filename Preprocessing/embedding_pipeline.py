"""Generate and publish the complete SMALL catalogue and three Gemini text variants.

Use --shared-root for the mounted project folder. --dry-run reads local inputs
and caches only: it never loads an API key, creates a client, or writes outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa


PROJECT_DIR = Path(__file__).resolve().parents[1]
MAX_TEXT_TOKENS = 8192
VARIANTS = ("title_subtitle", "title_subtitle_intro", "title_subtitle_body")
MIN_INTRO_CHARS = 40
MARKER = re.compile(r"^-+\s*(EKSTRA\s+)?SPLIT ELEMENT\s*-+$")
INTRO_DEFINITION = ("First body paragraph with at least 40 characters, skipping SPLIT "
                    "ELEMENT markers; else the first paragraph.")
CHECKPOINT_EVERY = 500


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
        except Exception as error:
            code = getattr(error, "code", None)
            if not isinstance(code, int):
                raise
            if code in (400, 401, 403, 404):
                raise RuntimeError(f"Gemini HTTP {code}; check API key, model and input") from None
            if (code == 429 or code >= 500) and attempt < 4:
                time.sleep(min(2 ** (attempt + 1), 60))
                continue
            return None, "api_error", f"http_{code}"


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def intro_from_body(body) -> str | None:
    """Match the intro definition used for the final Drive variants."""
    if not isinstance(body, str):
        return None
    paragraphs = [line.strip() for line in body.split("\n")]
    paragraphs = [line for line in paragraphs if line and not MARKER.match(line)]
    return next((p for p in paragraphs if len(p) >= MIN_INTRO_CHARS),
                paragraphs[0] if paragraphs else None)


def make_variant_text(article, variant: str) -> str | None:
    if variant == "title_subtitle_body":
        return make_text(article)
    if variant not in VARIANTS:
        raise ValueError(f"Unknown text variant: {variant}")
    pairs = [("title", article.get("title")), ("subtitle", article.get("subtitle"))]
    if variant == "title_subtitle_intro":
        pairs.append(("intro", intro_from_body(article.get("body"))))
    fields = [f"{name}: {value.strip()}" for name, value in pairs
              if isinstance(value, str) and value.strip()]
    return "\n".join(fields) or None


def load_catalogues(shared_root: Path):
    """Step 1 supplies the complete, identical SMALL catalogue in both folders."""
    paths = {split: shared_root / split / "articles.parquet" for split in ("train_val", "test")}
    frames = {}
    for split, path in paths.items():
        frame = pd.read_parquet(path, columns=["article_id", "title", "subtitle", "body"])
        if frame.empty or frame.article_id.isna().any() or frame.article_id.duplicated().any():
            raise ValueError(f"Expected nonempty, unique, non-null article IDs: {path}")
        frames[split] = frame.sort_values("article_id").reset_index(drop=True)
    if not frames["train_val"].equals(frames["test"]):
        raise ValueError("Both folders must contain the same SMALL catalogue and text; run Step 1 first.")
    return frames["train_val"], paths


def read_vectors(path: Path, columns: list[str]):
    """Legacy Drive files contain nullable fixed-size lists that Arrow cannot decode."""
    try:
        return pd.read_parquet(path, columns=columns)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        import duckdb
        selected = ", ".join(f'"{column}"' for column in columns)
        with duckdb.connect() as connection:
            return connection.execute(f"SELECT {selected} FROM read_parquet(?)", [str(path)]).df()


def load_images(articles: pd.DataFrame, path: Path):
    images = read_vectors(path, ["article_id", "image_embedding"])
    images = articles[["article_id"]].merge(images, on="article_id", how="left", validate="one_to_one")
    images["has_image_embedding"] = images.image_embedding.map(lambda v: isinstance(v, (list, np.ndarray)))
    images["image_embedding"] = [validate_vector(v, 1024) if present else None
                                 for v, present in zip(images.image_embedding, images.has_image_embedding)]
    images["status"] = np.where(images.has_image_embedding, "ok", "missing_input")
    images["error_code"] = None
    return images


def load_cache(paths: list[Path], dimension: int):
    cached = {}
    for path in dict.fromkeys(paths):
        if not path.is_file():
            continue
        columns = read_vectors(path, ["input_hash", "text_embedding"])
        for key, values in zip(columns.input_hash, columns.text_embedding):
            if (isinstance(key, str) and isinstance(values, (list, np.ndarray))
                    and len(values) == dimension and key not in cached):
                cached[key] = validate_vector(values, dimension)
    return cached


def create_client():
    """Only reached when actual embedding work remains; never used by dry-run."""
    from dotenv import load_dotenv
    from google import genai
    load_dotenv(PROJECT_DIR / ".env", override=False)
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY in the environment or repository .env before generating new vectors.")
    return genai.Client(api_key=key, vertexai=False, http_options={"timeout": 120_000})


def embed_all(todo: dict, cached: dict, cache_path: Path, *, model: str, dimension: int, workers: int):
    """Reuse the original bounded concurrency, retries and successful-vector cache."""
    if not todo:
        return
    client = create_client()
    remaining = dict(todo)
    completed = 0
    try:
        for attempt in range(1, 4):
            failures = {}
            consecutive = 0
            with ThreadPoolExecutor(max_workers=workers) as pool:
                items = iter(remaining.items())
                active = {}
                while active or items is not None:
                    while items is not None and len(active) < workers * 2:
                        item = next(items, None)
                        if item is None:
                            items = None
                        else:
                            key, text = item
                            active[pool.submit(embed_text, client, text, model, dimension)] = key
                    if not active:
                        break
                    finished, _ = wait(active, return_when=FIRST_COMPLETED)
                    for future in finished:
                        key = active.pop(future)
                        try:
                            vector, _, error = future.result()
                        except RuntimeError:
                            for pending in active:
                                pending.cancel()
                            raise
                        except Exception as caught:
                            vector, error = None, type(caught).__name__
                        if vector is None:
                            failures[key] = error or "unknown"
                            consecutive += 1
                        else:
                            cached[key] = vector
                            completed += 1
                            consecutive = 0
                            if completed % CHECKPOINT_EVERY == 0:
                                save_cache(cached, cache_path)
                                print(f"Embedded {completed}/{len(todo)} unique documents", flush=True)
                    if consecutive >= 50:
                        for pending in active:
                            pending.cancel()
                        raise RuntimeError("Stopped after 50 consecutive failures; successful vectors remain cached.")
            if not failures:
                return
            remaining = {key: todo[key] for key in failures}
            print(f"Pass {attempt}: {len(failures)} documents failed; {sorted(set(failures.values()))}", flush=True)
            if attempt < 3:
                time.sleep(30 * attempt)
        raise RuntimeError(f"{len(remaining)} documents failed; rerun to retry. Published vectors were not replaced.")
    finally:
        save_cache(cached, cache_path)
        client.close()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def generate_embeddings(shared_root: Path, image_path: Path | None = None, *,
                        dry_run: bool = False, workers: int = 10,
                        model: str = "gemini-embedding-2", dimension: int = 768,
                        cache_sources=()):
    """One workflow: full SMALL catalogue, all text variants, images and publication."""
    shared_root = Path(shared_root)
    if dimension not in (768, 1536, 3072) or workers < 1:
        raise ValueError("Use dimension 768, 1536 or 3072, and at least one worker")
    articles, catalogue_paths = load_catalogues(shared_root)
    image_path = Path(image_path) if image_path else shared_root / "train_val/article embeddings/image_embeddings.parquet"
    images = load_images(articles, image_path)
    variant_root = shared_root / "feature_engineering/text_embedding_variants"
    cache_path = variant_root / "text_cache.parquet"
    existing = [cache_path, shared_root / "train_val/article embeddings/text_embeddings.parquet"]
    existing += [variant_root / name / "train_val/article embeddings/text_embeddings.parquet" for name in VARIANTS]
    cached = load_cache(existing + list(map(Path, cache_sources)), dimension)
    prepared, todo, counts = {}, {}, {}
    for name in VARIANTS:
        texts = [make_variant_text(row, name) for row in articles.to_dict("records")]
        keys = [input_hash(text, model, dimension) if text else None for text in texts]
        prepared[name] = (texts, keys)
        new = {key: text for key, text in zip(keys, texts) if text and key not in cached and key not in todo}
        todo.update(new)
        counts[name] = {"ok": sum(text is not None for text in texts),
                        "missing_input": sum(text is None for text in texts)}
        print(f"{name}: {counts[name]['ok']:,} usable texts, {counts[name]['missing_input']} missing; "
              f"{len(new):,} additional unique documents to embed")
    report = {"articles": len(articles), "text_model": model, "text_dimension": dimension,
              "new_api_documents": len(todo), "variants": counts,
              "image_vectors": int(images.has_image_embedding.sum())}
    print(f"Complete shared catalogue: {len(articles):,} articles; {len(todo):,} uncached text documents. "
          "Long documents may require several API chunk calls.")
    if dry_run:
        print("Dry run only: no API client/key access, token-count calls, embedding calls or output writes.")
        return report

    # No published file changes until all nonempty texts have a valid vector.
    embed_all(todo, cached, cache_path, model=model, dimension=dimension, workers=workers)
    manifest = {**report, "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "reference_pipeline": "Preprocessing/embedding_pipeline.py",
                "intro_definition": INTRO_DEFINITION, "variants": {},
                "catalogues": {split: {"source": str(path), "articles": len(articles),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for split, path in catalogue_paths.items()},
                "long_text_processing": {"max_chunk_tokens": MAX_TEXT_TOKENS,
                    "pooling": "token-count-weighted mean followed by L2 normalisation"}}
    save_cache(cached, cache_path)
    for name, (texts, keys) in prepared.items():
        rows = [{"article_id": int(aid), "text_embedding": cached[key] if key else None,
                 "has_text_embedding": key is not None,
                 "status": "ok" if key else "missing_input", "error_code": None, "input_hash": key}
                for aid, key in zip(articles.article_id, keys)]
        first = variant_root / name / "train_val/article embeddings"
        save_text(rows, first / "text_embeddings.parquet")
        for split in ("train_val", "test"):
            directory = variant_root / name / split / "article embeddings"
            directory.mkdir(parents=True, exist_ok=True)
            if split == "test":
                shutil.copy2(first / "text_embeddings.parquet", directory / "text_embeddings.parquet")
            metadata = {"variant": name, "split": split, "articles": len(rows), "join_key": "article_id",
                        "text_model": model, "text_dimension": dimension, "counts": counts[name],
                        "source_catalogue": str(catalogue_paths[split]),
                        "catalogue_sha256": manifest["catalogues"][split]["sha256"],
                        "intro_definition": INTRO_DEFINITION if name == "title_subtitle_intro" else None,
                        "long_text_processing": manifest["long_text_processing"]}
            write_json(directory / "metadata.json", metadata)
            manifest["variants"].setdefault(name, {})[split] = {"rows": len(rows), "counts": counts[name]}

    full = variant_root / "title_subtitle_body/train_val/article embeddings/text_embeddings.parquet"
    base = shared_root / "train_val/article embeddings"
    base.mkdir(parents=True, exist_ok=True)
    shutil.copy2(full, base / "text_embeddings.parquet")
    save_frame(images, base / "image_embeddings.parquet")
    for split in ("train_val", "test"):
        directory = shared_root / split / "article embeddings"
        directory.mkdir(parents=True, exist_ok=True)
        if split == "test":
            for kind in ("text", "image"):
                shutil.copy2(base / f"{kind}_embeddings.parquet", directory / f"{kind}_embeddings.parquet")
        metadata = {"articles": len(articles), "text_model": model, "text_dimension": dimension,
                    "text_variant": "title_subtitle_body", "image_source": "ebnerd_supplied",
                    "image_dimension": 1024, "source_catalogues": [str(catalogue_paths[split])],
                    "catalogue_sha256": manifest["catalogues"][split]["sha256"],
                    "counts": {"text": counts["title_subtitle_body"], "image": images.status.value_counts().to_dict()},
                    "scope": "Complete EB-NeRD SMALL article catalogue shared by all project splits",
                    "long_text_processing": manifest["long_text_processing"]}
        if split == "test":
            metadata.update(source_embeddings="train_val/article embeddings", new_text_vectors_generated=0,
                            reused_text_vectors=counts["title_subtitle_body"]["ok"],
                            reused_image_vectors=int(images.has_image_embedding.sum()))
        write_json(directory / "metadata.json", metadata)
    write_json(variant_root / "manifest.json", manifest)
    print(f"Published all {len(articles):,} articles, three text variants and supplied images to {shared_root}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True,
                        help="Project folder containing train_val/articles.parquet and test/articles.parquet")
    parser.add_argument("--image-embeddings", type=Path,
                        help="Supplied EB-NeRD image file; defaults to existing train_val/article embeddings/image_embeddings.parquet")
    parser.add_argument("--cache-source", type=Path, action="append", default=[],
                        help="Additional text cache or text_embeddings.parquet to reuse; repeat as needed")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--model", default=os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2"))
    parser.add_argument("--dimension", type=int, default=int(os.environ.get("GEMINI_EMBEDDING_DIMENSION", "768")))
    parser.add_argument("--dry-run", action="store_true", help="Read inputs/cache and count work, without API calls or writes")
    args = parser.parse_args()
    generate_embeddings(args.shared_root, args.image_embeddings, dry_run=args.dry_run,
                        workers=args.workers, model=args.model, dimension=args.dimension,
                        cache_sources=args.cache_source)


if __name__ == "__main__":
    main()
