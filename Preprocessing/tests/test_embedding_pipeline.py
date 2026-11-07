"""Offline checks: fixtures and cached vectors only; never contact Gemini."""
import contextlib
import importlib.util
import io
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

PATH = Path(__file__).resolve().parents[1] / 'embedding_pipeline.py'
spec = importlib.util.spec_from_file_location('embedding_pipeline', PATH)
ep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ep)


class EmbeddingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.articles = pd.DataFrame({
            'article_id': [3, 1, 2],
            'title': [None, ' One ', 'Two'],
            'subtitle': [None, 'Subtitle', None],
            'body': [None, 'Label\n--- SPLIT ELEMENT ---\n' + 'A' * 45, None],
        })
        for split in ['train_val', 'test']:
            (self.root / split).mkdir()
            self.articles.to_parquet(self.root / split / 'articles.parquet', index=False)
        self.images = self.root / 'supplied_images.parquet'
        pd.DataFrame({'article_id': [1], 'image_embedding': [np.ones(1024, np.float32)]}).to_parquet(self.images)
        self.addCleanup(patch.stopall)
        patch.object(socket, 'create_connection', side_effect=AssertionError('Network forbidden')).start()

    def populate_cache(self):
        cache = self.root / 'seed.parquet'
        vectors = {}
        for article in self.articles.to_dict('records'):
            for name in ep.VARIANTS:
                text = ep.make_variant_text(article, name)
                if text:
                    key = ep.input_hash(text, 'gemini-embedding-2', 768)
                    vectors[key] = np.full(768, 1 / np.sqrt(768), np.float32)
        ep.save_cache(vectors, cache)
        return cache

    def test_text_variants_preserve_existing_format_and_intro_rule(self):
        article = self.articles.iloc[1]
        self.assertEqual(ep.make_variant_text(article, 'title_subtitle'), 'title: One\nsubtitle: Subtitle')
        self.assertEqual(ep.make_variant_text(article, 'title_subtitle_intro'),
                         'title: One\nsubtitle: Subtitle\nintro: ' + 'A' * 45)
        self.assertEqual(ep.make_variant_text(article, 'title_subtitle_body'), ep.make_text(article))
        self.assertIsNone(ep.make_variant_text(self.articles.iloc[0], 'title_subtitle_body'))

    def test_dry_run_does_not_create_client_read_key_or_write_outputs(self):
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with patch.object(ep, 'create_client', side_effect=AssertionError('API forbidden')), \
             patch.dict('os.environ', {'GEMINI_API_KEY': 'DO_NOT_READ'}), contextlib.redirect_stdout(io.StringIO()):
            report = ep.generate_embeddings(self.root, self.images, dry_run=True)
        self.assertEqual(report['articles'], 3)
        self.assertGreater(report['new_api_documents'], 0)
        after = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_cached_run_publishes_complete_matching_catalogues_and_all_variants(self):
        seed = self.populate_cache()
        with patch.object(ep, 'create_client', side_effect=AssertionError('API forbidden')), \
             contextlib.redirect_stdout(io.StringIO()):
            report = ep.generate_embeddings(self.root, self.images, cache_sources=[seed])
        self.assertEqual(report['new_api_documents'], 0)
        base = self.root / 'train_val/article embeddings'
        full = self.root / 'feature_engineering/text_embedding_variants/title_subtitle_body/train_val/article embeddings'
        self.assertEqual((base / 'text_embeddings.parquet').read_bytes(), (full / 'text_embeddings.parquet').read_bytes())
        for name in ep.VARIANTS:
            variant = self.root / 'feature_engineering/text_embedding_variants' / name
            a = variant / 'train_val/article embeddings/text_embeddings.parquet'
            b = variant / 'test/article embeddings/text_embeddings.parquet'
            self.assertEqual(a.read_bytes(), b.read_bytes())
            frame = pd.read_parquet(a)
            self.assertEqual(frame.article_id.tolist(), [1, 2, 3])
            self.assertEqual(frame.status.tolist(), ['ok', 'ok', 'missing_input'])
            self.assertFalse(frame.has_text_embedding.iloc[-1])
            self.assertIsNone(frame.text_embedding.iloc[-1])
        image = pd.read_parquet(base / 'image_embeddings.parquet')
        self.assertEqual(image.article_id.tolist(), [1, 2, 3])
        self.assertEqual(image.has_image_embedding.tolist(), [True, False, False])
        with patch.object(ep, 'create_client', side_effect=AssertionError('API forbidden')), \
             contextlib.redirect_stdout(io.StringIO()):
            rerun = ep.generate_embeddings(self.root, dry_run=True)
        self.assertEqual(rerun['new_api_documents'], 0)

    def test_different_catalogue_is_rejected_before_publishing(self):
        changed = self.articles.copy()
        changed.loc[0, 'article_id'] = 99
        changed.to_parquet(self.root / 'test/articles.parquet', index=False)
        with self.assertRaisesRegex(ValueError, 'same SMALL catalogue'):
            ep.generate_embeddings(self.root, self.images, dry_run=True)
        self.assertFalse((self.root / 'feature_engineering').exists())

    def test_existing_nullable_fixed_size_vectors_can_be_read(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.table({
            'article_id': [1, 2],
            'image_embedding': pa.array([np.ones(1024, np.float32), None],
                                        type=pa.list_(pa.float32(), 1024)),
        }), self.images)
        with contextlib.redirect_stdout(io.StringIO()):
            report = ep.generate_embeddings(self.root, self.images, dry_run=True)
        self.assertEqual(report['image_vectors'], 1)

    def test_other_dimensions_do_not_reuse_the_768_dimension_cache(self):
        seed = self.populate_cache()
        with contextlib.redirect_stdout(io.StringIO()):
            report = ep.generate_embeddings(self.root, self.images, dimension=1536,
                                            cache_sources=[seed], dry_run=True)
        self.assertGreater(report['new_api_documents'], 0)

    def test_fake_client_generates_unique_documents_once_then_reuses_cache(self):
        from types import SimpleNamespace as NS
        calls = []
        class Models:
            def count_tokens(self, *, model, contents):
                return NS(total_tokens=len(contents))
            def embed_content(self, *, model, contents, config):
                calls.append(contents)
                self_dimension = config['output_dimensionality']
                return NS(embeddings=[NS(values=np.ones(self_dimension, np.float32))])
        client = NS(models=Models(), close=lambda: None)
        with patch.object(ep, 'create_client', return_value=client), contextlib.redirect_stdout(io.StringIO()):
            result = ep.generate_embeddings(self.root, self.images, workers=2)
        self.assertEqual(len(calls), result['new_api_documents'])
        self.assertEqual(len(calls), len(set(calls)))
        with patch.object(ep, 'create_client', side_effect=AssertionError('No second client')), \
             contextlib.redirect_stdout(io.StringIO()):
            result = ep.generate_embeddings(self.root, self.images)
        self.assertEqual(result['new_api_documents'], 0)

    def test_chunking_keeps_all_text_and_uses_token_weighted_pooling(self):
        from types import SimpleNamespace as NS
        calls = []
        class Models:
            def count_tokens(self, *, model, contents):
                return NS(total_tokens=len(contents))
            def embed_content(self, *, model, contents, config):
                calls.append(contents)
                vector = [1., 0.] if contents.startswith('a') else [0., 1.]
                return NS(embeddings=[NS(values=vector)])
        client = NS(models=Models())
        with patch.object(ep, 'MAX_TEXT_TOKENS', 5):
            vector, status, error = ep.embed_text(client, 'aaaa bbbb', 'fixture', 2)
        self.assertEqual(''.join(calls), 'aaaa bbbb')
        expected = np.array([5., 4.]); expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(vector, expected, atol=1e-6)
        self.assertEqual(status, 'ok')
        self.assertIsNone(error)


if __name__ == '__main__':
    unittest.main()
