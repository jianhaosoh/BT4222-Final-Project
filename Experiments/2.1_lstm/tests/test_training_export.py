"""Offline checks of the training-export notebook; no fitted models are loaded."""
import ast
import json
from pathlib import Path
import tempfile
import unittest

NOTEBOOK = Path(__file__).resolve().parents[1] / '2.1.3_training_export.ipynb'


class TrainingExportTests(unittest.TestCase):
    def functions(self):
        self.assertTrue(NOTEBOOK.is_file(), 'Training-export notebook has not been prepared')
        book = json.loads(NOTEBOOK.read_text())
        nodes = []
        for cell in book['cells']:
            if cell['cell_type'] == 'code':
                nodes.extend(n for n in ast.parse(''.join(cell['source'])).body
                             if isinstance(n, ast.FunctionDef) and n.name == 'stage_training_metadata')
        self.assertEqual(len(nodes), 1)
        namespace = {'Path': Path, 'json': json}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
        return namespace['stage_training_metadata']

    def test_training_metadata_preserves_validation_holdout_and_original_model(self):
        stage = self.functions()
        with tempfile.TemporaryDirectory() as tmp:
            source, work = Path(tmp)/'source', Path(tmp)/'work'
            source.mkdir(); work.mkdir()
            config = {'catalogue_exports': {'validation': {'requests': 29110}, 'test': {'requests': 29142}},
                      'catalogue_by_split': {'validation': {'articles': 20738}, 'test': {'articles': 20738}},
                      'selection': {'epoch': 6}, 'model': {'lookup_articles': 125541},
                      'evaluation_scope': {'splits': ['validation','test'], 'training_exports_retained': False}}
            initial = {'run_config.json': config,
                       'export_manifest.json': {'splits': config['catalogue_exports'],
                                                'catalogue_by_split': config['catalogue_by_split']},
                       'metrics.json': {'full_validation_catalogue': {'recall': .9161}, 'test': {'scope': 'holdout'}},
                       'manifest.json': {'model/model_state.pt': 123},
                       'submission_artifacts.json': {'removed_files': ['rankings/train/submission_top_200.parquet',
                                                                       'model/code/__pycache__/old.pyc']}}
            for rel, obj in initial.items():
                (source/rel).write_text(json.dumps(obj))
            before = {p.name:p.read_bytes() for p in source.iterdir()}
            data_paths = ['rankings/train/submission_top_200.parquet',
                          'predictions/train/submission_article_scores.parquet',
                          'predictions/train/eligibility_manifest.parquet', 'catalogue/train.json',
                          'evaluation/train_catalogue.json', 'evaluation/training_export.json']
            for rel in data_paths:
                p = work/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
            record = {'requests': 232887, 'metrics': {'known_click_recall@100': .9}}
            files = stage(source, work, record, {'articles': 20738},
                          {'scope':'training in-sample; no tuning','source_model_sha256':'fixture'}, data_paths)
            self.assertEqual(before, {p.name:p.read_bytes() for p in source.iterdir()})
            current = json.loads((work/'run_config.json').read_text())
            for name in ['validation','test']:
                self.assertEqual(current['catalogue_exports'][name], config['catalogue_exports'][name])
                self.assertEqual(current['catalogue_by_split'][name], config['catalogue_by_split'][name])
            self.assertEqual(current['model'], config['model'])
            self.assertEqual(current['selection'], config['selection'])
            self.assertEqual(current['catalogue_by_split']['train']['articles'],20738)
            self.assertEqual(current['evaluation_scope']['splits'],['validation','test'])
            self.assertTrue(current['evaluation_scope']['training_exports_retained'])
            metrics = json.loads((work/'metrics.json').read_text())
            self.assertEqual(metrics['test'], initial['metrics.json']['test'])
            self.assertEqual(metrics['full_validation_catalogue'], initial['metrics.json']['full_validation_catalogue'])
            status = json.loads((work/'submission_artifacts.json').read_text())
            self.assertEqual(status['removed_files'], ['model/code/__pycache__/old.pyc'])
            manifest = json.loads((work/'manifest.json').read_text())
            self.assertEqual(manifest['model/model_state.pt'],123)
            for rel in files:
                if rel != 'manifest.json':self.assertEqual(manifest[rel],(work/rel).stat().st_size)
            self.assertFalse(any('/validation/' in p or '/test/' in p or p.startswith('model/') for p in files))

    def test_notebook_is_one_prediction_path_without_training_or_tuning(self):
        self.functions()
        book = json.loads(NOTEBOOK.read_text())
        code = '\n'.join(''.join(c['source']) for c in book['cells'] if c['cell_type']=='code')
        self.assertIn("INDEX = D.build_catalogue_index('train', 'SMALL'", code)
        self.assertIn('stage_training_metadata', code)
        self.assertIn('D._copy_atomic', code)
        self.assertNotIn('M.fit(', code)
        self.assertNotIn('optuna', code.lower())
        self.assertNotIn('features_validation', code)
        self.assertNotIn('features_test', code)
        self.assertTrue(all(c.get('execution_count') is None and not c.get('outputs')
                            for c in book['cells'] if c['cell_type']=='code'))


if __name__ == '__main__':
    unittest.main()
