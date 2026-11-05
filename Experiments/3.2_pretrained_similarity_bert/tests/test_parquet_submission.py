from pathlib import Path
import csv
import json
import sys
import tempfile
import unittest
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import catalogue_pipeline as cp

class ParquetSubmissionChecks(unittest.TestCase):
    def write_csv(self,path,rows):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['behaviour_id','article_ids','similarity_scores'])
            for key,ids in rows:writer.writerow([key,json.dumps(ids),json.dumps([.5]*len(ids))])

    def test_native_list_preserves_rank_order_big_ids_and_empty_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'submission_top_200.csv'
            rows=[('7:10',[9007199254740993,42,3]),('8:10',[])]
            self.write_csv(path,rows)
            report=cp.write_shortlist_parquet(path)
            self.assertEqual(report['rows'],2)
            table=pq.read_table(path.with_suffix('.parquet'))
            self.assertEqual(table.column_names,['behaviour_id','article_ids'])
            self.assertEqual(table.schema.field('article_ids').type,pa.list_(pa.int64()))
            self.assertEqual(table.to_pylist(),[{'behaviour_id':key,'article_ids':ids} for key,ids in rows])
            self.assertEqual(list(cp.shortlist_parquet_rows(path.with_suffix('.parquet'))),table.to_pylist())

    def test_duplicate_requests_or_articles_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.csv'
            for rows in [[('7:10',[1]),('7:10',[2])],[('7:10',[1,1])]]:
                self.write_csv(path,rows)
                with self.assertRaisesRegex(ValueError,'duplicate|Duplicate'):cp.write_shortlist_parquet(path)
                self.assertFalse(path.with_suffix('.parquet').exists())
            self.write_csv(path,[('7:10',[1.5])])
            with self.assertRaises(ValueError):cp.write_shortlist_parquet(path)

    def test_existing_run_conversion_does_not_rescore_and_updates_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);work=root/'work';saved=root/'saved';saved.mkdir()
            path=work/'rankings/validation/submission_top_200.csv'
            self.write_csv(path,[('7:10',[9,3,7])])
            original=path.read_bytes()
            cp.save_json(work/'run_config.json',{'exports':{'validation':{'shortlist':'rankings/validation/submission_top_200.csv'}}})
            report=cp.export_parquet_submissions(work,saved)
            self.assertEqual(report['validation']['rows'],1)
            self.assertEqual(path.read_bytes(),original)
            copied=saved/'rankings/validation/submission_top_200.parquet'
            self.assertEqual(pq.read_table(copied).to_pylist()[0]['article_ids'],[9,3,7])
            record=json.loads((saved/'run_config.json').read_text())
            self.assertEqual(record['exports']['validation']['shortlist'],'rankings/validation/submission_top_200.parquet')
            self.assertEqual(record['exports']['validation']['shortlist_scores'],'rankings/validation/submission_top_200_scores.parquet')
            self.assertEqual(cp.export_parquet_submissions(work,saved)['validation']['rows'],1)

if __name__=='__main__':unittest.main()
