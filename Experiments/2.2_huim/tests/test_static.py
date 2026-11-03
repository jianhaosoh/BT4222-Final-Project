"""Lightweight checks on synthetic inputs: no shared data, no optimiser steps.

Run from Experiments/2.2_huim:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huim import catalogue as C  # noqa: E402
from huim import features as F  # noqa: E402
from huim.metrics import impression_metrics, recall_summary  # noqa: E402
from huim.model import ModelConfig, build_model  # noqa: E402
from huim.validate import validate_export, validate_recorded  # noqa: E402


def synthetic_world(seed=0, n_articles=60, n_users=5, n_requests=12, history_len=6):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2023-05-18 07:00:00")
    ids = np.sort(rng.choice(np.arange(9_000_000, 9_100_000), n_articles, replace=False))
    pub = base + pd.to_timedelta(rng.integers(-500, 200, n_articles), unit="h")
    articles = pd.DataFrame({
        "article_id": ids, "published_time": pub,
        "premium": rng.random(n_articles) < 0.3,
        "category": rng.choice([118, 140, 142, 999], n_articles).astype("int16"),
        "subcategory": [list(rng.choice([1, 2, 3], rng.integers(0, 3))) for _ in range(n_articles)],
        "article_type": rng.choice(["article_default", "article_feature"], n_articles),
        "topics": [list(rng.choice(["Kendt", "Sport", "Politik", "Nyt"], rng.integers(0, 3), replace=False))
                   for _ in range(n_articles)],
        "sentiment_label": rng.choice(["Positive", "Negative", "Neutral"], n_articles),
        "body_word_count": rng.integers(50, 900, n_articles).astype(float),
        "_source": "train_val",
    })
    articles.loc[3, "published_time"] = pd.NaT  # missing publication time -> never eligible
    vocab = {"category": ["118", "140", "142"], "article_type": ["article_default", "article_feature"],
             "topics": ["Kendt", "Politik", "Sport"], "device_type": ["1", "2", "3"]}
    users = np.arange(100, 100 + n_users)
    ev = []
    for u in users:
        for k in range(rng.integers(0, 10)):
            ev.append((u, int(rng.choice(ids)), base - pd.Timedelta(hours=int(rng.integers(1, 400))),
                       float(rng.integers(1, 90)), float(rng.integers(0, 100)) if rng.random() < .8 else np.nan, k))
    events = pd.DataFrame(ev, columns=["user_id", "article_id", "time", "read_time", "scroll", "event_index"])
    events = events.sort_values(["user_id", "time", "event_index"]).reset_index(drop=True)
    req = []
    for i in range(n_requests):
        u = int(rng.choice(users))
        t = base + pd.Timedelta(minutes=int(rng.integers(0, 6000)))
        slate = rng.choice(ids, int(rng.integers(5, 9)), replace=False)
        clicked = slate[:1]
        req.append({"user_id": u, "impression_id": 1000 + i, "impression_time": t, "article_ids_inview": slate,
                    "article_ids_clicked": clicked, "device_type": int(rng.choice([1, 2, 0])),
                    "is_sso_user": bool(rng.random() < .5), "is_subscriber": bool(rng.random() < .3)})
    beh = pd.DataFrame(req).sort_values(["user_id", "impression_id"]).reset_index(drop=True)
    beh["behaviour_id"] = beh.user_id.astype(str) + ":" + beh.impression_id.astype(str)
    pairs = beh[["user_id", "impression_id", "article_ids_inview", "article_ids_clicked"]].copy()
    rows = [(r.user_id, r.impression_id, int(a), int(a in set(r.article_ids_clicked)))
            for r in pairs.itertuples() for a in r.article_ids_inview]
    pairs = pd.DataFrame(rows, columns=["user_id", "impression_id", "candidate_article_id", "clicked"])
    text_ids = ids[::2]
    text = rng.normal(size=(len(text_ids), 8)).astype(np.float32)
    image_ids = ids[::3]
    image = rng.normal(size=(len(image_ids), 6)).astype(np.float32)
    pre = F.Preprocessor.fit(vocab, articles, ids, events, history_len)
    art = F.build_article_table(articles, pre, text_ids, text, image_ids, image)
    ctx = F.build_split_context("validation", beh, events, users, art, pre, ids, pairs)
    F.fit_dense_scalers(pre, ctx, art)
    return articles, beh, events, pairs, pre, art, ctx


def make_model(pre, art, **kw):
    cfg = ModelConfig(**pre.sizes(), text_dim=art.report["text_dim"], image_dim=art.report["image_dim"],
                      d_model=16, text_proj=4, image_proj=4, attention_hidden=(8, 4), feature_hidden=8,
                      head_hidden=(16, 8), recency_edges_h=pre.state["recency_edges_h"], **kw)
    torch.manual_seed(0)
    return build_model(cfg, pre).eval()


class TestMetrics(unittest.TestCase):
    def test_auc_ties_half_credit(self):
        m = impression_metrics(np.zeros(4), np.array([1, 0, 0, 1]), np.array([0.5, 0.5, 0.1, 0.9]))
        # pairs (pos, neg): (0.5,0.5)=0.5, (0.5,0.1)=1, (0.9,*)=1,1 -> 3.5/4
        self.assertAlmostEqual(m["mean_impression_auc"], 0.875)

    def test_ndcg_expected_ties_brute_force(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            n = int(rng.integers(2, 8))
            y = rng.integers(0, 2, n)
            if y.sum() == 0:
                y[0] = 1
            s = rng.integers(0, 3, n).astype(float)
            vals = []
            for perm in itertools.permutations(range(n)):
                # order by score desc, ties by the permutation
                key = sorted(range(n), key=lambda i: (-s[i], perm.index(i)))
                k = min(5, n)
                dcg = sum(y[key[p]] / np.log2(p + 2) for p in range(k))
                idcg = sum(1 / np.log2(p + 2) for p in range(min(k, y.sum())))
                vals.append(dcg / idcg)
            m = impression_metrics(np.zeros(n), y, s)
            self.assertAlmostEqual(m["mean_impression_ndcg@5"], float(np.mean(vals)), places=9)

    def test_exclusions_and_macro(self):
        g = np.array([0, 0, 1, 1, 2, 2])
        y = np.array([1, 0, 1, 1, 0, 0])
        m = impression_metrics(g, y, np.array([2, 1, 1, 2, 1, 1]))
        self.assertEqual(m["auc_eligible_impressions"], 1)
        self.assertEqual(m["ndcg_excluded_no_positive"], 1)

    def test_recall(self):
        man = pd.DataFrame({"n_clicked": [2, 1, 0], "n_clicked_eligible": [1, 1, 0], "n_clicked_in_shortlist": [1, 0, 0],
                            "n_clicked_not_in_catalogue": [1, 0, 0], "n_clicked_missing_pub": [0, 0, 0],
                            "n_clicked_future": [0, 0, 0]})
        r = recall_summary(man)
        self.assertAlmostEqual(r["known_click_recall@100"], (0.5 + 0) / 2)
        self.assertAlmostEqual(r["eligible_only_recall@100"], (1 + 0) / 2)


class TestRanking(unittest.TestCase):
    def test_stable_tie_break_ascending_id(self):
        logits = torch.tensor([[1.0, 3.0, 3.0, 2.0, 3.0]])
        elig = torch.tensor([[True, True, False, True, True]])
        order, lengths = C.CatalogueScorer.rank(logits, elig, k=3)
        self.assertEqual(order[0, :3].tolist(), [1, 4, 3])  # ties 1 and 4 by position; 2 ineligible
        self.assertEqual(int(lengths[0]), 3)

    def test_empty_pool(self):
        order, lengths = C.CatalogueScorer.rank(torch.zeros(1, 4), torch.zeros(1, 4, dtype=torch.bool))
        self.assertEqual(int(lengths[0]), 0)


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.articles, cls.beh, cls.events, cls.pairs, cls.pre, cls.art, cls.ctx = synthetic_world()

    def logits(self, model, ctx):
        rp = torch.from_numpy(ctx.recorded["req_pos"])
        ai = torch.from_numpy(ctx.recorded["art_idx"])
        with torch.no_grad():
            return model(ctx.tensors, self.art.tensors, len(self.art.ids), rp, ai).numpy()

    def permuted_ctx(self, perm_seed):
        rng = np.random.default_rng(perm_seed)
        t = dict(self.ctx.tensors)
        H = t["hist_idx"].shape[1]
        perm = torch.from_numpy(rng.permutation(H))
        for k in ("hist_idx", "hist_s", "hist_read_bin", "hist_scroll_bin", "hist_mask"):
            t[k] = t[k][:, perm]
        return F.SplitContext(self.ctx.split, self.ctx.requests, t, self.ctx.catalogue_idx, self.ctx.recorded, {})

    def test_permutation_invariance_all_modes(self):
        for mode in ("softmax", "unnormalised", "uniform"):
            for fusion in ("concat", "gated"):
                model = make_model(self.pre, self.art, attention_mode=mode, fusion=fusion)
                a = self.logits(model, self.ctx)
                b = self.logits(model, self.permuted_ctx(3))
                self.assertTrue(np.isfinite(a).all())
                np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)

    def test_padding_is_masked(self):
        model = make_model(self.pre, self.art)
        t = dict(self.ctx.tensors)
        pad = ~t["hist_mask"]
        t["hist_idx"] = torch.where(pad, torch.full_like(t["hist_idx"], 5), t["hist_idx"])
        t["hist_read_bin"] = torch.where(pad, torch.full_like(t["hist_read_bin"], 2), t["hist_read_bin"])
        ctx2 = F.SplitContext(self.ctx.split, self.ctx.requests, t, self.ctx.catalogue_idx, self.ctx.recorded, {})
        np.testing.assert_allclose(self.logits(model, self.ctx), self.logits(model, ctx2), rtol=1e-5, atol=1e-5)

    def test_empty_history_finite(self):
        model = make_model(self.pre, self.art)
        t = dict(self.ctx.tensors)
        t["hist_mask"] = torch.zeros_like(t["hist_mask"])
        ctx2 = F.SplitContext(self.ctx.split, self.ctx.requests, t, self.ctx.catalogue_idx, self.ctx.recorded, {})
        self.assertTrue(np.isfinite(self.logits(model, ctx2)).all())

    def test_catalogue_path_matches_aligned_path_and_export_validates(self):
        model = make_model(self.pre, self.art)
        scorer = C.CatalogueScorer(model, self.ctx, self.art, torch.device("cpu"), pair_chunk=37)
        positions = np.arange(len(self.ctx.requests))
        logits = scorer.logits(positions)
        cat = self.ctx.catalogue_idx
        rp = torch.from_numpy(np.repeat(positions, len(cat)))
        ai = torch.from_numpy(np.tile(cat, len(positions)))
        with torch.no_grad():
            aligned = model(self.ctx.tensors, self.art.tensors, len(self.art.ids), rp, ai).reshape(len(positions), -1)
        np.testing.assert_allclose(logits.numpy(), aligned.numpy(), rtol=1e-5, atol=1e-5)
        rec_logits = self.logits(model, self.ctx)
        with tempfile.TemporaryDirectory() as tmp:
            clicks = C.click_eligibility(self.ctx, self.art)
            prog = C.export_split(scorer, Path(tmp), positions, write_full_scores=True, click_info=clicks,
                                  recorded_counts=None, request_block=5, checkpoint_every_blocks=1, log=lambda *_: None)
            self.assertTrue(prog["finished"])
            res = validate_export(self.ctx, self.art, Path(tmp), positions, full_scores=True, check_blocks=None,
                                  recorded_logits=rec_logits, score_tolerance=1e-4)
            self.assertTrue(res["passed"], res["errors"])
            self.assertIsNotNone(res["checks"]["recorded_vs_catalogue_max_abs_logit_diff"])
            ranks = pd.read_csv(Path(tmp) / "rankings/validation/submission_top_100.csv", dtype={"behaviour_id": str})
            first = json.loads(ranks.article_ids.iloc[0])
            self.assertTrue(all(isinstance(a, int) for a in first))
            path = Path(tmp) / "rec.csv"
            C.write_recorded_slate_scores(path, self.ctx, rec_logits)
            self.assertTrue(validate_recorded(path, self.ctx)["passed"])

    def test_resumed_export_identical(self):
        model = make_model(self.pre, self.art)
        scorer = C.CatalogueScorer(model, self.ctx, self.art, torch.device("cpu"), pair_chunk=64)
        positions = np.arange(len(self.ctx.requests))
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            C.export_split(scorer, Path(a), positions, write_full_scores=True, click_info=None, recorded_counts=None,
                           request_block=3, checkpoint_every_blocks=1, log=lambda *_: None)
            p1 = C.export_split(scorer, Path(b), positions, write_full_scores=True, click_info=None, recorded_counts=None,
                                request_block=3, checkpoint_every_blocks=1, log=lambda *_: None, max_blocks=2)
            self.assertFalse(p1["finished"])
            # simulate a crash after more bytes were written than the last checkpoint recorded
            with open(Path(b) / "predictions/validation/submission_click_probabilities.csv", "ab") as h:
                h.write(b"garbage,1,2,3\n")
            C.export_split(scorer, Path(b), positions, write_full_scores=True, click_info=None, recorded_counts=None,
                           request_block=3, checkpoint_every_blocks=1, log=lambda *_: None)
            for rel in ("predictions/validation/submission_click_probabilities.csv",
                        "rankings/validation/submission_top_100.csv"):
                self.assertEqual((Path(a) / rel).read_bytes(), (Path(b) / rel).read_bytes())
            # a file SHORTER than the recorded progress must not be zero-padded
            prog = Path(b) / "diagnostics/catalogue/validation_export_progress.json"
            state = json.loads(prog.read_text())
            state["finished"] = False
            prog.write_text(json.dumps(state))
            (Path(b) / "rankings/validation/submission_top_100.csv").write_bytes(b"x")
            with self.assertRaises(RuntimeError):
                C.export_split(scorer, Path(b), positions, write_full_scores=True, click_info=None,
                               recorded_counts=None, request_block=3, checkpoint_every_blocks=1, log=lambda *_: None)

    def test_pair_features_match_definitions(self):
        raw = F.pair_dense(self.ctx.tensors, self.art.tensors, torch.from_numpy(self.ctx.recorded["req_pos"]),
                           torch.from_numpy(self.ctx.recorded["art_idx"]), len(self.art.ids)).numpy()
        # previously_clicked must match the full fixed history
        ev = self.events
        hist = set(zip(ev.user_id, ev.article_id))
        users = self.ctx.requests.user_id.to_numpy()[self.ctx.recorded["req_pos"]]
        expect = np.array([(u, a) in hist for u, a in zip(users, self.ctx.recorded["article_id"])], float)
        np.testing.assert_array_equal(raw[:, -1], expect)
        # category preference = share of the user's history events in the candidate category
        cat_of = dict(zip(self.articles.article_id, self.articles.category))
        for j in range(len(users)):
            u, a = users[j], self.ctx.recorded["article_id"][j]
            e = ev[ev.user_id == u]
            want = (e.article_id.map(cat_of) == cat_of[a]).sum() / max(len(e), 1)
            self.assertAlmostEqual(raw[j, 7], want, places=5)



class TestDiagnostics(unittest.TestCase):
    def test_diagnostics_run_on_synthetic_workspace(self):
        from huim import pipeline as P

        articles, beh, events, pairs, pre, art, ctx = synthetic_world(seed=2)
        ws = P.Workspace(None, {}, {"validation": beh}, articles, {}, {}, {}, {}, {}, pre, art, {"validation": ctx})
        for mode in ("softmax", "uniform"):
            model = make_model(pre, art, attention_mode=mode)
            out = P.attention_examples(model, ws, torch.device("cpu"), n_requests=2)
            self.assertTrue(out["examples"])
            if mode == "softmax":
                self.assertTrue(0 < out["mean_entropy_ratio_vs_uniform"] <= 1.0 + 1e-6)
            m = P.vector_masked_metrics(model, ws, torch.device("cpu"))
            self.assertIsNotNone(m["mean_impression_auc"])


class TestPersistence(unittest.TestCase):
    """Serialisation only: no optimiser step is taken."""

    def test_recovery_and_bundle_roundtrip(self):
        from huim.persist import load_model, save_model_files
        from huim.train import Trainer

        articles, beh, events, pairs, pre, art, ctx = synthetic_world(seed=3)
        model = make_model(pre, art)
        cfg = {"lr": 1e-3, "weight_decay": 1e-4, "seed": 1, "batch_size": 8, "max_epochs": 1,
               "early_stopping_patience": 0, "amp": False}
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as persist:
            tr = Trainer(model, art, ctx, ctx, torch.device("cpu"), cfg, Path(tmp), Path(persist), log=lambda *_: None)
            tr.state["step_in_epoch"] = 7
            tr.save_selected()  # selected weights travel inside the recovery payload
            cpu_state = torch.get_rng_state().clone()
            tr.save_recovery()
            self.assertTrue((Path(persist) / "recovery/last.pt").is_file())
            torch.rand(10)  # advance RNG
            tr2 = Trainer(make_model(pre, art), art, ctx, ctx, torch.device("cpu"), cfg, Path(tmp), None,
                          log=lambda *_: None)
            tr2.load_recovery(Path(tmp) / "recovery/last.pt")
            self.assertEqual(tr2.state["step_in_epoch"], 7)
            self.assertTrue(torch.equal(torch.get_rng_state(), cpu_state))
            tr.save_selected()
            (Path(tmp) / "model/weights.pt").unlink()
            tr2.load_recovery(Path(tmp) / "recovery/last.pt")
            self.assertTrue((Path(tmp) / "model/weights.pt").is_file())  # restored from recovery
            hashes = save_model_files(Path(tmp), model, model.cfg, pre, art, {"selected_epoch": 0})
            self.assertEqual(len(hashes), 4)
            loaded, cfg2, pre2, meta = load_model(Path(tmp), torch.device("cpu"))
            self.assertEqual(pre2.to_json(), json.loads(json.dumps(pre.to_json())))
            rp = torch.from_numpy(ctx.recorded["req_pos"])
            ai = torch.from_numpy(ctx.recorded["art_idx"])
            with torch.no_grad():
                a = model(ctx.tensors, art.tensors, len(art.ids), rp, ai)
                b = loaded(ctx.tensors, art.tensors, len(art.ids), rp, ai)
            self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_completion_marker_requires_files(self):
        from huim.io_utils import marker_is_valid, write_completion_marker

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.json").write_text("{}")
            with self.assertRaises(FileNotFoundError):
                write_completion_marker(Path(tmp), "_COMPLETE.json", ["a.json", "missing.json"])
            self.assertFalse(marker_is_valid(Path(tmp), "_COMPLETE.json"))
            write_completion_marker(Path(tmp), "_COMPLETE.json", ["a.json"])
            self.assertTrue(marker_is_valid(Path(tmp), "_COMPLETE.json"))
            (Path(tmp) / "a.json").write_text("{\"changed\": 1}")
            self.assertFalse(marker_is_valid(Path(tmp), "_COMPLETE.json"))


class TestPipelineSynthetic(unittest.TestCase):
    """Export -> validate -> persist -> reload on a synthetic 3-split workspace (random weights, no training)."""

    def test_end_to_end_without_training(self):
        from huim import pipeline as P
        from huim.persist import save_model_files
        from huim.io_utils import atomic_torch_save, atomic_write_json

        articles, beh, events, pairs, pre, art, ctx_v = synthetic_world(seed=4)
        ctx = {}
        for s in ("train", "validation", "test"):
            c = F.build_split_context(s, beh, events, np.unique(events.user_id), art, pre, art.ids[1:], pairs)
            if s == "test":
                c.requests = c.requests.assign(article_ids_clicked=None)
                c.recorded["label"] = np.full(len(c.recorded["label"]), np.nan)
            ctx[s] = c
        ws = P.Workspace(None, {}, {s: beh for s in ctx}, articles, {}, {}, {}, {}, {}, pre, art, ctx)
        model = make_model(pre, art)
        cfg = P.make_config("smoke", pair_chunk=50, request_block=4)
        dev = torch.device("cpu")
        with tempfile.TemporaryDirectory() as run, tempfile.TemporaryDirectory() as persist:
            run, persist = Path(run), Path(persist)
            recorded, logits = P.export_recorded(model, ws, run, dev)
            self.assertTrue(all(r["validation"]["passed"] for r in recorded.values()))
            sets = {s: np.arange(len(ctx[s].requests)) for s in ctx}
            dev_root = run / "dev_sample"
            res = P.export_catalogue(model, ws, cfg, dev_root, dev, sets, {s: True for s in ctx}, logits,
                                     log=lambda *_: None)
            for s, r in res.items():
                self.assertTrue(r["validation"]["passed"], (s, r["validation"]["errors"]))
            self.assertIn("recall", res["validation"]["summary"])
            self.assertNotIn("recall", res["test"]["summary"])
            short = P.export_catalogue(model, ws, cfg, run, dev, {"validation": sets["validation"]},
                                       {"validation": False}, logits, log=lambda *_: None)
            chk = short["validation"]["validation"]["checks"]["recomputed_shortlist_check"]
            self.assertTrue(chk["passed"] and chk["requests_checked"] > 0, chk)
            self.assertFalse((run / "predictions/validation/submission_click_probabilities.csv").exists())
            atomic_write_json(dev_root / "dev_sample_requests.json", {s: v.tolist() for s, v in sets.items()})
            atomic_torch_save({k: v.cpu() for k, v in model.state_dict().items()}, run / "model" / "weights.pt")
            save_model_files(run, model, model.cfg, pre, art, {"selected_epoch": 0})
            for name in ("run_config.json", "metrics.json", "export_manifest.json"):
                atomic_write_json(run / name, {"synthetic": True})
            (run / "requirements.txt").write_text("torch\n")
            gate = P.full_score_gate({"eligibility": {s: {"feasibility": {"eligible_pairs": 10, "bytes_per_row_assumed": 50}}
                                                      for s in ctx}}, dict(cfg, shortlist_splits=["validation"],
                                                                           full_score_splits=["validation"]), [run], None)
            self.assertTrue(gate["validation"]["write"])
            self.assertFalse(gate["train"]["write"])
            P.persist_run(run, persist, log=lambda *_: None)
            self.assertFalse(P.run_is_complete(persist))  # no marker before the reload check
            reload = P.reload_check(persist, ws, cfg, dev, n_requests=2)
            self.assertTrue(reload["passed"], reload)
            atomic_write_json(run / "reload_check.json", reload)
            with self.assertRaises(RuntimeError):
                P.mark_run_complete(run, persist, {"passed": False}, log=lambda *_: None)
            P.mark_run_complete(run, persist, reload, log=lambda *_: None)
            self.assertTrue(P.run_is_complete(persist))


if __name__ == "__main__":
    unittest.main()
