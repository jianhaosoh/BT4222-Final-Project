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

    def test_recall_and_ndcg_summary(self):
        from huim.metrics import request_retrieval

        rows = []
        # (ranked shortlist, clicked IDs, eligible count, n eligible clicks)
        for ranked, clicked, n_el, n_cel in [([5, 7, 9], [7, 11], 3, 1),   # 11 ineligible: stays in denominator
                                             ([], [3], 0, 0),               # empty pool with a known click -> 0
                                             ([1, 2], [], 2, 0)]:           # no known clicks: reported separately
            n, hits, dcg, ideal = request_retrieval(ranked, clicked)
            rows.append({"n_clicked": n, "n_clicked_eligible": n_cel, "n_clicked_in_shortlist": hits,
                         "n_clicked_not_in_catalogue": n - n_cel, "n_clicked_missing_pub": 0, "n_clicked_future": 0,
                         "observed_dcg_at_100": dcg, "ideal_dcg_at_100": ideal, "eligible_count": n_el,
                         "tie_crosses_cutoff": 0})
        r = recall_summary(pd.DataFrame(rows))
        self.assertEqual(r["requests_with_known_clicks"], 2)
        self.assertEqual(r["requests_without_known_clicks"], 1)
        self.assertEqual(r["empty_pool_requests_with_known_clicks"], 1)
        self.assertAlmostEqual(r["known_click_recall@100"], (0.5 + 0) / 2)
        ideal_two = 1 + 1 / np.log2(3)  # both original clicks in the ideal, although 11 was ineligible
        self.assertAlmostEqual(r["observed_click_ndcg@100"], ((1 / np.log2(3)) / ideal_two + 0) / 2)
        self.assertAlmostEqual(r["eligible_only_recall@100"], 1.0)

    def test_request_retrieval_rules(self):
        from huim.metrics import request_retrieval

        n, hits, dcg, ideal = request_retrieval([4, 8], [8, 8, 4])  # duplicate click counts once
        self.assertEqual((n, hits), (2, 2))
        self.assertAlmostEqual(dcg / ideal, 1.0)
        # clicks beyond the cutoff are missed but remain in the (capped) ideal
        ranked = list(range(1000, 1150))
        n, hits, dcg, ideal = request_retrieval(ranked, [1000, 1120], k=100)
        self.assertEqual(hits, 1)
        self.assertAlmostEqual(ideal, 1 + 1 / np.log2(3))
        self.assertAlmostEqual(dcg, 1.0)
        # ideal capped at k clicks
        n, hits, dcg, ideal = request_retrieval([], list(range(300)), k=100)
        self.assertAlmostEqual(ideal, sum(1 / np.log2(r + 1) for r in range(1, 101)))

    def test_selection_order(self):
        from huim.metrics import better_selection

        a = {"known_click_recall@100": 0.4, "observed_click_ndcg@100": 0.2}
        self.assertTrue(better_selection(a, None))
        self.assertTrue(better_selection({**a, "known_click_recall@100": 0.41}, a))
        self.assertTrue(better_selection({**a, "observed_click_ndcg@100": 0.21}, a))  # tie -> nDCG
        self.assertFalse(better_selection(dict(a), a))                               # exact tie -> earlier wins
        self.assertFalse(better_selection({"known_click_recall@100": 0.39, "observed_click_ndcg@100": 0.9}, a))


class TestRanking(unittest.TestCase):
    def test_stable_tie_break_ascending_id(self):
        logits = torch.tensor([[1.0, 3.0, 3.0, 2.0, 3.0]])
        elig = torch.tensor([[True, True, False, True, True]])
        order, lengths, ties = C.CatalogueScorer.rank(logits, elig, k=3)
        self.assertEqual(order[0, :3].tolist(), [1, 4, 3])  # ties 1 and 4 by position; 2 ineligible
        self.assertEqual(int(lengths[0]), 3)
        self.assertFalse(bool(ties[0]))
        # tie crossing the cutoff: ranks 1-2 share a score with rank 3 (excluded by k=2)
        o2, l2, t2 = C.CatalogueScorer.rank(torch.tensor([[2.0, 2.0, 2.0, 1.0]]), torch.ones(1, 4, dtype=torch.bool), k=2)
        self.assertEqual(o2[0].tolist(), [0, 1])
        self.assertTrue(bool(t2[0]))

    def test_empty_pool(self):
        order, lengths, ties = C.CatalogueScorer.rank(torch.zeros(1, 4), torch.zeros(1, 4, dtype=torch.bool))
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
            self.assertIsNotNone(m["diagnostic_recorded_slate_auc"])


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


class TestVectorReaders(unittest.TestCase):
    """Regression: the DuckDB fallback returned a RecordBatchReader (DuckDB >= 1.4),
    which downstream decoding indexed as a Table. Every backend must yield a
    Table with identical values, dimensions and null masks."""

    @classmethod
    def setUpClass(cls):
        import pyarrow as pa
        import pyarrow.parquet as pq

        cls.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(0)
        cls.vectors = [rng.normal(size=1024).astype(np.float32).tolist() if i % 3 else None for i in range(12)]
        table = pa.table({"article_id": pa.array(np.arange(100, 112), pa.int32()),
                          # nullable fixed-size list, as in the original full-data image files
                          "image_embedding": pa.array(cls.vectors, type=pa.list_(pa.float32(), 1024)),
                          "has_image_embedding": [v is not None for v in cls.vectors],
                          "status": ["ok" if v is not None else "missing_input" for v in cls.vectors]})
        cls.path = Path(cls.tmp.name) / "image_embeddings.parquet"
        pq.write_table(table, cls.path)
        from huim import data as D

        cls.D = D
        ref_table, _ = D.read_vector_table(cls.path, "image_embedding", "has_image_embedding", readers=("pyarrow",))
        cls.ref = D._decode_vector_column(ref_table, "image_embedding", "has_image_embedding", 1024)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def assert_same_as_reference(self, table):
        ids, matrix, diag = self.D._decode_vector_column(table, "image_embedding", "has_image_embedding", 1024)
        np.testing.assert_array_equal(ids, self.ref[0])
        np.testing.assert_array_equal(matrix, self.ref[1])  # exact float32 values
        self.assertEqual(diag["null_vectors"], sum(v is None for v in self.vectors))
        self.assertEqual(diag["usable"], sum(v is not None for v in self.vectors))
        self.assertEqual(diag["dimension"], 1024)

    def failing_pyarrow(self):
        import pyarrow as pa
        from unittest import mock

        def fail(*args, **kwargs):
            raise pa.ArrowNotImplementedError("simulated nullable fixed-size-list decode failure")
        return mock.patch.object(self.D.pq, "read_table", fail)

    def test_coercion_of_reader_types(self):
        import pyarrow as pa

        table = pa.table({"a": [1, 2]})
        self.assertIs(self.D.as_arrow_table(table), table)
        reader = pa.RecordBatchReader.from_batches(table.schema, table.to_batches())
        self.assertTrue(self.D.as_arrow_table(reader).equals(table))
        self.assertTrue(self.D.as_arrow_table(table.to_batches()[0]).equals(table))
        with self.assertRaises(TypeError):
            self.D.as_arrow_table(object())

    def test_stubbed_duckdb_returning_record_batch_reader(self):
        """Independent of the installed DuckDB: a backend that returns a reader."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        from unittest import mock

        real_read = pq.read_table  # captured before the PyArrow reader is made to fail

        def reader_backend(path, columns, column):
            t = real_read(path, columns=columns)
            raw = pa.RecordBatchReader.from_batches(t.schema, t.to_batches())
            return self.D.as_arrow_table(raw), type(raw).__name__

        with mock.patch.object(self.D, "_duckdb_table", reader_backend), self.failing_pyarrow():
            table, info = self.D.read_vector_table(self.path, "image_embedding", "has_image_embedding")
        self.assertEqual(info["reader"], "duckdb")
        self.assertEqual(info["raw_return_type"], "RecordBatchReader")
        self.assertIsInstance(table, pa.Table)
        self.assert_same_as_reference(table)

    @unittest.skipUnless(__import__("importlib").util.find_spec("duckdb"), "duckdb not installed")
    def test_real_duckdb_fallback(self):
        import duckdb
        import pyarrow as pa

        with self.failing_pyarrow():
            table, info = self.D.read_vector_table(self.path, "image_embedding", "has_image_embedding",
                                                   readers=("pyarrow", "duckdb"))
        self.assertEqual(info["reader"], "duckdb", info)
        self.assertIsInstance(table, pa.Table)
        self.assertIn("pyarrow", info["earlier_reader_errors"])
        if tuple(int(x) for x in duckdb.__version__.split(".")[:2]) >= (1, 4):
            self.assertEqual(info["raw_return_type"], "RecordBatchReader")  # the reported failure mode
        self.assert_same_as_reference(table)

    @unittest.skipUnless(__import__("importlib").util.find_spec("polars"), "polars not installed")
    def test_real_polars_fallback(self):
        with self.failing_pyarrow():
            table, info = self.D.read_vector_table(self.path, "image_embedding", "has_image_embedding",
                                                   readers=("pyarrow", "polars"))
        self.assertEqual(info["reader"], "polars")
        self.assert_same_as_reference(table)

    def test_all_readers_failing_raises(self):
        from unittest import mock

        def fail(*a, **k):
            raise RuntimeError("unreadable")

        with self.failing_pyarrow(), mock.patch.object(self.D, "_duckdb_table", fail), \
                mock.patch.object(self.D, "_polars_table", fail):
            with self.assertRaises(RuntimeError):
                self.D.read_vector_table(self.path, "image_embedding", "has_image_embedding")


def selection_cfg(**kw):
    cfg = {"lr": 1e-3, "weight_decay": 1e-4, "seed": 1, "batch_size": 8, "max_epochs": 2, "amp": False,
           "early_stopping_patience": 1, "min_improvement": 0.0, "selection_interval_steps": 3,
           "selection_requests": 6, "selection_seed": 5, "pair_chunk": 64, "request_block": 4, "inference_amp": False,
           "catalogue_policy": {"validation": ["train_val"]}, "checkpoint_every_steps": 0}
    cfg.update(kw)
    return cfg


class ScriptedSelector:
    """Returns scripted metrics: exercises selection bookkeeping without training."""

    def __init__(self, script, protocol=None):
        self.script, self.calls = list(script), 0
        self.protocol = protocol or {"version": "catalogue_recall@100_v1", "sample": "scripted"}

    def evaluate(self, model, art_dev=None):
        r, n = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return {"known_click_recall@100": r, "observed_click_ndcg@100": n, "eligible_only_recall@100": r,
                "requests_with_ties_crossing_cutoff": 0, "seconds": 0.0, "requests_evaluated": 6}


class TestSelection(unittest.TestCase):
    """Selection logic, protocol guard and catalogue-metric consistency (no optimiser steps)."""

    @classmethod
    def setUpClass(cls):
        cls.articles, cls.beh, cls.events, cls.pairs, cls.pre, cls.art, cls.ctx = synthetic_world(seed=7)

    def trainer(self, tmp, selector, **kw):
        from huim.train import Trainer

        return Trainer(make_model(self.pre, self.art), self.art, self.ctx, self.ctx, torch.device("cpu"),
                       selection_cfg(**kw), Path(tmp), None, log=lambda *_: None, selector=selector)

    def test_best_checkpoint_tie_rule_and_patience(self):
        script = [(0.10, 0.05), (0.20, 0.10), (0.20, 0.12), (0.20, 0.11), (0.15, 0.30), (0.05, 0.01)]
        with tempfile.TemporaryDirectory() as tmp:
            tr = self.trainer(tmp, ScriptedSelector(script), early_stopping_patience=1)
            selected = []
            for _ in script:
                if tr.state["finished"]:
                    break
                rec = tr.evaluate()
                selected.append(rec["selected_so_far"])
            self.assertEqual(selected, [True, True, True, False, False])  # stops after 2 evals w/o improvement
            self.assertTrue(tr.state["finished"])
            self.assertEqual(tr.state["best"]["eval_index"], 2)            # equal recall, higher nDCG
            self.assertTrue((Path(tmp) / "model/weights.pt").is_file())
            hist = json.loads((Path(tmp) / "training_history.json").read_text())
            self.assertIn("diagnostic_recorded_slate_auc", hist[0])

    def test_min_improvement_counts_small_gains_as_no_improvement(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr = self.trainer(tmp, ScriptedSelector([(0.20, 0.1), (0.2005, 0.1), (0.2009, 0.1)]),
                              min_improvement=0.001, early_stopping_patience=5)
            for _ in range(3):
                tr.evaluate()
            self.assertEqual(tr.state["best"]["eval_index"], 2)  # still selects the best checkpoint
            self.assertEqual(tr.state["evals_without_improvement"], 2)

    def test_fit_schedule_without_training(self):
        """Stub train_steps: checks interval/epoch-end evaluations, resume of a pending evaluation."""
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            tr = self.trainer(tmp, ScriptedSelector([(0.1 * i, 0.0) for i in range(1, 50)]),
                              early_stopping_patience=99, max_epochs=2, selection_interval_steps=3)
            n = tr.steps_per_epoch()

            def fake_steps(epoch, start, stop=None):
                stop = n if stop is None else min(stop, n)
                tr.state["global_step"] += stop - start
                tr.state["step_in_epoch"] = stop
                return 0.0, n
            with mock.patch.object(tr, "train_steps", side_effect=fake_steps):
                state = tr.fit()
            steps = [h["global_step"] for h in state["history"]]
            expected = sorted({g for g in range(3, 2 * n + 1, 3)} | {n, 2 * n})
            self.assertEqual(steps, expected)
            self.assertEqual(len(set(steps)), len(steps))  # never evaluated twice at one step
            # a crash exactly at a boundary before evaluating: resume evaluates first
            tr2 = self.trainer(tmp, ScriptedSelector([(0.9, 0.9)]), early_stopping_patience=99, max_epochs=1,
                               selection_interval_steps=3)
            tr2.state.update(global_step=3, step_in_epoch=3, last_eval_global_step=0)
            with mock.patch.object(tr2, "train_steps", side_effect=lambda e, a, b=None: (0.0, n)):
                tr2.evaluate = mock.MagicMock(side_effect=lambda: tr2.state.update(
                    last_eval_global_step=tr2.state["global_step"], finished=True))
                tr2.fit()
            tr2.evaluate.assert_called_once()

    def test_fit_refuses_without_selector(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self.trainer(tmp, None).fit()

    def test_recovery_from_another_protocol_is_refused(self):
        from huim.train import ProtocolMismatch

        with tempfile.TemporaryDirectory() as tmp:
            old = self.trainer(tmp, ScriptedSelector([(0.1, 0.1)], protocol={"version": "other"}))
            old.save_recovery()
            new = self.trainer(tmp, ScriptedSelector([(0.1, 0.1)]))
            with self.assertRaises(ProtocolMismatch):
                new.load_recovery(Path(tmp) / "recovery/last.pt")
            # a pre-v1 (AUC-selected) checkpoint has no protocol at all
            payload = torch.load(Path(tmp) / "recovery/last.pt", weights_only=False)
            payload["state"].pop("protocol")
            payload["state"]["best_auc"] = 0.7
            torch.save(payload, Path(tmp) / "recovery/last.pt")
            with self.assertRaises(ProtocolMismatch):
                new.load_recovery(Path(tmp) / "recovery/last.pt")

    def test_selector_matches_export_manifest_metrics(self):
        from huim.selection import CatalogueSelector, selection_positions

        model = make_model(self.pre, self.art)
        cfg = selection_cfg()
        pos = selection_positions(len(self.ctx.requests), 6, cfg["selection_seed"])
        self.assertEqual(pos.tolist(), selection_positions(len(self.ctx.requests), 6, cfg["selection_seed"]).tolist())
        sel = CatalogueSelector(self.ctx, self.art, self.ctx, self.art, pos, torch.device("cpu"), cfg)
        out = sel.evaluate(model)
        self.assertEqual(out["requests_evaluated"], 6)
        scorer = C.CatalogueScorer(model, self.ctx, self.art, torch.device("cpu"), pair_chunk=64)
        with tempfile.TemporaryDirectory() as tmp:
            C.export_split(scorer, Path(tmp), pos, write_full_scores=False,
                           click_info=C.click_eligibility(self.ctx, self.art), recorded_counts=None,
                           request_block=4, log=lambda *_: None)
            man = pd.read_csv(Path(tmp) / "diagnostics/catalogue/validation_requests.csv")
        summ = recall_summary(man)
        self.assertAlmostEqual(out["known_click_recall@100"], summ["known_click_recall@100"], places=12)
        self.assertAlmostEqual(out["observed_click_ndcg@100"], summ["observed_click_ndcg@100"], places=9)
        self.assertEqual(sel.sample_record()["protocol"]["sample"]["requests"], 6)


class TestTuningRule(unittest.TestCase):
    @unittest.skipUnless(__import__("importlib").util.find_spec("optuna"), "optuna not installed")
    def test_trial_ranking_recall_then_ndcg_then_earlier(self):
        import optuna
        from huim.tune import select_best_trial

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        values = [(0.30, 0.10), (0.40, 0.10), (0.40, 0.20), (0.40, 0.20), (0.10, 0.90)]

        def objective(trial):
            r, n = values[trial.number]
            trial.set_user_attr("observed_click_ndcg@100", n)
            return r
        study.optimize(objective, n_trials=len(values))

        def pruned(trial):
            raise optuna.TrialPruned()
        study.optimize(pruned, n_trials=1)
        best = select_best_trial(study)
        self.assertEqual(best["number"], 2)  # recall tie -> higher nDCG -> earlier of the equal pair
        self.assertEqual(len(study.trials), 6)  # pruned trial retained


if __name__ == "__main__":
    unittest.main()
