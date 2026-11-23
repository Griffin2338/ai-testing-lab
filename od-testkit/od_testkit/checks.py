"""Acceptance checks.

Each check takes (ctx, cfg) and returns a TestRecord. suite/test_acceptance.py turns
records into pytest results; the functions also work outside pytest.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from od_testkit.context import SuiteContext, stable_seed
from od_testkit.detectors import build_detector
from od_testkit.geometry import iou_matrix
from od_testkit.matching import BACKGROUND, CLASS_CONFUSION, LOCALIZATION, preservation_rate
from od_testkit.metrics import calibration_error, mean_ap
from od_testkit.report import TestRecord
from od_testkit.stats import wilson_interval, zero_failure_upper_bound
from od_testkit.structures import Detections, read_image
from od_testkit.transforms import (
    CORRUPTIONS,
    adjust_brightness,
    corrupt,
    find_free_position,
    hflip,
    pad,
    paste_patch,
    rescale,
    unflip_boxes,
    unpad_boxes,
    unscale_boxes,
)


@dataclass(frozen=True)
class Meta:
    tc_id: str
    title: str
    reference: str


META = {
    "per_class": Meta("TC-01", "Per-class acceptance at the operating point",
                      "COCO protocol (Lin et al. 2014); bootstrap over images (Russakovsky et al. 2015)"),
    "brightness": Meta("TC-02", "Brightness robustness (metamorphic)", "DeepTest (Tian et al., ICSE 2018)"),
    "corruptions": Meta("TC-03", "Common-corruption robustness",
                        "ImageNet-C (Hendrycks & Dietterich 2019); COCO-C (Michaelis et al. 2019)"),
    "hflip": Meta("TC-04", "Horizontal-flip invariance (no labels needed)",
                  "Metamorphic testing; DeepTest (ICSE 2018)"),
    "scale": Meta("TC-05", "Scale and padding invariance", "Metamorphic testing"),
    "insertion": Meta("TC-06", "Object insertion (metamorphic)", "MetaOD (Wang & Su, ASE 2020)"),
    "negatives": Meta("TC-07", "False positives on empty / out-of-distribution images",
                      "VOS (Du et al., ICLR 2022); D³ (Xie et al., NeurIPS 2023)"),
    "small_objects": Meta("TC-08", "Recall by object size", "COCO size buckets; Hoiem et al. (ECCV 2012)"),
    "latency": Meta("TC-10", "Latency and throughput", "MLPerf Inference (Reddi et al., ISCA 2020)"),
    "parity": Meta("TC-11", "Export / quantisation parity (back-to-back)",
                   "Differential testing: DeepXplore (SOSP 2017); ISTQB CT-AI back-to-back testing"),
    "label_audit": Meta("TC-12", "Golden-set label audit queue",
                        "Northcutt et al. (NeurIPS 2021); ObjectLab (Tkachenko et al. 2023)"),
    "calibration": Meta("TC-14", "Confidence calibration at the operating point",
                        "Küppers et al. (CVPR-W 2020); Kuzucu et al. (ECCV 2024)"),
}


def _rec(key: str, status: str, summary: str, rows=None, notes=None) -> TestRecord:
    m = META[key]
    return TestRecord(m.tc_id, m.title, status, summary, rows or [], notes or [], m.reference)


def _pct(x: float) -> str:
    return "–" if x is None or np.isnan(x) else f"{x:.1%}"


def _pts(x: float) -> str:
    return "–" if np.isnan(x) else f"{x * 100:+.1f} pts"


def _target_classes(ctx: SuiteContext) -> set[int] | None:
    return ctx.classes_of_interest


def _coi_label(ctx: SuiteContext) -> str:
    c = ctx.classes_of_interest
    return "all classes" if c is None else ", ".join(ctx.name_of(i) for i in sorted(c))


def _enough(ctx: SuiteContext, n: int, cfg: dict | None = None) -> bool:
    # a test section can override the global min_instances
    return n >= int((cfg or {}).get("min_instances", ctx.min_instances))


# ------------------------------------------------------------------ TC-01


def per_class(ctx: SuiteContext, cfg: dict) -> TestRecord:
    rmin = float(cfg.get("recall_min", 0.90))
    pmin = float(cfg.get("precision_min", 0.85))
    ci_min = cfg.get("recall_ci_lower_min")
    ev = ctx.clean_eval_with_ap
    classes = ctx.classes_of_interest or {c for c, cs in ev.per_class.items() if cs.instances > 0}
    rows, failed, passed, insufficient = [], [], [], []
    for c in sorted(classes):
        cs = ev.per_class[c]
        reasons = []
        if not _enough(ctx, cs.instances, cfg):
            verdict = "insufficient"
            insufficient.append(cs.name)
        else:
            if cs.recall < rmin:
                reasons.append(f"recall < {rmin:.2f}")
            if np.isnan(cs.precision) or cs.precision < pmin:
                reasons.append(f"precision < {pmin:.2f}")
            if ci_min is not None and cs.recall_ci[0] < float(ci_min):
                reasons.append(f"recall CI low < {float(ci_min):.2f}")
            verdict = "fail" if reasons else "pass"
            (failed if reasons else passed).append(cs.name)
        rows.append({
            "class": cs.name, "instances": cs.instances, "TP": cs.tp, "FP": cs.fp, "FN": cs.fn,
            "precision": cs.precision, "recall": cs.recall, "recall 95% CI": cs.recall_ci,
            "F1": cs.f1, "AP50": cs.ap50, "AP50-95": cs.ap50_95,
            "verdict": verdict + (f" ({'; '.join(reasons)})" if reasons else ""),
        })
    notes = [
        f"Operating point: conf ≥ {ctx.conf}, IoU ≥ {ctx.iou}. The gate uses precision and recall at this "
        "point. AP uses all predictions and is shown for reference.",
        f"Recall CI: percentile bootstrap over images ({ctx.n_boot} resamples).",
    ]
    if insufficient:
        notes.append(f"Fewer than {ctx.min_instances} instances (not gated): {', '.join(insufficient)}.")
    if failed:
        return _rec("per_class", "fail", f"{len(failed)} class(es) below target: {', '.join(failed)}.",
                    rows, notes)
    if not passed:
        return _rec("per_class", "insufficient",
                    f"No class has the {ctx.min_instances} instances needed for a verdict.", rows, notes)
    return _rec("per_class", "pass",
                f"{len(passed)} class(es) meet recall ≥ {rmin:.2f} and precision ≥ {pmin:.2f}.", rows, notes)


# ------------------------------------------------------------------ TC-02


def brightness(ctx: SuiteContext, cfg: dict) -> TestRecord:
    factors = [float(f) for f in cfg.get("factors", [0.6, 1.4])]
    max_r = float(cfg.get("max_recall_drop", 0.05))
    max_p = float(cfg.get("max_precision_drop", max_r))
    coi = _target_classes(ctx)
    base = ctx.clean_eval
    tp, _, gt = base.pooled(coi)
    if not _enough(ctx, gt, cfg):
        return _rec("brightness", "insufficient", f"Only {gt} ground-truth objects for {_coi_label(ctx)}.")
    br, bp = base.pooled_recall(coi), base.pooled_precision(coi)
    rows, fails = [{"variant": "clean", "recall": br, "precision": bp, "recall change": "",
                    "precision change": "", "verdict": ""}], []
    for f in factors:
        preds = ctx.predictions(f"brightness_{f}", lambda img, s, f=f: adjust_brightness(img, f))
        ev = ctx.evaluate(preds)
        r, p = ev.pooled_recall(coi), ev.pooled_precision(coi)
        dr, dp = br - r, bp - p
        bad = dr > max_r or (not np.isnan(dp) and dp > max_p)
        if bad:
            fails.append(f"×{f}")
        rows.append({"variant": f"brightness ×{f} ({(f - 1):+.0%})", "recall": r, "precision": p,
                     "recall change": _pts(-dr), "precision change": _pts(-dp),
                     "verdict": "fail" if bad else "pass"})
    status = "fail" if fails else "pass"
    summary = (f"Recall/precision of {_coi_label(ctx)} must not drop by more than {max_r * 100:.0f} pts. "
               + (f"Failed at {', '.join(fails)}." if fails else "All variants within tolerance."))
    return _rec("brightness", status, summary, rows)


# ------------------------------------------------------------------ TC-03


def corruptions(ctx: SuiteContext, cfg: dict) -> TestRecord:
    names = cfg.get("names") or list(CORRUPTIONS)
    sevs = [int(s) for s in cfg.get("severities", [1, 2, 3])]
    min_rel = float(cfg.get("min_relative_recall", 0.75))
    coi = _target_classes(ctx)
    base = ctx.clean_eval
    _, _, gt = base.pooled(coi)
    if not _enough(ctx, gt, cfg):
        return _rec("corruptions", "insufficient", f"Only {gt} ground-truth objects for {_coi_label(ctx)}.")
    br = base.pooled_recall(coi)
    rows, fails, all_rel = [], [], []
    for name in names:
        row = {"corruption": name}
        rels = []
        for s in sevs:
            preds = ctx.predictions(
                f"corrupt_{name}_{s}",
                lambda img, smp, name=name, s=s: corrupt(img, name, s, seed=stable_seed(name, s, smp.image_id)),
            )
            r = ctx.evaluate(preds).pooled_recall(coi)
            row[f"recall sev{s}"] = r
            rels.append(r / br if br else float("nan"))
        rel = float(np.nanmean(rels))
        all_rel.append(rel)
        row["relative recall"] = rel
        row["verdict"] = "fail" if rel < min_rel else "pass"
        if rel < min_rel:
            fails.append(name)
        rows.append(row)
    mean_rel = float(np.nanmean(all_rel))
    summary = (f"Mean relative recall (corrupted / clean, severities {sevs}) = {mean_rel:.2f}; "
               f"each corruption needs ≥ {min_rel:.2f}. "
               + (f"Below target: {', '.join(fails)}." if fails else "All corruptions within target."))
    notes = [f"Clean pooled recall for {_coi_label(ctx)}: {_pct(br)}. Objects stay in place, so the "
             "original labels are reused."]
    return _rec("corruptions", "fail" if fails else "pass", summary, rows, notes)


# ------------------------------------------------------------------ TC-04


def hflip_invariance(ctx: SuiteContext, cfg: dict) -> TestRecord:
    min_pres = float(cfg.get("min_preserved", 0.95))
    iou_thr = float(cfg.get("iou", 0.8))
    max_diff = int(cfg.get("max_count_diff", 1))
    min_share = float(cfg.get("min_count_ok_share", 0.98))
    coi = _target_classes(ctx)
    flipped = ctx.predictions("hflip", lambda img, s: hflip(img), lambda b, s: unflip_boxes(b, s.width))
    hi_conf = float(cfg.get("high_confidence", min(0.95, ctx.conf + 0.3)))
    p_ab = t_ab = p_ba = t_ba = 0
    p_hi = t_hi = 0
    count_ok = 0
    worst = []
    for s, a_raw, b_raw in zip(ctx.samples, ctx.clean, flipped, strict=True):
        a = a_raw.above(ctx.conf).of_classes(coi)
        b = b_raw.above(ctx.conf).of_classes(coi)
        p1, t1 = preservation_rate(a, b, iou_thr)
        p2, t2 = preservation_rate(b, a, iou_thr)
        ph, th = preservation_rate(a.above(hi_conf), b, iou_thr)
        p_hi, t_hi = p_hi + ph, t_hi + th
        p_ab, t_ab, p_ba, t_ba = p_ab + p1, t_ab + t1, p_ba + p2, t_ba + t2
        diff = abs(len(a) - len(b))
        count_ok += diff <= max_diff
        lost = (t1 - p1) + (t2 - p2)
        if lost:
            worst.append((lost, s.image_id, len(a), len(b)))
    if t_ab + t_ba < 30:
        return _rec("hflip", "insufficient", f"Only {t_ab + t_ba} detections above the threshold to compare.")
    r_ab, r_ba = p_ab / max(t_ab, 1), p_ba / max(t_ba, 1)
    share = count_ok / len(ctx.samples)
    ok = min(r_ab, r_ba) >= min_pres and share >= min_share
    rows = [
        {"check": "clean detections found again after flip", "value": r_ab, "target": f"≥ {min_pres:.2f}",
         "n": t_ab, "verdict": "pass" if r_ab >= min_pres else "fail"},
        {"check": "flipped detections present in clean run", "value": r_ba, "target": f"≥ {min_pres:.2f}",
         "n": t_ba, "verdict": "pass" if r_ba >= min_pres else "fail"},
        {"check": f"images with |count difference| ≤ {max_diff}", "value": share, "target": f"≥ {min_share:.2f}",
         "n": len(ctx.samples), "verdict": "pass" if share >= min_share else "fail"},
    ]
    if t_hi:
        rows.append({"check": f"diagnostic: clean detections with score ≥ {hi_conf:.2f} found after flip",
                     "value": p_hi / t_hi, "target": "", "n": t_hi, "verdict": ""})
    worst.sort(reverse=True)
    notes = [f"Match = same class and IoU ≥ {iou_thr} after mapping boxes back."]
    if t_hi and p_hi / t_hi - r_ab > 0.05:
        notes.append(f"{p_hi / t_hi:.1%} of detections with score ≥ {hi_conf:.2f} survive, against {r_ab:.1%} "
                     f"overall. Most unstable boxes sit just above conf {ctx.conf} and will flicker in the app.")
    if worst:
        notes.append("Most unstable images: " + ", ".join(f"{w[1]} ({w[2]}→{w[3]} dets)" for w in worst[:8]))
    summary = (f"{r_ab:.1%} of detections survive the flip, {r_ba:.1%} of flipped detections match the clean "
               f"run (target ≥ {min_pres:.0%}).")
    return _rec("hflip", "pass" if ok else "fail", summary, rows, notes)


# ------------------------------------------------------------------ TC-05


def _scale_factors(s, f: float) -> tuple[float, float]:
    nw, nh = max(1, int(round(s.width * f))), max(1, int(round(s.height * f)))
    return nw / s.width, nh / s.height


def scale_invariance(ctx: SuiteContext, cfg: dict) -> TestRecord:
    factors = [float(f) for f in cfg.get("factors", [0.75, 1.25])]
    pad_frac = cfg.get("pad_fraction", 0.10)
    max_drop = float(cfg.get("max_recall_drop", 0.03))
    min_size = float(cfg.get("min_object_size", 32))
    coi = _target_classes(ctx)
    rows, fails = [], []

    def run(label, preds, shrink):
        def gt_filter(s):
            wh = np.minimum(s.boxes[:, 2] - s.boxes[:, 0], s.boxes[:, 3] - s.boxes[:, 1])
            keep = wh * shrink >= min_size
            if coi is not None:
                keep &= np.isin(s.classes, list(coi))
            return keep

        base = ctx.evaluate(ctx.clean, gt_filter=gt_filter)
        ev = ctx.evaluate(preds, gt_filter=gt_filter)
        _, _, gt = base.pooled(coi)
        br, r = base.pooled_recall(coi), ev.pooled_recall(coi)
        if not _enough(ctx, gt, cfg):
            rows.append({"variant": label, "objects": gt, "clean recall": br, "variant recall": r,
                         "change": _pts(r - br), "verdict": "insufficient"})
            return
        bad = br - r > max_drop
        if bad:
            fails.append(label)
        rows.append({"variant": label, "objects": gt, "clean recall": br, "variant recall": r,
                     "change": _pts(r - br), "verdict": "fail" if bad else "pass"})

    for f in factors:
        preds = ctx.predictions(
            f"scale_{f}",
            lambda img, s, f=f: rescale(img, f)[0],
            lambda b, s, f=f: unscale_boxes(b, *_scale_factors(s, f)),
        )
        run(f"resize ×{f}", preds, min(f, 1.0))
    if pad_frac:
        frac = float(pad_frac)
        preds = ctx.predictions(
            f"pad_{frac}",
            lambda img, s: pad(img, frac)[0],
            lambda b, s: unpad_boxes(b, int(round(frac * max(s.width, s.height)))),
        )
        run(f"pad {frac:.0%} border", preds, 1.0)
    gated = [r for r in rows if r["verdict"] != "insufficient"]
    if not gated:
        return _rec("scale", "insufficient", "Not enough objects above the minimum size in any variant.", rows)
    summary = (f"Recall on objects ≥ {min_size:.0f} px (after scaling) must not drop by more than "
               f"{max_drop * 100:.0f} pts. " + (f"Failed: {', '.join(fails)}." if fails else "All variants pass."))
    return _rec("scale", "fail" if fails else "pass", summary, rows)


# ------------------------------------------------------------------ TC-06


def insertion(ctx: SuiteContext, cfg: dict) -> TestRecord:
    by_name = {v: k for k, v in ctx.class_names.items()}
    cls_name = cfg.get("class")
    if cls_name is None:
        counts = np.bincount(np.concatenate([s.classes for s in ctx.samples] or [np.zeros(0, int)]),
                             minlength=max(ctx.class_names) + 1)
        pool = ctx.classes_of_interest or set(ctx.class_names)
        cls = max(pool, key=lambda c: counts[c] if c < len(counts) else 0)
    else:
        cls = by_name[cls_name]
    min_patch = int(cfg.get("min_patch_size", 32))
    max_images = int(cfg.get("max_images", 200))
    keep_iou = float(cfg.get("preserve_iou", 0.9))
    min_pres = float(cfg.get("min_preserved", 0.97))
    min_det = float(cfg.get("min_detected", 0.90))
    rng = np.random.default_rng(ctx.seed)
    coi = _target_classes(ctx)

    patches = []
    for i, s in enumerate(ctx.samples):
        for b, c in zip(s.boxes, s.classes, strict=True):
            w, h = b[2] - b[0], b[3] - b[1]
            if c == cls and min(w, h) >= min_patch and max(w, h) <= 0.5 * min(s.width, s.height):
                patches.append((i, b.astype(int)))
        if len(patches) >= 200:
            break
    if len(patches) < 2:
        return _rec("insertion", "insufficient",
                    f"Fewer than 2 '{ctx.name_of(cls)}' objects of ≥ {min_patch} px to use as patches.")

    inserted = detected = preserved = total = skipped = 0
    misses = []
    for i, s in enumerate(ctx.samples[:max_images]):
        j = int(rng.integers(len(patches)))
        for _ in range(len(patches)):  # take the patch from a different image
            if patches[j][0] != i:
                break
            j = (j + 1) % len(patches)
        src_i, pb = patches[j]
        if src_i == i:
            skipped += 1
            continue
        patch = ctx.image(src_i)[pb[1]:pb[3], pb[0]:pb[2]]
        img = ctx.image(i)
        clean = ctx.clean[i].above(ctx.conf)
        occupied = np.vstack([s.boxes, clean.boxes]) if len(clean) else s.boxes
        margin = int(cfg.get("margin", round(0.05 * max(s.width, s.height))))
        pos = find_free_position(img.shape, patch.shape, occupied, margin, rng)
        if pos is None:
            skipped += 1
            continue
        new_img, box = paste_patch(img, patch, *pos)
        pred = ctx.detector.predict(new_img).above(ctx.conf)
        inserted += 1
        ious = iou_matrix(box[None], pred.boxes)[0] if len(pred) else np.zeros(0)
        hit = bool(np.any((pred.classes == cls) & (ious >= ctx.iou))) if len(pred) else False
        detected += hit
        if not hit:
            misses.append(s.image_id)
        ref = clean.of_classes(coi)
        if len(ref):
            away = iou_matrix(ref.boxes, box[None])[:, 0] == 0
            p, t = preservation_rate(ref.select(away), pred, keep_iou)
            preserved, total = preserved + p, total + t
    if inserted < 30:
        return _rec("insertion", "insufficient", f"Only {inserted} images had room for an inserted object.")
    det_rate = detected / inserted
    pres_rate = preserved / total if total else float("nan")
    ok_det = det_rate >= min_det
    ok_pres = np.isnan(pres_rate) or pres_rate >= min_pres
    rows = [
        {"check": f"inserted '{ctx.name_of(cls)}' detected", "value": det_rate, "n": inserted,
         "target": f"≥ {min_det:.2f}", "verdict": "pass" if ok_det else "fail"},
        {"check": f"existing detections unchanged (IoU ≥ {keep_iou})", "value": pres_rate, "n": total,
         "target": f"≥ {min_pres:.2f}", "verdict": "pass" if ok_pres else "fail"},
    ]
    notes = [f"Patches: rectangular crops of '{ctx.name_of(cls)}' GT boxes (≥ {min_patch} px), pasted at least "
             f"`margin` px from labelled and detected objects. Skipped {skipped} image(s) with no free space."]
    if misses:
        notes.append("Inserted object missed in: " + ", ".join(misses[:10]))
    summary = f"Inserted object found in {det_rate:.1%} of {inserted} images; {_pct(pres_rate)} of existing " \
              "detections unchanged."
    return _rec("insertion", "pass" if ok_det and ok_pres else "fail", summary, rows, notes)


# ------------------------------------------------------------------ TC-07


def negatives(ctx: SuiteContext, cfg: dict) -> TestRecord:
    max_fppi = float(cfg.get("max_fp_per_image", 0.05))
    max_cls_share = cfg.get("max_class_fp_image_share", 0.10)
    coi = _target_classes(ctx)
    preds: list[Detections] = []
    if cfg.get("dir"):
        samples = ctx.negatives(cfg)
        source = f"folder {cfg['dir']}"
        for s in samples:
            preds.append(ctx.detector.predict(read_image(s.image_path)).above(ctx.conf).of_classes(coi))
    elif cfg.get("from_dataset", True):
        target = coi or set(ctx.class_names)
        idx = [i for i, s in enumerate(ctx.samples) if not np.isin(s.classes, list(target)).any()]
        samples = [ctx.samples[i] for i in idx]
        source = f"dataset images without {_coi_label(ctx)}"
        preds = [ctx.clean[i].above(ctx.conf).of_classes(target) for i in idx]
    else:
        samples = []
        source = "none"
    n = len(samples)
    min_images = int(cfg.get("min_images", 30))
    if n < min_images:
        return _rec("negatives", "insufficient", f"Only {n} negative images from {source} (need {min_images}).")
    total_fp = sum(len(p) for p in preds)
    fppi = total_fp / n
    with_fp = sum(len(p) > 0 for p in preds)
    lo, hi = wilson_interval(with_fp, n)
    rows = [{"metric": "false positives per image", "value": fppi, "target": f"≤ {max_fppi}",
             "verdict": "pass" if fppi <= max_fppi else "fail"},
            {"metric": "images with ≥ 1 false positive", "value": with_fp / n, "95% CI": (lo, hi), "target": "",
             "verdict": ""}]
    fails = [] if fppi <= max_fppi else ["FP per image"]
    classes = sorted({int(c) for p in preds for c in p.classes})
    for c in classes:
        share = sum(int(np.any(p.classes == c)) for p in preds) / n
        bad = max_cls_share is not None and share > float(max_cls_share)
        if bad:
            fails.append(ctx.name_of(c))
        rows.append({"metric": f"images with a '{ctx.name_of(c)}' FP", "value": share,
                     "target": f"≤ {float(max_cls_share):.2f}" if max_cls_share is not None else "",
                     "verdict": "fail" if bad else "pass"})
    notes = [f"Source: {source}; {n} images; {total_fp} detections above conf {ctx.conf}."]
    if with_fp == 0:
        notes.append(f"No false positives in {n} images: 95 % upper bound on the per-image FP rate is "
                     f"{zero_failure_upper_bound(n):.2%} (rule of three).")
    return _rec("negatives", "fail" if fails else "pass",
                f"{fppi:.3f} false positives per image on {n} images without target objects.", rows, notes)


# ------------------------------------------------------------------ TC-08


def small_objects(ctx: SuiteContext, cfg: dict) -> TestRecord:
    min_small = float(cfg.get("min_recall_small", 0.70))
    max_gap = float(cfg.get("max_gap_to_large", 0.25))
    min_n = int(cfg.get("min_instances", 50))
    coi = _target_classes(ctx)
    ev = ctx.clean_eval if coi is None else ctx.evaluate(
        ctx.clean, gt_filter=lambda s: np.isin(s.classes, list(coi)))
    rows, rec = [], {}
    for bucket, (tp, gt) in ev.size_recall.items():
        r = tp / gt if gt else float("nan")
        rec[bucket] = (r, gt)
        rows.append({"size": bucket, "objects": gt, "recall": r,
                     "95% CI (Wilson)": wilson_interval(tp, gt) if gt else (float("nan"), float("nan")),
                     "enough data": "yes" if gt >= min_n else "no"})
    rs, ns = rec["small"]
    rl, nl = rec["large"]
    if ns < min_n:
        return _rec("small_objects", "insufficient", f"Only {ns} small objects (need {min_n}).", rows)
    fails = []
    if rs < min_small:
        fails.append(f"small recall {rs:.2f} < {min_small:.2f}")
    if nl >= min_n and rl - rs > max_gap:
        fails.append(f"gap to large {(rl - rs) * 100:.0f} pts > {max_gap * 100:.0f}")
    notes = ["COCO size buckets on box area in the original image: small < 32², medium < 96², large ≥ 96² px. "
             "Wilson treats boxes as independent, so the interval is too narrow for crowded images."]
    return _rec("small_objects", "fail" if fails else "pass",
                "; ".join(fails) if fails else f"Small-object recall {rs:.1%} meets the target.", rows, notes)


# ------------------------------------------------------------------ TC-10


def latency(ctx: SuiteContext, cfg: dict) -> TestRecord:
    warmup = int(cfg.get("warmup", 10))
    frames = int(cfg.get("frames", 200))
    p95_max, p99_max = cfg.get("p95_ms"), cfg.get("p99_ms")
    min_fps = cfg.get("min_fps")
    n_imgs = min(len(ctx.samples), int(cfg.get("distinct_images", 50)))
    imgs = [ctx.image(i) for i in range(n_imgs)]
    for i in range(warmup):
        ctx.detector.predict(imgs[i % n_imgs])
    times = []
    for i in range(frames):
        t0 = time.perf_counter()
        ctx.detector.predict(imgs[i % n_imgs])
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    p50, p95, p99 = np.percentile(t, [50, 95, 99])
    fps = 1000.0 / t.mean()
    rows, fails = [], []

    def add(metric, value, limit, higher_is_better=False):
        if limit is None:
            rows.append({"metric": metric, "value": value, "target": "", "verdict": ""})
            return
        bad = value < float(limit) if higher_is_better else value > float(limit)
        if bad:
            fails.append(metric)
        rows.append({"metric": metric, "value": value,
                     "target": f"{'≥' if higher_is_better else '≤'} {limit}", "verdict": "fail" if bad else "pass"})

    add("p50 latency (ms)", float(p50), None)
    add("p95 latency (ms)", float(p95), p95_max)
    add("p99 latency (ms)", float(p99), p99_max)
    add("throughput (FPS, batch 1)", float(fps), min_fps, higher_is_better=True)
    notes = [f"{frames} timed calls after {warmup} warm-up calls over {n_imgs} images. Timer wraps "
             "detector.predict() (pre/post-processing included, image decoding excluded). Use target "
             "hardware numbers for release decisions."]
    if p95_max is None and p99_max is None and min_fps is None:
        return _rec("latency", "info", f"p95 = {p95:.1f} ms, p99 = {p99:.1f} ms (no limits configured).",
                    rows, notes)
    return _rec("latency", "fail" if fails else "pass",
                f"p95 = {p95:.1f} ms, p99 = {p99:.1f} ms, {fps:.1f} FPS.", rows, notes)


# ------------------------------------------------------------------ TC-11


def parity(ctx: SuiteContext, cfg: dict) -> TestRecord:
    if "reference" not in cfg:
        raise ValueError("tests.parity.reference must describe the reference (e.g. FP32) detector")
    max_r = float(cfg.get("max_recall_diff", 0.02))
    max_ap = float(cfg.get("max_map_diff", 0.01))
    min_pres = float(cfg.get("min_preserved", 0.98))
    keep_iou = float(cfg.get("preserve_iou", 0.9))
    ref_det = build_detector(cfg["reference"], ctx.base_dir)
    ref_preds = ctx.predictions("parity_reference", detector=ref_det)
    ev_ref = ctx.evaluate(ref_preds, compute_ap=True)
    ev_c = ctx.clean_eval_with_ap
    coi = _target_classes(ctx)
    classes = coi or {c for c, cs in ev_c.per_class.items() if cs.instances > 0}
    rows, fails = [], []
    for c in sorted(classes):
        a, b = ev_ref.per_class[c], ev_c.per_class[c]
        if not _enough(ctx, a.instances, cfg):
            rows.append({"class": a.name, "instances": a.instances, "recall ref": a.recall,
                         "recall shipped": b.recall, "diff": _pts(b.recall - a.recall), "verdict": "insufficient"})
            continue
        bad = abs(b.recall - a.recall) > max_r
        if bad:
            fails.append(a.name)
        rows.append({"class": a.name, "instances": a.instances, "recall ref": a.recall, "recall shipped": b.recall,
                     "diff": _pts(b.recall - a.recall), "AP50-95 ref": a.ap50_95, "AP50-95 shipped": b.ap50_95,
                     "verdict": "fail" if bad else "pass"})
    map_ref, map_c = mean_ap(ev_ref, classes, "ap50_95"), mean_ap(ev_c, classes, "ap50_95")
    p = t = 0
    for a_raw, b_raw in zip(ref_preds, ctx.clean, strict=True):
        pp, tt = preservation_rate(a_raw.above(ctx.conf).of_classes(coi), b_raw.above(ctx.conf), keep_iou)
        p, t = p + pp, t + tt
    pres = p / t if t else float("nan")
    if abs(map_c - map_ref) > max_ap:
        fails.append("mAP50-95")
    if not np.isnan(pres) and pres < min_pres:
        fails.append("detection preservation")
    rows.append({"class": "mAP50-95", "recall ref": map_ref, "recall shipped": map_c,
                 "diff": _pts(map_c - map_ref),
                 "verdict": "fail" if abs(map_c - map_ref) > max_ap else "pass"})
    rows.append({"class": f"reference detections reproduced (IoU ≥ {keep_iou})", "recall ref": "",
                 "recall shipped": pres, "diff": "", "verdict": "pass" if np.isnan(pres) or pres >= min_pres
                 else "fail"})
    ref_name = cfg["reference"].get("weights") or cfg["reference"]["type"]
    summary = (f"Shipped model vs reference ({ref_name}): mAP50-95 {_pts(map_c - map_ref)}, "
               f"{_pct(pres)} of reference detections reproduced."
               + (f" Failed: {', '.join(fails)}." if fails else ""))
    return _rec("parity", "fail" if fails else "pass", summary, rows)


# ------------------------------------------------------------------ TC-12


def label_audit(ctx: SuiteContext, cfg: dict) -> TestRecord:
    min_score = float(cfg.get("min_score", 0.8))
    top = int(cfg.get("top", 25))
    ev = ctx.clean_eval
    suspects = []
    for s, raw, m in zip(ctx.samples, ctx.clean, ev.matches, strict=True):
        pred = raw.above(ctx.conf)
        for k in range(len(pred)):
            sc = float(pred.scores[k])
            if sc < min_score:
                continue
            kind = int(m.kind[k])
            box = [round(float(v), 1) for v in pred.boxes[k]]
            if kind == BACKGROUND:
                reason = "confident detection with no label nearby: missing label?"
            elif kind == CLASS_CONFUSION:
                j = int(np.argmax(m.ious[k]))
                if m.ious[k, j] < 0.7:
                    continue
                reason = f"overlaps a '{ctx.name_of(int(s.classes[j]))}' label: wrong class label?"
            elif kind == LOCALIZATION:
                reason = "same-class label overlaps only partly: loose or shifted box?"
            else:
                continue
            suspects.append({"image": s.image_id, "predicted": ctx.name_of(int(pred.classes[k])),
                             "score": sc, "box": box, "reason": reason})
    suspects.sort(key=lambda r: -r["score"])
    summary = f"{len(suspects)} detection(s) with score ≥ {min_score} disagree with the labels."
    notes = ["Review queue, never fails the run. Expect a mix of model errors and label errors; Northcutt "
             "et al. (2021) measured about 3.4 % label errors across 10 common test sets."]
    return _rec("label_audit", "info", summary, suspects[:top], notes)


# ------------------------------------------------------------------ TC-14


def calibration(ctx: SuiteContext, cfg: dict) -> TestRecord:
    max_ece = float(cfg.get("max_ece", 0.10))
    n_bins = int(cfg.get("bins", 10))
    min_det = int(cfg.get("min_detections", 100))
    max_gap = cfg.get("max_bin_gap")
    min_bin = int(cfg.get("min_bin_size", 50))
    coi = _target_classes(ctx)
    ece, bins = calibration_error(ctx.clean_eval.matches, ctx.clean, ctx.conf, n_bins, coi)
    n = sum(b["n"] for b in bins)
    if n < min_det:
        return _rec("calibration", "insufficient", f"Only {n} detections above the threshold (need {min_det}).")
    fails = [] if ece <= max_ece else [f"ECE {ece:.3f} > {max_ece}"]
    rows = []
    for b in bins:
        gap = b["precision"] - b["mean_score"]
        bad = max_gap is not None and b["n"] >= min_bin and abs(gap) > float(max_gap)
        if bad:
            fails.append(f"bin {b['bin']}")
        rows.append({**b, "gap (precision − score)": gap,
                     "verdict": ("fail" if bad else "pass") if max_gap is not None and b["n"] >= min_bin else ""})
    notes = ["Detection ECE over boxes with score ≥ conf: precision vs. mean score per bin. "
             "Negative gap = over-confident."]
    return _rec("calibration", "fail" if fails else "pass",
                f"ECE = {ece:.3f} over {n} detections (target ≤ {max_ece}).", rows, notes)


CHECKS = {
    "per_class": per_class,
    "brightness": brightness,
    "corruptions": corruptions,
    "hflip": hflip_invariance,
    "scale": scale_invariance,
    "insertion": insertion,
    "negatives": negatives,
    "small_objects": small_objects,
    "latency": latency,
    "parity": parity,
    "label_audit": label_audit,
    "calibration": calibration,
}
