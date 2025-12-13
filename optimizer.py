import os
import json
import random
import itertools
import time

import numpy as np
import joblib
from flask import Flask, request, jsonify

# ============================================================
#  COSTANTI GEOMETRICHE
# ============================================================

L = 4.0
H = 6.0

MARGIN = 0.30
PIER_MIN = 0.30
CORDOLI_Y = [
    (2.7, 3.0),
    (5.7, 6.0),
]

WIDTH_MIN = 0.80

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "wall_model_v2.pkl")

# ============================================================
#  FUNZIONI GEOMETRICHE
# ============================================================

def openings_valid(openings, cordoli_y, margin):
    for (x1, x2, y1, y2) in openings:
        if x1 < PIER_MIN or x2 > (L - PIER_MIN):
            return False
        if not (margin <= x1 < x2 <= L - margin):
            return False
        if not (margin <= y1 < y2 <= H - margin):
            return False
        for (yc1, yc2) in cordoli_y:
            if not (y2 <= yc1 - margin or y1 >= yc2 + margin):
                return False

    n = len(openings)
    for i in range(n):
        x1i, x2i, y1i, y2i = openings[i]
        for j in range(i + 1, n):
            x1j, x2j, y1j, y2j = openings[j]
            dx_gap = max(0.0, max(x1i, x1j) - min(x2i, x2j))
            dy_gap = max(0.0, max(y1i, y1j) - min(y2i, y2j))
            overlap_y = not (y2i <= y1j or y2j <= y1i)
            if overlap_y and dx_gap < PIER_MIN:
                return False
            if dx_gap < margin and dy_gap < margin:
                return False
    return True


def openings_to_features(openings):
    data = []
    areas_bottom = 0.0
    areas_top = 0.0
    area_openings = 0.0

    for (x1, x2, y1, y2) in openings:
        xc = 0.5 * (x1 + x2)
        yc = 0.5 * (y1 + y2)
        w = x2 - x1
        h = y2 - y1
        A = w * h

        area_openings += A
        if yc < H / 2:
            areas_bottom += A
        else:
            areas_top += A

        data.append((yc / H, xc / L, xc / L, yc / H, w / L, h / H, A / (L * H)))

    data.sort(key=lambda t: (t[0], t[1]))

    feats = []
    for (_, _, xc, yc, w, h, A) in data:
        feats.extend([xc, yc, w, h, A])

    wall_area = L * H
    void_ratio = area_openings / wall_area
    solid_ratio = 1.0 - void_ratio
    ratio_bottom = areas_bottom / area_openings if area_openings > 0 else 0.0

    feats.extend([
        areas_bottom / wall_area,
        areas_top / wall_area,
        void_ratio,
        solid_ratio,
        ratio_bottom,
    ])

    return np.array(feats)

# ============================================================
#  PERTURBAZIONI
# ============================================================

def perturb_openings(openings, max_shift_m, max_shrink_ratio):
    new_openings = []
    for (x1, x2, y1, y2) in openings:
        w = x2 - x1
        xc = 0.5 * (x1 + x2)

        shrink = random.uniform(0.0, min(max_shrink_ratio * w, max(0.0, w - WIDTH_MIN)))
        w_new = w - shrink
        xc_new = xc + random.uniform(-max_shift_m, max_shift_m)

        x1n = xc_new - w_new / 2
        x2n = xc_new + w_new / 2

        shift = 0.0
        if x1n < MARGIN:
            shift = MARGIN - x1n
        elif x2n > L - MARGIN:
            shift = (L - MARGIN) - x2n

        new_openings.append((x1n + shift, x2n + shift, y1, y2))

    return new_openings if openings_valid(new_openings, CORDOLI_Y, MARGIN) else None

# ============================================================
#  OTTIMIZZAZIONE (FIRST-HIT)
# ============================================================

def optimize_project_layout(
    openings_project,
    model,
    max_shift_m,
    max_shrink_ratio,
    n_shift_candidates,
    n_level_candidates,
    improvement_target_ratio,
    time_limit_sec,
    max_evals=20000
):
    t0 = time.time()

    base_feats = openings_to_features(openings_project).reshape(1, -1)
    base_pred = float(model.predict(base_feats)[0])
    target_pred = base_pred * (1.0 + improvement_target_ratio)

    evals = 0

    # ---- FASE 1: shift casuale ----
    for _ in range(n_shift_candidates):
        if time.time() - t0 > time_limit_sec:
            break
        cand = perturb_openings(openings_project, max_shift_m, 0.0)
        if cand is None:
            continue
        pred = float(model.predict(openings_to_features(cand).reshape(1, -1))[0])
        evals += 1
        if pred >= target_pred:
            return cand, pred, True, evals, target_pred, base_pred

    # ---- FASE 2: shift + shrink ----
    for _ in range(n_level_candidates):
        if time.time() - t0 > time_limit_sec:
            break
        cand = perturb_openings(openings_project, max_shift_m, max_shrink_ratio)
        if cand is None:
            continue
        pred = float(model.predict(openings_to_features(cand).reshape(1, -1))[0])
        evals += 1
        if pred >= target_pred:
            return cand, pred, True, evals, target_pred, base_pred

    return openings_project, base_pred, False, evals, target_pred, base_pred

# ============================================================
#  FLASK APP
# ============================================================

app = Flask(__name__)
model = joblib.load(MODEL_PATH)

@app.route("/optimize", methods=["POST"])
def optimize_endpoint():
    try:
        payload = request.get_json(force=True)

        if isinstance(payload, list):
            payload = payload[0]

        data = payload.get("body", payload)

        openings_json = data["openings"]
        openings = [(o["x1"], o["x2"], o["y1"], o["y2"]) for o in openings_json]

        target_improvement_percent = float(data.get("target_improvement_percent", 15.0))
        improvement_ratio = target_improvement_percent / 100.0

        n_shift_candidates = int(request.args.get("n_shift_candidates", data.get("n_shift_candidates", 300)))
        n_level_candidates = int(request.args.get("n_level_candidates", data.get("n_level_candidates", 200)))
        time_limit_sec = float(request.args.get("time_limit_sec", data.get("time_limit_sec", 240)))

        random.seed(1)
        np.random.seed(1)

        best_openings, best_pred, *_ = optimize_project_layout(
            openings,
            model,
            max_shift_m=0.15,
            max_shrink_ratio=0.15,
            n_shift_candidates=n_shift_candidates,
            n_level_candidates=n_level_candidates,
            improvement_target_ratio=improvement_ratio,
            time_limit_sec=time_limit_sec
        )

        return jsonify([
            {
                "V_achieved_kN": float(best_pred),
                "optimized_openings": [
                    {"x1": x1, "x2": x2, "y1": y1, "y2": y2}
                    for (x1, x2, y1, y2) in best_openings
                ]
            }
        ])

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
