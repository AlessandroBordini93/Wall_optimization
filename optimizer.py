import os
import json
import random
import itertools

import numpy as np
import joblib
from flask import Flask, request, jsonify

# ============================================================
#  COSTANTI GEOMETRICHE
# ============================================================

L = 4.0
H = 6.0

MARGIN = 0.30
PIER_MIN = 0.30      # minimo maschio strutturale orizzontale
CORDOLI_Y = [
    (2.7, 3.0),
    (5.7, 6.0),
]

WIDTH_MIN = 0.80  # usato per non stringere troppo le aperture

# Path robusto al modello (funziona in locale e su Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "wall_model_v2.pkl")


# ============================================================
#  FUNZIONI GEOMETRICHE / FEATURES
# ============================================================

def openings_valid(openings, cordoli_y, margin):
    """
    Controlla che:
      - le aperture siano dentro la parete
      - non troppo vicine ai cordoli
      - non si sovrappongano / siano troppo vicine
      - maschi orizzontali >= PIER_MIN
    """
    # 1) limiti della parete + maschi ai bordi
    for (x1, x2, y1, y2) in openings:
        # bordo parete: almeno PIER_MIN di muratura
        if x1 < PIER_MIN or x2 > (L - PIER_MIN):
            return False

        if not (0.0 + margin <= x1 < x2 <= L - margin):
            return False
        if not (0.0 + margin <= y1 < y2 <= H - margin):
            return False

        # 2) distanza da cordoli
        for (yc1, yc2) in cordoli_y:
            # se non sono almeno a "margin" di distanza → invalido
            if not (y2 <= yc1 - margin or y1 >= yc2 + margin):
                return False

    # 3) distanza tra aperture / maschi orizzontali
    n = len(openings)
    for i in range(n):
        x1i, x2i, y1i, y2i = openings[i]
        for j in range(i + 1, n):
            x1j, x2j, y1j, y2j = openings[j]

            dx_gap = max(0.0, max(x1i, x1j) - min(x2i, x2j))
            dy_gap = max(0.0, max(y1i, y1j) - min(y2i, y2j))

            # sovrapposizione in quota?
            overlap_y = not (y2i <= y1j or y2j <= y1i)

            # se sullo stesso piano (overlap_y), voglio maschio orizzontale >= PIER_MIN
            if overlap_y and dx_gap < PIER_MIN:
                return False

            # se "diagonali", tengo comunque separazione generica = margin
            if dx_gap < margin and dy_gap < margin:
                return False

    return True


def openings_to_features(openings):
    """
    Features ML (come nel training):
      - per ogni apertura:
          [xc/L, yc/H, w/L, h/H, A/(L*H)]
        ordinate per quota (yc/H) e poi per xc/L
      - + 5 feature globali:
          * area vuoti piano terra / area parete
          * area vuoti piano primo / area parete
          * void_ratio = area aperture / area parete
          * solid_ratio = 1 - void_ratio
          * ratio_bottom = area vuoti piano terra / area totale vuoti
    """
    data = []
    areas_bottom = 0.0
    areas_top = 0.0
    area_openings = 0.0

    for (x1, x2, y1, y2) in openings:
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Apertura non valida: x2<=x1 o y2<=y1")

        xc = 0.5 * (x1 + x2)
        yc = 0.5 * (y1 + y2)
        w = x2 - x1
        h = y2 - y1
        A = w * h

        area_openings += A

        if yc < H / 2.0:
            areas_bottom += A
        else:
            areas_top += A

        data.append((
            yc / H,     # per ordinamento
            xc / L,
            xc / L,
            yc / H,
            w / L,
            h / H,
            A / (L * H),
        ))

    # ordina: piano basso prima, poi sinistra → destra
    data.sort(key=lambda t: (t[0], t[1]))

    feats = []
    for (_, _, xc_n, yc_n, w_n, h_n, A_n) in data:
        feats.extend([xc_n, yc_n, w_n, h_n, A_n])

    # feature globali
    wall_area = L * H
    void_ratio = area_openings / wall_area
    solid_ratio = 1.0 - void_ratio
    ratio_bottom = areas_bottom / area_openings if area_openings > 0 else 0.0

    feats.extend([
        areas_bottom / wall_area,   # vuoti piano terra / area parete
        areas_top / wall_area,      # vuoti piano primo / area parete
        void_ratio,                 # VUOTO / TOTALE
        solid_ratio,                # PIENO / TOTALE
        ratio_bottom,               # quota di vuoti che sta sotto
    ])

    return np.array(feats)


# ============================================================
#  PERTURBAZIONI (SHIFT + SHRINK)
# ============================================================

def perturb_openings_x_only(openings, max_shift_m=0.15):
    """Sposta solo il baricentro in x, mantenendo w,h e y1,y2."""
    dx_max = max_shift_m
    new_openings = []

    for (x1, x2, y1, y2) in openings:
        w = x2 - x1
        xc = 0.5 * (x1 + x2)

        xc_new = xc + random.uniform(-dx_max, dx_max)

        x1n = xc_new - w / 2.0
        x2n = xc_new + w / 2.0
        y1n = y1
        y2n = y2

        # rispetto margini coi bordi della parete
        shift_x = 0.0
        if x1n < MARGIN:
            shift_x = MARGIN - x1n
        elif x2n > L - MARGIN:
            shift_x = (L - MARGIN) - x2n
        x1n += shift_x
        x2n += shift_x

        new_openings.append((x1n, x2n, y1n, y2n))

    if not openings_valid(new_openings, CORDOLI_Y, MARGIN):
        return None
    return new_openings


def perturb_openings_shift_and_shrink_subset(openings,
                                             subset_indices,
                                             max_shift_m=0.15,
                                             max_shrink_ratio=0.15):
    """
    Perturba SOLO le aperture i in subset_indices con:
      - spostamento in x del baricentro
      - riduzione larghezza (simmetrica) fino a max_shrink_ratio
      - non scende sotto WIDTH_MIN
    """
    dx_max = max_shift_m
    new_openings = []

    for idx, (x1, x2, y1, y2) in enumerate(openings):
        w = x2 - x1
        xc = 0.5 * (x1 + x2)

        if idx in subset_indices:
            max_shrink_allowed = max_shrink_ratio * w
            max_shrink_allowed = min(max_shrink_allowed, max(0.0, w - WIDTH_MIN))
            shrink = random.uniform(0.0, max_shrink_allowed)
            w_new = w - shrink

            xc_new = xc + random.uniform(-dx_max, dx_max)
        else:
            w_new = w
            xc_new = xc

        x1n = xc_new - w_new / 2.0
        x2n = xc_new + w_new / 2.0
        y1n = y1
        y2n = y2

        shift_x = 0.0
        if x1n < MARGIN:
            shift_x = MARGIN - x1n
        elif x2n > L - MARGIN:
            shift_x = (L - MARGIN) - x2n
        x1n += shift_x
        x2n += shift_x

        new_openings.append((x1n, x2n, y1n, y2n))

    if not openings_valid(new_openings, CORDOLI_Y, MARGIN):
        return None
    return new_openings


# ============================================================
#  OTTIMIZZAZIONE (target % rispetto al progetto)
# ============================================================

def optimize_project_layout(openings_project, model,
                            max_shift_m=0.15,
                            max_shrink_ratio=0.15,
                            n_shift_candidates=2000,
                            n_level_candidates=500,
                            improvement_target_ratio=0.15):
    """
    Ottimizza il layout partendo da openings_project cercando di
    migliorare V_target(15mm) del modello ML di una certa percentuale.

    Ritorna:
      best_openings, best_pred, improved (bool)
    """
    base_feats = openings_to_features(openings_project).reshape(1, -1)
    base_pred = float(model.predict(base_feats)[0])

    target_pred = base_pred * (1.0 + improvement_target_ratio)

    best_pred = base_pred
    best_openings = openings_project
    improved = False

    # ============================
    # Fase 1: SOLO SHIFT IN X
    # ============================
    for _ in range(n_shift_candidates):
        cand = perturb_openings_x_only(openings_project, max_shift_m=max_shift_m)
        if cand is None:
            continue

        feats = openings_to_features(cand).reshape(1, -1)
        pred = float(model.predict(feats)[0])

        if pred > best_pred:
            best_pred = pred
            best_openings = cand
            improved = True

        if best_pred >= target_pred:
            return best_openings, best_pred, improved

    # ============================
    # Fase 2: SHIFT + SHRINK SU SOTTOINSIEMI
    # ============================
    idxs = list(range(len(openings_project)))

    for level in range(1, len(idxs) + 1):
        subsets = list(itertools.combinations(idxs, level))
        n_subsets = len(subsets)
        n_per_subset = max(1, n_level_candidates // n_subsets)

        for subset in subsets:
            for _ in range(n_per_subset):
                cand = perturb_openings_shift_and_shrink_subset(
                    openings_project,
                    subset_indices=subset,
                    max_shift_m=max_shift_m,
                    max_shrink_ratio=max_shrink_ratio
                )
                if cand is None:
                    continue

                feats = openings_to_features(cand).reshape(1, -1)
                pred = float(model.predict(feats)[0])

                if pred > best_pred:
                    best_pred = pred
                    best_openings = cand
                    improved = True

                if best_pred >= target_pred:
                    return best_openings, best_pred, improved

    return best_openings, best_pred, improved


# ============================================================
#  FLASK APP (API HTTP per Render / n8n)
# ============================================================

app = Flask(__name__)

# Carico il modello una volta sola all'avvio
model = joblib.load(MODEL_PATH)


@app.route("/optimize", methods=["POST"])
def optimize_endpoint():
    """
    Input JSON atteso:
    {
      "openings": [
        {"x1": ..., "x2": ..., "y1": ..., "y2": ...},
        ...
      ],
      "V_exist": 125.3,                # ignorato
      "target_improvement_percent": 15 # usato
    }

    Output JSON:
    {
      "V_achieved_kN": ...,
      "optimized_openings": [
        {"x1": ..., "x2": ..., "y1": ..., "y2": ...},
        ...
      ]
    }
    """
    try:
        data = request.get_json(force=True)

        openings_json = data.get("openings")
        if not openings_json or not isinstance(openings_json, list):
            return jsonify({"error": "Campo 'openings' mancante o non valido"}), 400

        openings_project = []
        for o in openings_json:
            openings_project.append(
                (float(o["x1"]), float(o["x2"]), float(o["y1"]), float(o["y2"]))
            )

        if not openings_valid(openings_project, CORDOLI_Y, MARGIN):
            return jsonify({"error": "Geometrie 'openings' non valide secondo i vincoli geometrici"}), 400

        target_improvement_percent = float(data.get("target_improvement_percent", 15.0))
        improvement_ratio = target_improvement_percent / 100.0

        # Seed per riproducibilità
        random.seed(1)
        np.random.seed(1)

        best_openings, best_pred, _ = optimize_project_layout(
            openings_project,
            model,
            max_shift_m=0.15,
            max_shrink_ratio=0.15,
            n_shift_candidates=2000,
            n_level_candidates=500,
            improvement_target_ratio=improvement_ratio,
        )

        optimized_openings_json = [
            {"x1": x1, "x2": x2, "y1": y1, "y2": y2}
            for (x1, x2, y1, y2) in best_openings
        ]

        response = {
            "V_achieved_kN": best_pred,
            "optimized_openings": optimized_openings_json
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
