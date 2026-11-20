"""Tableaux PEARP 0,1° par département : moyenne, médiane, P10 et P90 des 35 membres.

Même format « points + communes » que les autres ensembles (AIFS ENS, GEFS, AIGEFS) : un fichier par
département, `forecast` = moyenne, `forecast_statistics` = médiane, p10, p90. Colonnes : température à 2 m (°C),
pluie des 3 dernières heures (mm, absente à H+0), pluie cumulée depuis le run (mm) et rafale maximale des
3 dernières heures (km/h, absente à H+0). Les statistiques sont calculées membre par membre puis sur les 35 membres.
Pour quelques départements (DEPS_MEMBRES), les 35 membres sont aussi publiés tels quels dans membres/XX.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

from source import MEMBERS, Api, decode, grid_index, heures_disponibles, latest_run, steps

VERSION = "1.0.0"
REPO = os.environ.get("GITHUB_REPOSITORY", "meteofrance66/PEARP-01-europe")
COLONNES = ["temperature_c", "precipitation_mm", "precipitation_total_mm", "wind_gust_kmh"]
COLONNES_MEMBRES = ["temperature_c", "precipitation_total_mm", "wind_gust_kmh"]
# Pyrénées-Orientales et départements voisins (Aude, Ariège, Hérault) : la grille couvre toute l'Europe,
# mais le catalogue publié reste limité à la zone suivie. `--departements` pour en ajouter.
DEPS_DEFAUT = ("66", "11", "09", "34")
# Départements dont les 35 membres sont aussi publiés tels quels (membres/XX.json) : voir ecrire_membres.
DEPS_MEMBRES = ("66",)
STATISTIQUES = ("mean", "median", "p10", "p90")
CATALOGUE = Path(__file__).resolve().parent.parent / "config" / "communes-france.json"
MS_EN_KMH = 3.6


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def charger_catalogue(departements, path: Path = CATALOGUE):
    """Points de grille utiles (un par maille contenant au moins une commune) et communes par département."""
    communes = json.loads(path.read_text(encoding="utf-8"))["communes"]
    voulus = set(departements)
    points: dict[int, tuple[float, float]] = {}
    par_dep: dict[str, list[tuple[list, int]]] = defaultdict(list)
    n = 0
    for c in communes:
        if str(c[2]) not in voulus:
            continue
        idx, lat, lon = grid_index(float(c[5]), float(c[6]))
        points[idx] = (lat, lon)
        par_dep[str(c[2])].append((c, idx))
        n += 1
    manquants = voulus - set(par_dep)
    if manquants:
        raise ValueError(f"départements absents du catalogue : {sorted(manquants)}")
    return sorted(points), points, par_dep, n


def statistiques(valeurs: np.ndarray) -> dict[str, np.ndarray]:
    """valeurs : (membres, …) -> moyenne, médiane, p10, p90 sur l'axe des membres."""
    if valeurs.shape[0] != MEMBERS:
        raise ValueError(f"{valeurs.shape[0]} membres au lieu de {MEMBERS}")
    if not np.isfinite(valeurs).all():
        raise ValueError("valeurs non finies")
    return {
        "mean": valeurs.mean(axis=0),
        "median": np.median(valeurs, axis=0),
        "p10": np.quantile(valeurs, 0.1, axis=0),
        "p90": np.quantile(valeurs, 0.9, axis=0),
    }


def rafale_3h(vitesse_horaire: np.ndarray, n_steps: int) -> np.ndarray:
    """Rafale maximale de chaque tranche de 3 h (km/h) : max des 3 vitesses horaires (membres, heures 0..H, points).

    Chaque rafale horaire est le maximum de l'heure écoulée ; la tranche (h-3, h] en couvre trois. NaN à H+0.
    """
    sortie = np.full((vitesse_horaire.shape[0], n_steps, vitesse_horaire.shape[2]), np.nan)
    for k in range(1, n_steps):
        h = 3 * k
        fenetre = vitesse_horaire[:, h - 2 : h + 1]
        if np.isnan(fenetre).all(axis=1).any():
            raise ValueError(f"aucune rafale publiée pour la tranche H+{h - 3} à H+{h}")
        # Au-delà de H+48 l'API ne publie qu'une échéance sur trois : la tranche n'a alors qu'une rafale horaire (la dernière heure).
        sortie[:, k] = np.nanmax(fenetre, axis=1) * MS_EN_KMH
    return sortie


def lignes(stats: dict[str, dict[str, np.ndarray]], stat: str, k: int, ids: list[int]) -> list[list]:
    """Lignes d'une échéance pour les points d'un département, dans l'ordre de COLONNES."""
    out = []
    for i in ids:
        pluie = stats["pluie"][stat][k, i]
        rafale = stats["rafale"][stat][k, i]
        out.append([
            round(float(stats["temperature"][stat][k, i]), 2),
            None if not np.isfinite(pluie) else round(float(pluie), 2),
            round(float(stats["cumul"][stat][k, i]), 2),
            None if not np.isfinite(rafale) else round(float(rafale), 1),
        ])
    return out


def ecrire_membres(run, table, indexes, points, par_dep, temperature, cumul, rafale, codes, sortie: Path) -> None:
    """Valeurs des 35 membres pour quelques départements (membres/XX.json), en plus des statistiques.

    Les quantiles d'une pluie du jour ne se déduisent pas de ceux des cumuls : l'aval calcule donc chaque jour
    membre par membre, puis les quantiles. Mêmes points et communes que departements/XX.json ;
    values[membre][échéance][point] = [température, cumul de pluie, rafale 3 h (None à H+0)].
    """
    position = {idx: i for i, idx in enumerate(indexes)}
    (sortie / "membres").mkdir(parents=True, exist_ok=True)
    for code in codes:
        entrees = par_dep[code]
        idx_dep = sorted({idx for _, idx in entrees})
        local = {idx: j for j, idx in enumerate(idx_dep)}
        ids = [position[idx] for idx in idx_dep]

        def g(m, k, i):
            r = rafale[m, k, i]
            return None if not np.isfinite(r) else round(float(r), 1)

        payload = {
            "schema_version": 1,
            "department": code,
            "members": MEMBERS,
            "run_time": iso(run),
            "times": [iso(run + timedelta(hours=h)) for h in table],
            "columns": COLONNES_MEMBRES,
            "points": [[idx, points[idx][0], points[idx][1], None] for idx in idx_dep],
            "communes": [[c[0], c[1], c[3], c[4], c[5], c[6], local[idx]] for c, idx in entrees],
            "values": [
                [[[round(float(temperature[m, k, i]), 1), round(float(cumul[m, k, i]), 2), g(m, k, i)] for i in ids] for k in range(len(table))]
                for m in range(MEMBERS)
            ],
        }
        (sortie / "membres" / f"{code}.json").write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8"
        )


def ecrire(run, table, indexes, points, par_dep, n_communes, stats, sortie: Path, membres=()) -> dict:
    position = {idx: i for i, idx in enumerate(indexes)}
    (sortie / "departements").mkdir(parents=True, exist_ok=True)
    colonnes = {
        "points": ["model_index", "latitude", "longitude", "altitude_m"],
        "communes": ["code_insee", "name", "postal_codes", "population", "latitude", "longitude", "point_id"],
        "values": COLONNES,
    }
    temps = [iso(run + timedelta(hours=h)) for h in table]
    deps = {}
    for code, entrees in sorted(par_dep.items()):
        idx_dep = sorted({idx for _, idx in entrees})
        local = {idx: j for j, idx in enumerate(idx_dep)}
        ids = [position[idx] for idx in idx_dep]
        prev = {stat: [[temps[k], lignes(stats, stat, k, ids)] for k in range(len(table))] for stat in STATISTIQUES}
        payload = {
            "schema_version": 4,
            "columns": colonnes,
            "department": code,
            "members": MEMBERS,
            "statistic": "mean",
            "run_time": iso(run),
            "points": [[idx, points[idx][0], points[idx][1], None] for idx in idx_dep],
            "communes": [[c[0], c[1], c[3], c[4], c[5], c[6], local[idx]] for c, idx in entrees],
            "forecast": prev["mean"],
            "forecast_statistics": {s: prev[s] for s in STATISTIQUES if s != "mean"},
        }
        (sortie / "departements" / f"{code}.json").write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8"
        )
        deps[code] = {"communes": len(entrees), "points": len(idx_dep)}
    index = {
        "schema_version": 4,
        "status": "ok",
        "version": VERSION,
        "run": iso(run),
        "generated_at": iso(datetime.now(timezone.utc)),
        "model": {
            "name": "PEARP 0,1° Europe (ensemble ARPEGE)",
            "provider": "Météo-France",
            "dataset": "API Modèle ARPEGE Prévision d'Ensemble (PE-ARPEGE), WCS PEARP-01-EUROPE",
            "members": MEMBERS,
            "resolution_degrees": 0.1,
            "forecast_hours_requested": table[-1],
            "run_time": iso(run),
            "source_url": "https://portail-api.meteofrance.fr/web/fr/api/PE-ARPEGE",
            "license": "Licence ouverte Etalab 2.0 (données publiques de Météo-France)",
        },
        "members": MEMBERS,
        "table_steps": table,
        "columns": COLONNES,
        "statistics": list(STATISTIQUES),
        "commune_count": n_communes,
        "departments": deps,
        "member_departments": list(membres),
        "limitations": "Température à 2 m toutes les 3 h, pluie sur 3 h et cumulée depuis le run, rafale maximale de la tranche de 3 h "
        "(composantes U et V de la rafale horaire). Pas de neige ni de vent moyen. Statistiques des 35 membres "
        "(contrôle + 34 perturbés), pas un scénario déterministe. Valeurs au point de grille 0,1° le plus proche de chaque commune, "
        "sans correction d'altitude.",
    }
    (sortie / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    return index


def run_publie() -> str | None:
    """Run actuellement publié sur la branche data (API GitHub : pas le CDN raw, qui peut avoir 5 min de retard)."""
    entetes = {"Accept": "application/vnd.github.raw"}
    if os.environ.get("GITHUB_TOKEN"):
        entetes["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    try:
        r = requests.get(f"https://api.github.com/repos/{REPO}/contents/index.json?ref=data", headers=entetes, timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json().get("run")
    except (requests.RequestException, ValueError) as e:
        print(f"Publication actuelle illisible ({e}) : on reconstruit.", flush=True)
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", help="AAAAMMJJHH (sinon le plus récent complet)")
    p.add_argument("--force", action="store_true", help="reconstruire même si ce run est déjà publié")
    p.add_argument("--departements", default=",".join(DEPS_DEFAUT), help="codes séparés par des virgules")
    p.add_argument("--out-dir", default="build/data")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--pause", type=float, default=0.0, help="secondes d'attente avant chaque appel (quota partagé)")
    a = p.parse_args()

    api = Api(os.environ.get("METEOFRANCE_PEARP_API_KEY", ""), pause=a.pause)
    if a.run:
        run = datetime.strptime(a.run, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        from source import horizon_du_run

        horizon = horizon_du_run(api, run)
        horizon -= horizon % 3
    else:
        run, horizon = latest_run(api)
    table = steps(horizon)
    print(f"Run PEARP 0,1° : {iso(run)}, horizon H+{horizon}, {len(table)} échéances", flush=True)
    if not a.force and run_publie() == iso(run):
        print("Ce run est déjà publié : rien à faire.", flush=True)
        return 0

    deps = [d.strip() for d in a.departements.split(",") if d.strip()]
    indexes, points, par_dep, n_communes = charger_catalogue(deps)
    print(f"{n_communes} communes, {len(par_dep)} départements, {len(indexes)} points de grille", flush=True)
    n_pts, n_steps = len(indexes), len(table)
    temperature = np.full((MEMBERS, n_steps, n_pts), np.nan)
    pluie3h = np.full((MEMBERS, n_steps, n_pts), np.nan)
    vitesse = np.full((MEMBERS, horizon + 1, n_pts), np.nan)  # rafale horaire (m/s), heures 1..H
    comp_u = {}

    heures_rafale = [h for h in heures_disponibles(api, run, "u") if 1 <= h <= horizon]
    manque = [h for h in table if h >= 3 and h not in heures_disponibles(api, run, "p")]
    if manque:
        raise ValueError(f"pluie sur 3 h absente aux échéances {manque}")
    # Appels : T aux échéances du tableau, pluie 3 h aux échéances ≥ 3, rafales U et V à toutes les heures ≥ 1.
    taches = []
    for m in range(MEMBERS):
        for h in table:
            taches.append(("t", m, h))
            if h >= 3:
                taches.append(("p", m, h))
        for h in heures_rafale:
            taches.append(("u", m, h))
            taches.append(("v", m, h))

    def telecharger(champ: str, m: int, h: int):
        octets = api.coverage(run, champ, m, h)
        if octets is None:
            raise RuntimeError(f"{champ} membre {m} H+{h} : absent (403/404)")
        return octets

    fait = 0
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(telecharger, *t): t for t in taches}
        for f in as_completed(futs):
            champ, m, h = futs[f]
            v = decode(f.result(), run, champ, m, h, indexes)
            if champ == "t":
                temperature[m, table.index(h)] = v - 273.15
            elif champ == "p":
                pluie3h[m, table.index(h)] = np.maximum(v, 0.0)
            else:  # rafale : U et V arrivent dans un ordre quelconque ; vitesse dès que les deux sont là
                morceaux = comp_u.setdefault((m, h), {})
                morceaux[champ] = v
                if len(morceaux) == 2:
                    vitesse[m, h] = np.hypot(morceaux["u"], morceaux["v"])
                    del comp_u[(m, h)]
            fait += 1
            if fait % 500 == 0 or fait == len(taches):
                print(f"{fait}/{len(taches)} champs validés", flush=True)
    print(f"Appels terminés : {len(taches)} champs, {api.n_429} réponses « quota atteint » (429) absorbées", flush=True)
    if comp_u:
        raise ValueError(f"{len(comp_u)} rafales avec une seule composante (U ou V)")

    if not (-80 < np.nanmin(temperature) and np.nanmax(temperature) < 60):
        raise ValueError(f"températures hors bornes ({np.nanmin(temperature):.1f} / {np.nanmax(temperature):.1f} °C)")
    pluie3h[:, 0] = 0.0
    cumul = np.cumsum(pluie3h, axis=1)
    pluie = pluie3h.copy()
    pluie[:, 0] = np.nan
    rafale = rafale_3h(vitesse, n_steps)
    stats = {
        "temperature": statistiques(temperature),
        "cumul": statistiques(cumul),
        "pluie": {s: np.full((n_steps, n_pts), np.nan) for s in STATISTIQUES},
        "rafale": {s: np.full((n_steps, n_pts), np.nan) for s in STATISTIQUES},
    }
    for s, v in statistiques(pluie[:, 1:]).items():
        stats["pluie"][s][1:] = v
    for s, v in statistiques(rafale[:, 1:]).items():
        stats["rafale"][s][1:] = v
    sortie = Path(a.out_dir)
    codes_membres = [c for c in DEPS_MEMBRES if c in par_dep]
    ecrire_membres(run, table, indexes, points, par_dep, temperature, cumul, rafale, codes_membres, sortie)
    index = ecrire(run, table, indexes, points, par_dep, n_communes, stats, sortie, codes_membres)
    print(f"Publication prête : run {index['run']}, {index['commune_count']} communes, {len(index['departments'])} départements", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
