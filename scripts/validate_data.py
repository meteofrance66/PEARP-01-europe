"""Contrôle complet de build/data avant publication ; toute anomalie bloque (la publication précédente reste en ligne)."""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

from build_data import COLONNES, COLONNES_MEMBRES, DEPS_DEFAUT, DEPS_MEMBRES, run_publie
from source import MEMBERS

racine = Path(sys.argv[1] if len(sys.argv) > 1 else "build/data")
index = json.loads((racine / "index.json").read_text(encoding="utf-8"))
assert index["status"] == "ok" and index["members"] == MEMBERS, "index.json incohérent"
STEPS = index["table_steps"]
assert STEPS[0] == 0 and all(b - a == 3 for a, b in zip(STEPS, STEPS[1:])) and len(STEPS) >= 10, f"échéances anormales : {STEPS}"
assert index["columns"] == COLONNES
assert index["commune_count"] > 0

publie = run_publie()
if publie and publie > index["run"]:
    sys.exit(f"Run {index['run']} plus ancien que le run publié {publie} : publication conservée.")

fichiers = sorted((racine / "departements").glob("*.json"))
assert len(fichiers) == len(index["departments"]) >= 1, f"{len(fichiers)} départements"
assert {f.stem for f in fichiers} >= set(DEPS_DEFAUT) or len(fichiers) == len(index["departments"]), "départements manquants"
communes = 0
for f in fichiers:
    d = json.loads(f.read_text(encoding="utf-8"))
    assert d["schema_version"] == 4 and d["members"] == MEMBERS and d["run_time"] == index["run"]
    assert d["columns"]["values"] == COLONNES
    assert set(d["forecast_statistics"]) == {"median", "p10", "p90"}
    communes += len(d["communes"])
    assert all(0 <= c[6] < len(d["points"]) for c in d["communes"])
    series = {"mean": d["forecast"], **d["forecast_statistics"]}
    for stat, prev in series.items():
        assert len(prev) == len(STEPS), f"{f.name} {stat} : {len(prev)} échéances"
        for k, (t, lignes) in enumerate(prev):
            assert datetime.fromisoformat(t.replace("Z", "+00:00")), t
            assert len(lignes) == len(d["points"])
            for temp, pluie, cumul, rafale in lignes:
                assert math.isfinite(temp) and -80 < temp < 60, f"{f.name} {stat} H+{STEPS[k]} : {temp} °C"
                assert (pluie is None) if k == 0 else (pluie >= 0), f"{f.name} {stat} H+{STEPS[k]} : pluie {pluie}"
                assert (rafale is None) if k == 0 else (0 <= rafale < 400), f"{f.name} {stat} H+{STEPS[k]} : rafale {rafale}"
                assert cumul >= 0
    # quantiles ordonnés et cumul croissant (vrai membre par membre, donc pour chaque quantile)
    for k in range(len(STEPS)):
        for i in range(len(d["points"])):
            p10, med, p90 = (series[s][k][1][i] for s in ("p10", "median", "p90"))
            assert p10[0] <= med[0] + 1e-6 <= p90[0] + 2e-6, f"{f.name} H+{STEPS[k]} : quantiles de température désordonnés"
            if k:
                assert p10[3] <= med[3] + 1e-6 <= p90[3] + 2e-6, f"{f.name} H+{STEPS[k]} : quantiles de rafale désordonnés"
                for s in ("p10", "median", "p90"):
                    assert series[s][k][1][i][2] >= series[s][k - 1][1][i][2] - 0.011, f"{f.name} {s} : cumul décroissant"
assert communes == index["commune_count"], f"{communes} communes au lieu de {index['commune_count']}"

# Membres publiés tels quels : mêmes points et communes que le fichier du département, cumuls croissants,
# et leur moyenne redonne la moyenne publiée (à l'arrondi près).
assert index["member_departments"] == [c for c in DEPS_MEMBRES if c in index["departments"]]
for code in index["member_departments"]:
    m = json.loads((racine / "membres" / f"{code}.json").read_text(encoding="utf-8"))
    dep = json.loads((racine / "departements" / f"{code}.json").read_text(encoding="utf-8"))
    assert m["members"] == MEMBERS == len(m["values"]) and m["run_time"] == index["run"]
    assert m["columns"] == COLONNES_MEMBRES and len(m["times"]) == len(STEPS)
    assert [t for t, _ in dep["forecast"]] == m["times"]
    assert m["points"] == dep["points"] and m["communes"] == dep["communes"], f"membres/{code}.json : points ou communes différents"
    for n, membre in enumerate(m["values"]):
        assert len(membre) == len(STEPS), f"membre {n} : {len(membre)} échéances"
        for k, pts in enumerate(membre):
            assert len(pts) == len(m["points"])
            for i, (temp, cum, raf) in enumerate(pts):
                assert math.isfinite(temp) and -80 < temp < 60, f"membre {n} H+{STEPS[k]} : {temp} °C"
                assert cum <= 0.01 if k == 0 else cum >= membre[k - 1][i][1], f"membre {n} H+{STEPS[k]} : cumul {cum}"
                assert (raf is None) if k == 0 else (raf is not None and 0 <= raf < 400), f"membre {n} H+{STEPS[k]} : rafale {raf}"
    for k in range(len(STEPS)):
        for i in range(len(m["points"])):
            moy = sum(mb[k][i][0] for mb in m["values"]) / MEMBERS
            assert abs(moy - dep["forecast"][k][1][i][0]) < 0.06, f"membres/{code}.json H+{STEPS[k]} : moyenne {moy:.2f} ≠ {dep['forecast'][k][1][i][0]}"
print(f"Validation OK : run {index['run']}, {communes} communes, {len(fichiers)} départements, {len(STEPS)} échéances, {MEMBERS} membres"
      f" (membres publiés : {', '.join(index['member_departments'])}).")
