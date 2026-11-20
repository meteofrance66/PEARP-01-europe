"""Lecture stricte de PEARP 0,1° Europe (ensemble ARPEGE, API WCS de Météo-France).

35 membres (000 à 034), grille 0,1° sur l'Europe (lon -32 à 42, lat 20 à 72), échéances horaires,
4 runs par jour (00, 06, 12 et 18 UTC), 5 jours de rétention, pas d'archive. Un appel = 1 membre,
1 paramètre, 1 échéance : champ complet de la grille (l'API refuse de découper lat/lon), environ 0,6 Mo.

Dans l'adresse du service, le « {run} » de la documentation est en réalité le NUMÉRO DE MEMBRE :
`MF-NWP-GLOBAL-PEARP{membre:03d}-01-EUROPE-WCS`.

Champs lus : température à 2 m (instantanée), pluie cumulée sur 3 h (`_PT3H`) et composantes U et V
de la rafale (maximum de l'heure écoulée). Rien d'autre : pas de neige, pas de vent moyen.
"""
from __future__ import annotations

import math
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import requests

MEMBERS = 35
PAS = 0.1
NI, NJ = 741, 521
LAT_FIRST, LON_FIRST = 72.0, -32.0
PAS_TABLE = 3  # heures entre deux échéances publiées (les rafales utilisent les 3 heures de la tranche)
BASE = "https://public-api.meteofrance.fr/public/pearpege/1.0/wcs"

# Champ -> (nom de couverture, dimensions supplémentaires). `{run}` est l'instant du run : AAAA-MM-JJThh.mm.ssZ.
COUVERTURES = {
    "t": "TEMPERATURE__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND___{run}",
    "p": "TOTAL_WATER_PRECIPITATION__GROUND_OR_WATER_SURFACE___{run}_PT3H",
    "u": "U_COMPONENT_OF_WIND_GUST__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND___{run}",
    "v": "V_COMPONENT_OF_WIND_GUST__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND___{run}",
}
HAUTEUR = {"t": 2, "u": 10, "v": 10}

# Contrôles de chaque message GRIB (en plus du run, de l'échéance et du membre).
NOMS = {"t": {"2t"}, "p": {"tirf"}, "u": {"max_10efg", "10efg"}, "v": {"max_10nfg", "10nfg"}}
UNITES = {"t": "K", "p": "kg m**-2", "u": "m s**-1", "v": "m s**-1"}
NIVEAUX = {"t": ("heightAboveGround", 2), "p": ("surface", 0), "u": ("heightAboveGround", 10), "v": ("heightAboveGround", 10)}
TYPES_PAS = {"t": "instant", "p": "accum", "u": "max", "v": "max"}
DUREE = {"t": 0, "p": 3, "u": 1, "v": 1}  # endStep - startStep


def iso_wcs(run: datetime) -> str:
    return run.astimezone(timezone.utc).strftime("%Y-%m-%dT%H.%M.%SZ")


def steps(horizon: int) -> list[int]:
    """Échéances du tableau : H+0, H+3, … jusqu'à l'horizon (arrondi vers le bas à un multiple de 3 h)."""
    return list(range(0, horizon - horizon % PAS_TABLE + 1, PAS_TABLE))


class Api:
    """Client WCS de Météo-France : clé d'application en en-tête `apikey`, reprise sur coupure et sur 429."""

    def __init__(self, cle: str, pause: float = 0.0, tentatives: int = 4):
        if not cle:
            raise SystemExit("Clé absente : définir METEOFRANCE_PEARP_API_KEY (secret GitHub, jamais en dur).")
        self.cle, self.pause, self.tentatives = cle, pause, tentatives
        self._local = threading.local()

    def session(self) -> requests.Session:
        s = getattr(self._local, "s", None)
        if s is None:
            s = self._local.s = requests.Session()
            s.headers["apikey"] = self.cle
        return s

    def get(self, membre: int, operation: str, params: dict[str, Any], binaire: bool = False):
        """Appelle une opération WCS ; renvoie le corps (texte ou octets) ou None si 403/404 (donnée pas encore là)."""
        url = f"{BASE}/MF-NWP-GLOBAL-PEARP{membre:03d}-01-EUROPE-WCS/{operation}"
        q = [("service", "WCS"), ("version", "2.0.1")]
        for cle, valeur in params.items():
            if isinstance(valeur, list):
                q += [(cle, v) for v in valeur]
            else:
                q.append((cle, valeur))
        for k in range(self.tentatives):
            if self.pause:
                time.sleep(self.pause)
            try:
                r = self.session().get(url, params=q, timeout=120)
                if r.status_code in (403, 404):
                    return None
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()
                return r.content if binaire else r.text
            except requests.RequestException as e:
                if k == self.tentatives - 1:
                    raise
                attente = 20 * (k + 1)
                print(f"{operation} membre {membre} : à reprendre ({e}) ; attente {attente} s", flush=True)
                time.sleep(attente)
        raise AssertionError("inaccessible")

    def coverage(self, run: datetime, champ: str, membre: int, heure: int) -> bytes | None:
        sous = [f"time({heure * 3600})"]
        if champ in HAUTEUR:
            sous.append(f"height({HAUTEUR[champ]})")
        return self.get(
            membre,
            "GetCoverage",
            {"coverageid": COUVERTURES[champ].format(run=iso_wcs(run)), "subset": sous, "format": "application/wmo-grib"},
            binaire=True,
        )


def runs_disponibles(api: Api) -> list[datetime]:
    """Runs listés par GetCapabilities (membre 0), du plus récent au plus ancien."""
    xml = api.get(0, "GetCapabilities", {"language": "eng"})
    if xml is None:
        raise RuntimeError("GetCapabilities indisponible")
    trouves = set(re.findall(r"TEMPERATURE__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND___(\d{4}-\d\d-\d\dT\d\d\.\d\d\.\d\dZ)<", xml))
    runs = [datetime.strptime(t, "%Y-%m-%dT%H.%M.%SZ").replace(tzinfo=timezone.utc) for t in trouves]
    return sorted(runs, reverse=True)


def horizon_du_run(api: Api, run: datetime) -> int:
    """Dernière échéance (heures) annoncée par DescribeCoverage de la température."""
    xml = api.get(0, "DescribeCoverage", {"coverageid": COUVERTURES["t"].format(run=iso_wcs(run))})
    if xml is None:
        raise RuntimeError(f"DescribeCoverage introuvable pour le run {run:%Y-%m-%d %H} UTC")
    fin = re.search(r"<gml:endPosition[^>]*>([^<]+)<", xml)
    if not fin:
        raise RuntimeError("DescribeCoverage sans endPosition")
    t = datetime.strptime(fin.group(1), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return int((t - run).total_seconds() // 3600)


def latest_run(api: Api, candidats: int = 3) -> tuple[datetime, int]:
    """Run le plus récent complet : sa dernière échéance doit exister pour le dernier membre (sinon on recule)."""
    for run in runs_disponibles(api)[:candidats]:
        horizon = horizon_du_run(api, run)
        h = horizon - horizon % PAS_TABLE
        if h >= PAS_TABLE and api.coverage(run, "p", MEMBERS - 1, h) is not None:
            return run, h
        print(f"Run {run:%Y-%m-%d %H} UTC pas encore complet (H+{h} absent du membre {MEMBERS - 1}).", flush=True)
    raise RuntimeError("aucun run complet parmi les plus récents")


def grid_index(lat: float, lon: float) -> tuple[int, float, float]:
    """Point de grille le plus proche : indice dans le champ, latitude et longitude du point."""
    row = min(NJ - 1, max(0, int(round((LAT_FIRST - lat) / PAS))))
    col = min(NI - 1, max(0, int(round((lon - LON_FIRST) / PAS))))
    return row * NI + col, round(LAT_FIRST - row * PAS, 4), round(LON_FIRST + col * PAS, 4)


def check_message(get: Callable[[str], Any], run: datetime, champ: str, membre: int, heure: int) -> None:
    """Vérifie un message (grille, run, échéance, membre, unité, niveau, type de pas) ; lève ValueError sinon."""
    etiquette = f"{champ} membre {membre} H+{heure}"
    nom = str(get("shortName"))
    if nom not in NOMS[champ]:
        raise ValueError(f"{etiquette} : champ inattendu {nom!r}")
    lon_first = (float(get("longitudeOfFirstGridPointInDegrees")) + 180.0) % 360.0 - 180.0
    attendu = {
        "Ni": NI,
        "Nj": NJ,
        "jScansPositively": 0,
        "numberOfForecastsInEnsemble": MEMBERS,
        "number": membre,
        "dataDate": int(run.strftime("%Y%m%d")),
        "dataTime": run.hour * 100,
        "endStep": heure,
        "startStep": heure - DUREE[champ],
        "stepType": TYPES_PAS[champ],
        "units": UNITES[champ],
        "typeOfLevel": NIVEAUX[champ][0],
        "level": NIVEAUX[champ][1],
    }
    for cle, valeur in attendu.items():
        recu = get(cle)
        if recu != valeur:
            raise ValueError(f"{etiquette} : {cle} = {recu!r} au lieu de {valeur!r}")
    if not math.isclose(float(get("latitudeOfFirstGridPointInDegrees")), LAT_FIRST, abs_tol=1e-6) or not math.isclose(lon_first, LON_FIRST, abs_tol=1e-6):
        raise ValueError(f"{etiquette} : grille inattendue")
    if not math.isclose(float(get("iDirectionIncrementInDegrees")), PAS, abs_tol=1e-9):
        raise ValueError(f"{etiquette} : pas de grille inattendu")


def decode(octets: bytes, run: datetime, champ: str, membre: int, heure: int, indexes: list[int]) -> np.ndarray:
    """Valeurs du message aux points demandés ; échoue sur un message absent, multiple ou à valeurs manquantes."""
    from eccodes import codes_get, codes_get_double_elements, codes_new_from_message, codes_release

    gid = codes_new_from_message(octets)
    try:
        check_message(lambda k: codes_get(gid, k), run, champ, membre, heure)
        valeurs = np.asarray(codes_get_double_elements(gid, "values", indexes), dtype=np.float64)
        manquant = float(codes_get(gid, "missingValue"))
        if not np.isfinite(valeurs).all() or np.isclose(valeurs, manquant).any():
            raise ValueError(f"{champ} membre {membre} H+{heure} : valeurs manquantes")
        return valeurs
    finally:
        codes_release(gid)
