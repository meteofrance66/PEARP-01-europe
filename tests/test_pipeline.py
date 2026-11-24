import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_data  # noqa: E402
from source import MEMBERS, NI, check_message, grid_index, steps  # noqa: E402

RUN = datetime(2026, 11, 20, 12, tzinfo=timezone.utc)


def message(**surcharge):
    """Clés d'un message 2t du membre 3 à H+24 du run 12 UTC (tel que lu dans les vrais fichiers de l'API)."""
    m = {
        "shortName": "2t", "number": 3, "Ni": 741, "Nj": 521, "jScansPositively": 0,
        "numberOfForecastsInEnsemble": 35, "dataDate": 20261120, "dataTime": 1200, "startStep": 24, "endStep": 24,
        "stepType": "instant", "units": "K", "typeOfLevel": "heightAboveGround", "level": 2,
        "latitudeOfFirstGridPointInDegrees": 72.0, "longitudeOfFirstGridPointInDegrees": 328.0, "iDirectionIncrementInDegrees": 0.1,
    }
    m.update(surcharge)
    return m.__getitem__


class Grille(unittest.TestCase):
    def test_perpignan_tombe_sur_la_maille_la_plus_proche(self):
        idx, lat, lon = grid_index(42.699, 2.9045)
        self.assertEqual((lat, lon), (42.7, 2.9))
        self.assertEqual(idx, 293 * NI + 349)

    def test_longitudes_ouest(self):
        _, _, lon = grid_index(48.39, -4.49)  # Brest
        self.assertEqual(lon, -4.5)

    def test_echeances(self):
        self.assertEqual(steps(102)[:3], [0, 3, 6])
        self.assertEqual(steps(102)[-1], 102)
        self.assertEqual(steps(100)[-1], 99)


class Messages(unittest.TestCase):
    def test_message_conforme(self):
        check_message(message(), RUN, "t", 3, 24)

    def test_pluie_3h(self):
        m = message(shortName="tirf", units="kg m**-2", stepType="accum", startStep=21, typeOfLevel="surface", level=0)
        check_message(m, RUN, "p", 3, 24)

    def test_rafale_horaire(self):
        m = message(shortName="max_10efg", units="m s**-1", stepType="max", startStep=23, level=10)
        check_message(m, RUN, "u", 3, 24)

    def test_refus(self):
        for cle, valeur in [("dataTime", 0), ("endStep", 30), ("units", "C"), ("Ni", 720), ("numberOfForecastsInEnsemble", 51),
                            ("number", 4), ("longitudeOfFirstGridPointInDegrees", 0.0), ("shortName", "10u")]:
            with self.subTest(cle=cle), self.assertRaises(ValueError):
                check_message(message(**{cle: valeur}), RUN, "t", 3, 24)


class Statistiques(unittest.TestCase):
    def test_quantiles_sur_les_membres(self):
        v = np.arange(MEMBERS, dtype=float).reshape(MEMBERS, 1)
        s = build_data.statistiques(v)
        self.assertAlmostEqual(s["mean"][0], 17.0)
        self.assertAlmostEqual(s["median"][0], 17.0)
        self.assertAlmostEqual(s["p10"][0], 3.4)
        self.assertAlmostEqual(s["p90"][0], 30.6)

    def test_membre_manquant_refuse(self):
        with self.assertRaises(ValueError):
            build_data.statistiques(np.zeros((MEMBERS - 1, 3)))
        v = np.zeros((MEMBERS, 2))
        v[4, 1] = np.nan
        with self.assertRaises(ValueError):
            build_data.statistiques(v)

    def test_rafale_3h_est_le_maximum_des_trois_heures(self):
        v = np.full((1, 7, 1), np.nan)  # heures 0..6, 1 membre, 1 point
        v[0, 1:, 0] = [1.0, 5.0, 2.0, 3.0, 10.0, 4.0]  # m/s, heures 1 à 6
        r = build_data.rafale_3h(v, 3)
        self.assertTrue(np.isnan(r[0, 0, 0]))
        self.assertAlmostEqual(r[0, 1, 0], 5.0 * 3.6)
        self.assertAlmostEqual(r[0, 2, 0], 10.0 * 3.6)

    def test_lignes_dans_l_ordre_des_colonnes(self):
        stats = {
            "temperature": {"mean": np.array([[12.345]])},
            "pluie": {"mean": np.array([[np.nan]])},
            "cumul": {"mean": np.array([[0.0]])},
            "rafale": {"mean": np.array([[np.nan]])},
        }
        self.assertEqual(build_data.lignes(stats, "mean", 0, [0]), [[12.35, None, 0.0, None]])
        self.assertEqual(build_data.COLONNES, ["temperature_c", "precipitation_mm", "precipitation_total_mm", "wind_gust_kmh"])


class Membres(unittest.TestCase):
    def test_membres_publies_tels_quels_dans_l_ordre_du_departement(self):
        import json
        import tempfile

        table = steps(24)
        indexes = [10, 20, 30]
        points = {10: (42.5, 2.7), 20: (43.0, 1.0), 30: (42.7, 3.0)}
        c1 = ["66136", "Perpignan", "66", ["66000"], 121616, 42.699, 2.9045]
        c2 = ["66025", "Bourg-Madame", "66", ["66760"], 1300, 42.43, 1.94]
        par_dep = {"66": [(c1, 30), (c2, 10)], "11": [(["11069", "Carcassonne", "11", ["11000"], 46000, 43.21, 2.35], 20)]}
        temperature = np.zeros((MEMBERS, len(table), 3))
        cumul = np.zeros((MEMBERS, len(table), 3))
        rafale = np.full((MEMBERS, len(table), 3), np.nan)
        rafale[:, 1:] = 50.0
        temperature[7, 4, 2] = 21.26  # membre 7, H+12, point 30
        cumul[7, 4:, 2] = 3.456
        with tempfile.TemporaryDirectory() as d:
            build_data.ecrire_membres(RUN, table, indexes, points, par_dep, temperature, cumul, rafale, ["66"], Path(d))
            m = json.loads((Path(d) / "membres" / "66.json").read_text(encoding="utf-8"))
            self.assertFalse((Path(d) / "membres" / "11.json").exists())
        self.assertEqual(m["members"], MEMBERS)
        self.assertEqual(len(m["values"]), MEMBERS)
        self.assertEqual(m["times"][4], "2026-11-21T00:00:00Z")
        self.assertEqual(m["points"], [[10, 42.5, 2.7, None], [30, 42.7, 3.0, None]])
        self.assertEqual([c[-1] for c in m["communes"]], [1, 0])
        self.assertEqual(m["values"][7][4][1], [21.3, 3.46, 50.0])
        self.assertEqual(m["values"][0][0][0], [0.0, 0.0, None])


class Catalogue(unittest.TestCase):
    def test_toute_la_france_par_defaut(self):
        indexes, points, par_dep, n = build_data.charger_catalogue(None)
        self.assertEqual(len(par_dep), 96)
        self.assertGreater(n, 34000)
        self.assertEqual(len(indexes), len(points))
        self.assertTrue({"66", "11", "09", "34", "2A", "2B", "75"} <= set(par_dep))

    def test_zone_d_origine_avec_les_departements_demandes(self):
        _, _, par_dep, _ = build_data.charger_catalogue(build_data.DEPS_DEFAUT)
        self.assertEqual(sorted(par_dep), ["09", "11", "34", "66"])

    def test_departement_inconnu_refuse(self):
        with self.assertRaises(ValueError):
            build_data.charger_catalogue(["66", "999"])


if __name__ == "__main__":
    unittest.main()
