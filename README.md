# PEARP 0,1° Europe — Alertes Météo

Ensemble ARPEGE (PEARP, 35 membres) à la maille de 0,1° (environ 11 km) sur l'Europe, en tableaux par commune
pour les Pyrénées-Orientales et les départements voisins (Aude, Ariège, Hérault), au même format que les autres
ensembles (AIFS ENS, GEFS, AIGEFS).

## Source

API « Modèle ARPEGE Prévision d'Ensemble » (PE-ARPEGE) du portail Météo-France, service WCS
`https://public-api.meteofrance.fr/public/pearpege/1.0/wcs/MF-NWP-GLOBAL-PEARP{membre}-01-EUROPE-WCS`.

- **Dans l'adresse, le « {run} » de la documentation est le NUMÉRO DE MEMBRE** (000 à 034), pas l'heure du run.
  Le run est dans l'identifiant de couverture (`…___2026-11-20T12.00.00Z`).
- 4 runs par jour (00, 06, 12, 18 UTC), échéances horaires jusqu'à H+102 (lue dans DescribeCoverage), 5 jours de rétention, pas d'archive.
- Un appel = 1 membre, 1 paramètre, 1 échéance. L'API refuse le découpage lat/lon : chaque appel renvoie toute la
  grille (environ 0,6 Mo, 0,3 s).
- Champs lus : `TEMPERATURE` à 2 m, `TOTAL_WATER_PRECIPITATION` sur 3 h (`_PT3H`), composantes U et V de la rafale
  (maximum de l'heure écoulée). Rien d'autre : pas de neige, pas de vent moyen.
- Environ 9 500 appels par run (35 membres × (35 températures + 34 pluies + 204 rafales)).

## Publication (branche `data`, réécrite à chaque run)

- `index.json` : run, échéances (`table_steps`, pas de 3 h), colonnes, départements, limites ;
- `departements/XX.json` : format « points + communes », schéma 4. Colonnes `temperature_c`, `precipitation_mm`
  (3 h, absente à H+0), `precipitation_total_mm`, `wind_gust_kmh` (maximum des 3 rafales horaires de la tranche, absente à H+0).
  `forecast` = moyenne des 35 membres ; `forecast_statistics` = `median`, `p10`, `p90` ;
- `membres/66.json` : les 35 membres tels quels, `values[membre][échéance][point]` = `[température, cumul de pluie, rafale]`.

Les statistiques sont calculées membre par membre puis sur les 35 membres. Valeurs au point de grille le plus proche de
chaque commune, sans correction d'altitude (en montagne, les modèles s'écartent de plusieurs °C des stations).

## Production

Workflow `build-pearp.yml` (programmé 2 fois par jour et manuel, option « force »). Le run le plus récent dont la dernière
échéance existe pour le dernier membre est choisi, sinon on recule d'un run. Chaque message GRIB est contrôlé (run,
échéance, membre, grille, unité, niveau, type de pas, valeurs manquantes) et `validate_data.py` vérifie toute la publication
avant de la mettre en ligne.

**Secret requis : `METEOFRANCE_PEARP_API_KEY`**, clé de l'application Météo-France abonnée à PE-ARPEGE
(jamais en dur). Le quota du portail est partagé avec les autres pipelines : `--pause` espace les appels si besoin.

Tests : `python -m unittest discover -s tests`.
