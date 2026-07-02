# ancient_ML

A local research/experiment framework for a **Synthetic Vedic Oracle Kundli** system.

The project has three independent layers:

1. **Searcher daemon**: continuously searches artificial Vedic birth seeds: date, time, latitude, longitude.
2. **Trainer daemon**: trains or continues a number-ranking ML model when a better seed appears.
3. **Predictor**: answers, at any time, what to play for a requested Lotto 6/49 draw date using the latest promoted stable model.

The Vedic/Jyotish calculation layer is based on real astronomical ephemeris calculation through **Swiss Ephemeris / `pyswisseph`**, using **Lahiri sidereal ayanamsha**. It computes traditional chart features such as graha longitudes, rāśi, navāṁśa, nakshatra, pada, Lagna, Rahu/Ketu, Vimshottari dasha, and whole-sign houses.

The final conversion from Vedic chart features into Lotto 6/49 numbers is necessarily experimental. Traditional Vedic astrology does not contain a classical Lotto 6/49 rule.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Input data format

Create a CSV file such as `data/draws.csv` using rows like:

```csv
"649",4421,0,"2026-06-03",3,5,6,28,30,37,19
"649",4422,0,"2026-06-06",9,27,30,33,45,47,20
```

Columns are interpreted as:

```text
game, draw_id, ignored, draw_date, n1, n2, n3, n4, n5, n6, bonus
```

## Layer 1: continuous seed search

Run a bounded search:

```bash
python -m ancient_ml search \
  --data data/draws.csv \
  --state state/ancient_ml.sqlite \
  --trials 50000 \
  --top-k 50
```

Run an infinite search:

```bash
python -m ancient_ml search \
  --data data/draws.csv \
  --state state/ancient_ml.sqlite \
  --trials 10000 \
  --forever
```

The searcher writes promising synthetic kundli seeds into SQLite.

## Layer 2: ML training

Train from the best untrained/promising seeds:

```bash
python -m ancient_ml train \
  --data data/draws.csv \
  --state state/ancient_ml.sqlite \
  --models models \
  --max-candidates 5
```

The trainer promotes a model only if it improves validation score compared with the currently promoted model.

## Layer 3: prediction

Ask what to play for a draw date:

```bash
python -m ancient_ml predict \
  --date 2026-06-27 \
  --state state/ancient_ml.sqlite \
  --models models
```

The predictor never waits for search/training. It uses the latest promoted model if available, otherwise it falls back to the best available seed and cheap ranker.

## Project structure

```text
ancient_ML/
├── ancient_ml/
│   ├── __main__.py
│   ├── data.py
│   ├── feature_builder.py
│   ├── predictor.py
│   ├── registry.py
│   ├── scoring.py
│   ├── searcher.py
│   ├── trainer.py
│   └── vedic_engine.py
├── configs/
│   └── default.json
├── data/
│   └── draws_example.csv
├── tests/
│   └── test_data.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Notes

This is an experimental research toy. The validation logic is intentionally time-split so that a seed or model has to perform on later draws, not only on the data used to fit it.
