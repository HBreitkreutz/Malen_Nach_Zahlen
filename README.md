# Malen_Nach_Zahlen

Python-Tool, um aus einem beliebigen Bild eine druckbare **Malen-nach-Zahlen**-Vorlage als PDF zu erzeugen.

## Features

- Eingabe: PNG, JPEG und weitere von Pillow unterstützte Formate
- Konfigurierbare Anzahl reduzierter Farben (`--colors`)
- Zusammenführen zu kleiner Flächen über eine Mindestflächen-Grenze (`--min-region-ratio`, Standard `0.005` = 0,5%)
- Dünne schwarze Umrisslinien auf weißem Hintergrund
- Zahl je Fläche (Farbindex), möglichst zentral über Distanztransform-Punkt innerhalb der Fläche
- Farbleiste unter dem Bild mit Mapping `Nummer -> RGB-Farbe`
- Ausgabe als skalierbare Vektor-PDF

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Nutzung

```bash
python paint_by_numbers.py input.jpg output.pdf --colors 12 --min-region-ratio 0.005
```

### Parameter

- `input_image`: Eingabebild
- `output_pdf`: Ausgabedatei als PDF
- `--colors`: Anzahl Ziel-Farben (>=2)
- `--min-region-ratio`: Minimale Flächengröße relativ zur Gesamtfläche (zwischen 0 und 1)

## Hinweise zur Methodik

- Die Farbquantisierung erfolgt per K-Means, weil dies eine kontrollierte Reduktion auf genau `N` Farben ermöglicht.
- Dithering wird bewusst nicht verwendet, damit keine „gesprenkelten“ Mini-Flächen entstehen.
- Nach der Quantisierung werden verbundene Flächen unterhalb der Mindestgröße erkannt und in benachbarte, farblich ähnliche Regionen gemerged.
- Die Zahlenposition pro Fläche wird über das Maximum der Distanztransform bestimmt (liegt garantiert innerhalb der Fläche, auch bei ringförmigen/konkaven Geometrien).
