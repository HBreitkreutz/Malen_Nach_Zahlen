# Malen_Nach_Zahlen

Python-Tool, um aus einem beliebigen Bild eine druckbare **Malen-nach-Zahlen**-Vorlage als PDF zu erzeugen.

## Features

- Eingabe: PNG, JPEG und weitere von Pillow unterstützte Formate
- Konfigurierbare Anzahl reduzierter Farben (`--colors`)
- Formorientierte Vorsegmentierung via Superpixel (standardmäßig aktiv)
- Zusammenführen zu kleiner Flächen über eine Mindestflächen-Grenze (`--min-region-ratio`, Standard `0.002` = 0,2%)
- Schutz seltener Farben, damit kleine aber wichtige Bereiche (z.B. Augen) nicht verschwinden
- Schutz kontrastreicher Mini-Flächen, auch wenn sie seltene Farbe-Regeln nicht treffen
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
python paint_by_numbers.py input.jpg output.pdf --colors 12 --min-region-ratio 0.002 \
  --rare-color-threshold-ratio 0.02 --rare-color-preserve-components 2 \
  --shape-first --superpixels 1800 --preserve-contrast-threshold 35 \
  --preserve-edge-strength-threshold 0.04 --preserve-thin-structures \
  --detail-edge-threshold 0.085 --detail-min-contour-points 28 --detail-max-contours 0
```

### Parameter

- `input_image`: Eingabebild
- `output_pdf`: Ausgabedatei als PDF
- `--colors`: Anzahl Ziel-Farben (>=2)
- `--shape-first` / `--no-shape-first`: Erst Formen (Superpixel), dann Farben (Standard: aktiv)
- `--superpixels`: Zielanzahl Superpixel (nur mit `--shape-first`, Standard `1800`)
- `--superpixel-compactness`: Formtreue der Superpixel (Standard `10.0`)
- `--min-region-ratio`: Minimale Flächengröße relativ zur Gesamtfläche (zwischen 0 und 1)
- `--rare-color-threshold-ratio`: Farben unterhalb dieses Anteils gelten als „selten" (Standard `0.02` = 2%)
- `--rare-color-preserve-components`: Für seltene Farben werden die größten N Teilflächen nie gemerged (Standard `2`)
- `--preserve-contrast-threshold`: Mini-Flächen mit starkem Nachbarschaftskontrast bleiben erhalten (Standard `35`)
- `--preserve-edge-strength-threshold`: Mini-Flächen mit starken Originalkanten bleiben erhalten (Standard `0.04`)
- `--preserve-thin-structures` / `--no-preserve-thin-structures`: Schützt dünne Linienstrukturen (Standard: aktiv)
- `--detail-edge-threshold`: Zusätzliche Konturen aus Originalkanten, falls farbliche Grenze fehlt (Standard `0.085`, `0` = aus)
- `--detail-min-contour-points`: Mindestlänge dieser Zusatzkonturen (Standard `28`)
- `--detail-max-contours`: Maximale Anzahl Zusatzkonturen, nach Länge priorisiert (Standard `0` = unbegrenzt)

## Hinweise zur Methodik

- Standard-Pipeline: SLIC-Superpixel (Form/Kanten) und danach K-Means auf Superpixel-Mittelfarben.
- Optional kann mit `--no-shape-first` auf reine Farbquantisierung pro Pixel zurückgeschaltet werden.
- Dithering wird bewusst nicht verwendet, damit keine „gesprenkelten“ Mini-Flächen entstehen.
- Nach der Quantisierung werden verbundene Flächen unterhalb der Mindestgröße erkannt und in benachbarte, farblich ähnliche Regionen gemerged.
- Seltene Farben können geschützt werden: Die größten Teilflächen pro seltener Farbe bleiben erhalten und werden nicht wegoptimiert.
- Zusätzlich bleiben kleine Regionen mit hohem Farbkontrast zur direkten Nachbarschaft erhalten.
- Zusätzlich bleiben kleine Regionen mit starken Originalbild-Kanten erhalten.
- Dünne, längliche Regionen (z.B. Schnüre) werden bevorzugt erhalten.
- Fehlende Objektkonturen können zusätzlich aus dem Originalbild gezeichnet werden, falls sie nicht schon durch Farbgrenzen erfasst wurden.
- Die Zahlenposition pro Fläche wird über das Maximum der Distanztransform bestimmt (liegt garantiert innerhalb der Fläche, auch bei ringförmigen/konkaven Geometrien).
