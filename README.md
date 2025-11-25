# Chess Tracker

## Modelle und Datensätze

### Modelle

Alle verwendeten Modelle sind im Order `model/` zu finden:

- `chessboard_corners.pt` für die Eckenerkennung mittels YOLO11n
- `chessboard_segmentation.pt` für die Segmentierung mittels YOLO11n-seg
- `piece_detection.pt` für die Figurenerkennung mittels YOLO11n-pose

### Datensätze

Alle Datensätze sind auf Roboflow zu finde:

- [Eckenerkennung](https://app.roboflow.com/realtime-chessbord-tracking/chessboard-corners-wg40n/7)
- [Segmentierung](https://app.roboflow.com/realtime-chessbord-tracking/chessboard-segmentation-7pdqo/7)
- [Figurenerkennung](https://app.roboflow.com/realtime-chessbord-tracking/chess-pieces-with-keypoints/14)


## Project Structure

```
gradio-chess-tracker/
├── .env
├── 1_accuracy.py
├── 2_euclidean_distance.py
├── 3_pck.py
├── algorithms/
├── misc/
├── data/
├── images/
├── model/

├── media/
├── position_to_image.py
├── record_game.py
├── setup_game.py
├── board.py
├── gradio_app.py

├── Dockerfile
└── requirements.txt
```

## Lokale Installation

> **Hinweis:** Für das Ausführen der Web-App, kann auch das Docker-Image verwendet werden. Die Anleitung dazu ist weiter unten zu finden.

### 1. Repository klonen

```
git clone git@github.com:semdner/gradio-chess-tracker.git
```

```
cd gradio-chess-tracker/
```

### 2. Python Virtual Environment erstellen und aktivieren

```
python3 -m venv .venv
```

```
source .venv/bin/activate
```

### 3. Dependencies insallieren

```
pip install -r requirements.txt
```

## Schachbretterkennungs-Algorithmen ausprobieren

Für jeden Algorithmus ist ein Jupyter-Notebook im Ordner `algorithms/`. Die Testdaten sind im Ordner `images/`. Zum probieren eines anderen Bildes in der folgenden Zeile das Bild definieren:

```
img = cv2.imread("../images/image_name.png")
```

### Algorithmen

- **`cameracalibration_2.ipynb`**: Angepasste Kamerakalibrierung mit 3 x 7 Corners
- **`cameracalibration.ipynb`**: Angepasste Kamerakalibrierung mit 7 x 7 Corners
- **`contourapprox.ipynb`**: Schachbretterkennung mittels Konturappoximation
- **`harriscorner.ipynb`**: Schachbretterkennung mittels Harris Corner Detection
- **`yolodetect.py`**: Eckenereknnung mittels YOLO11n (Modell: `model/chessboard_corners.pt`)
- **`yolosegment.py`**: Schachbretterkennung durch das YOLO11n-seg Modell (Modell: `model/chessboard_seg.pt`)


## Auswertung der Algorithmen

### Relevante Dateien

Die für die Auswertung der Algorithmen relevante Dateien befinden sind:

- **`.env`**: hier wird definiert welcher Algorithmus ausgewertet werden soll indem man den wert `ALGORITHM` einen der Werte `harris_corner_detection`, `contour_approx`, `camera_calibration`, `line_detection`, `yolo_detect`, `yolo_segment`
- **`images/`**: enthält die Testdaten (Bilder)
- **`data/`**: enthält eine JSON mit den Ground-Truth-Werten für die Testdaten
- **`misc/`**: enthält Dateien die Dateipfade definieren, der Berechnung der Homography dienen und Hilfsfunktionen für die Auswertung.
- **`algorithms/`**: enthält die Algorithmen zur Auswertung
- **`1_accuracy.py`**: Auswertung wieviele Schachbretter richtig erkannt wurden (Genauigkeit)
- **`2_euclidean_distance.py`**: Auswertung durchschnittliche euklidische Distanz
- **`3_pck.py`** Auswertung Percentage of Correct Keypoints

### Auswertung testen

#### 1. In `.env` den Wert ALGORITHM setzen (default `ALGORITHM="yolo_segment"`) 

#### 2. Dateien ausführen

```
python 1_accuracy.py
python 2_euclidean_distance.py
python 3_pck.py
```

## Web-App Ausführen

### Docker


```
docker pull ghcr.io/semdner/chess-tracker:latest
```

```
docker run -p 7860:7860 ghcr.io/semdner/chess-tracker:latest
```

Dannach muss im Browser `http://127.0.0.1:7860/` geöffnet werden und die App kann benutzt werden.

### Lokale Ausführung

Nach der Installation folgenden Command ausführen:

```
python gradio-app.py
```

Dannach muss im Browser `http://127.0.0.1:7860/` geöffnet werden und die App kann benutzt werden.

### Nutztung der Web-App

1. Kamera links vom Brett positionen.

2. Die weißen Figuren müssen rechts im Bild bzw. die schwarzen Figuren links im Bild seien.

3. Das Brett und die Figuren müssen gut sichtbar sein

4. Die Kamera auswählen und auf "Record" drücken

5. Dannach den "Calibration"-Button drücken um die Kalibrierung zu starten. Wenn das Schachbrett in der Startposition als Bild erscheint ist die Kalibierung erfolgreich.

6. Dannach auf den "Record"-Button Drücken und den ersten Zug spielen (Die Zugerkennung kann von wenigen Sekunden bis etwa 1 Minute dauern). Sobald das Bild mit der neuen Position erscheint kann der nächst Zug gespielt werden.