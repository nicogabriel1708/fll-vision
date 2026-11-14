# FLL-Vision: Complete Production-Ready Robot Game Livestream & AI Tracking System

`fll-vision` ist ein vollständiges, produktionsreifes Computer-Vision-, Telemetrie- und Livestream-System für das **FIRST LEGO League (FLL) Robot Game**, speziell optimiert für den Betrieb in **einem einzigen Docker-Container** auf dem **NVIDIA DGX Spark (GB10 Grace Blackwell Superchip, 20-Core ARM64 CPU + SM 12.1 GPU, 128 GB Unified Memory)**.

Das Repository vereint das KI-Backend (Multi-Class YOLO11 + NVIDIA TensorRT FP16, 4D-Kalman-Filter, Hand-Veto State Machine, Halbtisch-Maskierung, 2D-Heatmaps und paralleler Dual-Team-Auto-Zoom-Compositor) mit dem originalen `fll-robot-game-livestream` Frontend in einer einzigen Anwendung.

--------------------------------------------------------------------------------

## 1\. Übersicht der Seiten, Routen & Benutzer (Pages, Routes & Users)

Name der Seite           | Route            | Login-Route                    | User (Benutzer / Rolle)            | Beschreibung
:----------------------- | :--------------- | :----------------------------- | :--------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Match-Stream**         | `/`              | `-` _(Öffentlich / Read-Only)_ | `-` _(Kein Login erforderlich)_    | Öffentlicher Livestream im exakten `fll-robot-game-livestream` Design (Orange Topbar, abgeschrägte Red/Blue Team-Banner in `CARDOT`, `8:88` 7-Segment-Timer in `Digital Dismay`, Original-Audio-SFX, Idle-Screen) mit beiden sauberen Team-Tisch-Streams nebeneinander (`/video-stream/red-team` & `/video-stream/blue-team`) bei identischem Auto-Zoom und ohne Heatmap/HUD.
**Volunteer Control**    | `/control`       | `/login/control`               | **`volunteers`** _(sowie `admin`)_ | Mobile-optimierte Match-Steuerung für Volunteers: `.txt`-Teamlisten-Upload & Team-Auswahl (Red Team / Blue Team), Einstellen der Match- & Pre-Countdown-Zeiten, Start/Pause/Reset/Abort, Idle-Screen, Kamera-Swap, **Umschalten der Video-Quelle (Standard: USB-Kamera `/dev/video0`, oder RTSP-Link / `.mp4` mit Bild-Check vor dem Speichern)** sowie sämtliche Tastatur-Shortcuts (`Hold S`, `Hold R`, `Hold E`, `Space`, `A`/`Esc`, `W`, `↑`/`↓`).
**System Configuration** | `/configuration` | `/login/configuration`         | **`engineers`** _(sowie `admin`)_  | Geschützte Backend-Konfigurationsseite mit 4 Menüs/Tabs auf derselben Seite:<br>
1\. Video-Quellen-Auswahl (USB/RTSP/MP4 mit Bild-Check), Tisch-Größe, Selection-Box Crop & **Aktive Tischhälften-Maskierung** (ignoriert die ungenutzte Tischhälfte mit Händen/Flaschen sowie den Außenbereich)<br>
2\. **Einheitlicher Auto-Zoom-Faktor (`zoom_factor`)** für beide Team-Streams & 1:1 Seitenrand-Korrektur<br>
3\. Live **2D Bird's-Eye Heatmap** & **Lauf-Dokumentation (Runs)** mit PNG-, CSV- & JSON-Export<br>
4\. KI-Tracker-, Kalman- & Hand-Veto-Parameter
**Admin Panel**          | `/admin`         | `/login/admin`                 | **`admin`**                        | Geschützte Benutzer- und Teamverwaltung für den Hauptadministrator (`admin`) zum Anlegen, Passwort-Ändern und Löschen von `volunteers`-, `engineers`- und `admin`-Benutzern sowie zur Pflege der FLL-Teams in der SQLite-Datenbank.

--------------------------------------------------------------------------------

## 2\. NVIDIA DGX Spark Optimierungen (Multi-Core CPU + Blackwell GPU)

Das System nutzt die Architektur des **NVIDIA DGX Spark** (20 ARM64 Grace CPU-Kerne, 128 GB Unified LPDDR5x Memory, GB10 Blackwell GPU) auf allen Ebenen optimal aus:

- **Native ARM64 + Blackwell (`sm_121`) Container-Basis:** Basiert auf `nvcr.io/nvidia/pytorch:26.10-py3` (`linux/arm64`) mit `TORCH_CUDA_ARCH_LIST="12.0;12.1"`, `cudnn.benchmark = True`, `allow_tf32 = True` und `ipc: host` (Zero-Copy Shared Memory).
- **FP16 NVIDIA TensorRT Engine (`.engine`):** Kompiliert das trainierte YOLO11-Modell beim ersten Start automatisch nativ für die GB10 GPU (`< 2 ms` Inferenzzeit pro Frame).
- **Multi-Thread CPU-Parallelisierung (`src/main.py`, `src/stream_compositor.py`, `src/server.py`):**

  1. **`AsyncVideoReader` (`video-decoder` Thread):** Entkoppelt das Kamera-/Video-Decoding in einen eigenen Hintergrund-Thread, sodass Video-I/O niemals die GPU blockiert, und unterstützt unterbrechungsfreies Live-Hot-Swapping der Quelle.
  2. **Paralleler 2-Thread `DualStreamCompositor` (`stream-crop-0..1`):** Berechnet Auto-Zoom-Crop und Skalierung für beide Team-Tische parallel auf zwei CPU-Kernen.
  3. **Paralleler 3-Thread JPEG-Encoder (`jpeg-enc-0..2`):** Kodiert `Stream 1`, `Stream 2` und die `Minimap` gleichzeitig über drei CPU-Threads.
  4. **ARM NEON SIMD & OpenCV-Threading:** Aktiviert `cv2.setUseOptimized(True)` und skaliert OpenCV- sowie PyTorch-DataLoader-Worker automatisch auf die verfügbaren CPU-Kerne (`min(16, max(4, os.cpu_count() - 2))`).

- **Zentrales Logging & Error-Handling (`src/logger.py`):** Alle Module loggen strukturiert auf die Konsole sowie in ein rotierendes Logfile unter `data/outputs/fll_vision.log` (max. `5 × 10 MB`).

--------------------------------------------------------------------------------

## 3\. Setup (Keine Default-Passwörter – Interaktiv oder per CLI-Parameter)

Im gesamten Projekt existieren **keine Standard-/Default-Passwörter**. Beim Setup ist lediglich das Passwort für den **`admin`**-User erforderlich; die Accounts für **`volunteers`** (`/control`) und **`engineers`** (`/configuration`) können optional direkt mit angelegt oder später im **Admin Panel** (`/admin`) erstellt werden.

### Schritt 1: Repository klonen (oder ZIP entpacken)

```bash
git clone <dein-github-repo-url> fll-vision
cd fll-vision
chmod +x setup.sh entrypoint.sh run_all.sh scripts/setup_db.py
```

### Schritt 2: `./setup.sh` ausführen (Interaktiv oder mit Argumenten)

#### Variante A: Interaktives Setup (ohne Argumente – Skript fragt alles Schritt für Schritt ab)

```bash
./setup.sh
```

Wenn du keine Argumente übergibst, fragt dich `./setup.sh` automatisch interaktiv nach:

1. Dem **erforderlichen Admin-Passwort** für `admin` (`/admin`).
2. **Optional** nach Benutzername & Passwort für **Volunteer Control** (`/control`, Standardname `volunteers` – einfach `Enter` drücken zum Überspringen).
3. **Optional** nach Benutzername & Passwort für **System Configuration** (`/configuration`, Standardname `engineers` – einfach `Enter` drücken zum Überspringen).

#### Variante B: Direkt per CLI-Argumenten übergeben

```bash
./setup.sh \
  --admin-user admin \
  --admin-password "<DEIN_ADMIN_PASSWORT>" \
  --volunteers-user volunteers \
  --volunteers-password "<OPTIONAL_VOLUNTEERS_PASSWORT>" \
  --engineers-user engineers \
  --engineers-password "<OPTIONAL_ENGINEERS_PASSWORT>" \
  --port 8080
```

Alternativ kann auch das Python-Datenbankskript jederzeit direkt (interaktiv oder mit Parametern) aufgerufen werden:

```bash
python3 scripts/setup_db.py --db-path data/db/fll_vision.sqlite3
```

--------------------------------------------------------------------------------

## 4\. Production Build vs. Development Build starten

Die gesamte Applikation läuft in **einem einzigen Docker-Container**.

### A. Production Build (`docker-compose.yml`)

Für den Turnier- und Produktivbetrieb auf dem NVIDIA DGX Spark (`restart: unless-stopped`, `LOG_LEVEL=INFO`, automatische USB-Kamera-Durchreichung):

```bash
# Container bauen und im Hintergrund starten
docker compose up --build -d

# Live-Logs verfolgen
docker compose logs -f

# Healthcheck prüfen
curl http://localhost:8080/api/health
```

Oder als einzelner `docker build` & `docker run` Befehl ohne Compose:

```bash
docker build -t fll-vision:dgx-spark-production .

docker run -d \
  --name fll-vision \
  --restart unless-stopped \
  --gpus all \
  --ipc=host \
  --device-cgroup-rule="c 81:* rmw" \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8080:8080 \
  --env-file .env \
  -v "$(pwd)/config:/workspace/config" \
  -v "$(pwd)/models:/workspace/models" \
  -v "$(pwd)/data:/workspace/data" \
  -v /dev:/dev \
  -v /sys/class/video4linux:/sys/class/video4linux:ro \
  fll-vision:dgx-spark-production
```

### B. Development Build (`docker-compose.dev.yml`)

Für die Entwicklung und schnelle Anpassungen an Python-Code (`./src`), HTML/CSS-Templates (`./frontend`), Skripten (`./scripts`) oder Configs (`./config`). Alle Quellordner werden live vom Host in den Container gemountet und das Logging läuft auf `LOG_LEVEL=DEBUG`:

```bash
# Dev-Container im Vordergrund mit Debug-Logs starten
docker compose -f docker-compose.dev.yml up --build

# Oder im Hintergrund starten
docker compose -f docker-compose.dev.yml up --build -d
```

--------------------------------------------------------------------------------

## 5\. Standard-Kamera im Produktivbetrieb & Wechsel auf RTSP / MP4 in `/control`

### Standard in Produktion: Angeschlossene USB-Kamera (`/dev/video0`)

- In `config/config.yaml` ist standardmäßig **`video.source: "/dev/video0"`** (die per USB angeschlossene Kamera) eingestellt.
- Durch `device_cgroup_rules: ["c 81:* rmw"]` und die Mounts `/dev:/dev` sowie `/sys/class/video4linux:/sys/class/video4linux:ro` in `docker-compose.yml` erkennt der Container jede angeschlossene USB-Kamera automatisch (auch bei Hot-Plugging).
- Falls beim Container-Start noch keine USB-Kamera angesteckt ist, nutzt der Streamer vorübergehend das lokale Fallback-Video (`data/videos/match_input.mp4`), damit der Webserver sofort erreichbar ist.

### Wechsel auf RTSP-Stream, andere USB-Kamera oder `.mp4` in `/control` (und `/configuration`)

Sowohl unter **Volunteer Control (`/control`)** als auch unter **System Configuration (`/configuration`)** befindet sich die Karte **„🎥 Video-Quelle (Standard: USB-Kamera / RTSP / MP4)"**:

1. Klicke auf **„🔄 Kameras & Quellen scannen"**: Alle am System angeschlossenen USB-Kameras (inkl. Hardware-Name & Auflösung) sowie lokale `.mp4`-Dateien werden automatisch im Dropdown aufgelistet.
2. Alternativ kannst du im Eingabefeld einen **RTSP-Link** (z. B. `rtsp://192.168.1.50:554/stream`) oder einen `.mp4`-Pfad eingeben.
3. Klicke auf **„🔍 Bild-Check testen"** oder **„💾 Prüfen & Live umschalten"**:

  - Das Backend prüft **zwingend vor dem Speichern**, ob die angegebene Kamera / der RTSP-Stream / die `.mp4`-Datei ein gültiges Bildsignal liefert (`probe_video_source`).
  - Nur wenn ein gültiger Frame empfangen wird, zeigt die UI das Live-Vorschaubild samt Auflösung an, schaltet den laufenden Stream ohne Container-Neustart um (`Hot-Swap`) und speichert die Quelle in `config/config.yaml`.

--------------------------------------------------------------------------------

## 6\. Bedeutung der Ordner unter `data/` & Modell-Training mit Videos

### Übersicht der `data/` Unterordner

Ordner                      | Zweck                                                                                                                                                                                                                                                                | In Git enthalten?
:-------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------
**`data/training_videos/`** | **Nur fürs Training:** Hier legst du deine Trainingsvideos ab (egal ob 1 langes Video oder mehrere kurze Clips, Dateinamen beliebig). `./run_all.sh` liest diesen Ordner automatisch aus.                                                                            | Nur `.gitkeep` (Videos durch `.gitignore` ausgeschlossen)
**`data/videos/`**          | Enthält optionale `.mp4`-Videos (oder das Fallback-Video `match_input.mp4`), die bei Bedarf in `/control` ausgewählt werden können.                                                                                                                                  | Nur `.gitkeep`
**`data/dataset/`**         | Wird beim Start von `./run_all.sh` vollautomatisch generiert: enthält die auf die aktive Tischhälfte maskierten Bilder (`images/train`, `images/val`) und die automatisch erzeugten YOLO-Labels (`labels/train`, `labels/val`).                                      | Nein (wird automatisch generiert)
**`data/outputs/`**         | Ausgaben aus dem Live-Betrieb: rotierendes System-Log (`fll_vision.log`), exportierte Heatmaps (`match_heatmap_minimap.png`), Lauf-Tabellen (`runs_summary.csv`), Telemetrie (`match_telemetry.json`) und annotierte Match-Aufzeichnungen (`fll_tracked_match.mp4`). | Nur `.gitkeep`
**`data/db/`**              | Enthält die persistente SQLite-Datenbank (`fll_vision.sqlite3` im WAL-Modus) mit den gehashten Benutzern (`admin`, `volunteers`, `engineers`), Sessions, FLL-Teams und gespeicherten Runs.                                                                           | Nur `.gitkeep`

### 1-Klick-Training mit deinen Videos (`./run_all.sh`)

1. Lege deine Trainingsvideos in `data/training_videos/` ab.
2. Wähle unter `/configuration` (Tab 1) die aktive Tischhälfte (`left_half` oder `right_half`), damit die ungenutzte Tischhälfte (Hände, Flaschen etc.) und der Außenbereich zu 100 % schwarz ausmaskiert werden.
3. Starte das Training und den TensorRT-Export:

  ```bash
  ./run_all.sh
  ```

   Das Skript extrahiert die maskierten Frames, annotiert `robot` und `hand` automatisch via `YOLO-World` + Bewegungsanalyse, trainiert `YOLO11s` auf der DGX Spark GPU und kompiliert das fertige Modell als **FP16 TensorRT Engine (`models/fll_yolo11s.engine`)**.

--------------------------------------------------------------------------------

## 7\. Git / GitHub Upload & Wartung

Das Projekt enthält `.gitignore`, `.dockerignore`, `.gitattributes` und `.gitkeep`-Dateien für alle benötigten leeren Ordner (`data/training_videos/`, `data/videos/`, `data/outputs/`, `data/db/`, `models/`). Alle Rechte bleiben geschützt (keine Open-Source-Lizenzdatei enthalten).

```bash
cd fll-vision
git init
git branch -M main
git add .
git commit -m "Initial release of fll-vision"
git remote add origin https://github.com/<dein-user>/fll-vision.git
git push -u origin main
```

### Troubleshooting & Systemtest

- **Automatisierten End-to-End-Systemtest ausführen:**

  ```bash
  docker compose exec fll-vision python3 scripts/verify_system.py
  ```

- **Passwörter ändern oder neue User anlegen:** Entweder direkt im Browser unter `/admin` (Login als `admin`) oder per CLI über `./setup.sh`.
