# SEAM: start here

This folder is a ready-to-run copy of the SEAM project (Sign-language Emotion-Aware
Multimodal framework). You do not need to train anything. Two steps: set up once, then run.

## 1. Set up (once)

You need **Python 3.12** (3.11 also works). Get it from
<https://www.python.org/downloads/release/python-3120/>. On Windows, tick
**"Add python.exe to PATH"** in the installer. Python 3.13 is too new for the full install,
because the version of MediaPipe this project is pinned to has no build for it; it can stay
installed beside 3.12, and the website-only install (`--lite`) works on it.

| Your computer | Do this in the folder you extracted |
|---|---|
| **Windows** | double-click `setup_windows.bat` |
| **Linux / macOS** | open a terminal here and run `./setup_linux.sh` |

It creates a private environment in a `.venv` folder and installs the packages into it.
The full install downloads about 1 to 1.5 GB the first time. If you only want the website,
add `--lite` (`setup_windows.bat --lite` or `./setup_linux.sh --lite`): about 60 MB.

At the end it prints a checklist. Every line the website needs should say `[ok]`.

## 2. Run

| Your computer | Do this |
|---|---|
| **Windows** | double-click `run_demo.bat` |
| **Linux / macOS** | `./run_demo.sh` |

Your browser opens <http://127.0.0.1:8000/>. Click **Start camera** and allow the camera.
The first time, the page downloads about 15 MB of MediaPipe models, so you need internet
for that first load. Your video never leaves your computer.

To stop it, press `Ctrl+C` in the window that started it.

## What you can do with it

| Want to... | Do this |
|---|---|
| See the live demo | Run it and click **Start camera**. Raise your brows, furrow them, shake or nod your head, and watch the read-outs. |
| Use a video instead of the camera | On the page, click **Use a video file**. |
| Run the model on your own video, offline | `python scripts/analyse_video.py my_video.mp4` (full install; activate the environment first, see below). Add `--emotion` to also run the three trained emotion models, and `--out result.json` to save the result. |
| See the 3D avatar | The **3D avatar** link in the page's top bar. If this copy has no avatar clips, it shows rendered frames instead (see "What is not in this folder"). |
| Read what the project found | `README.md` is the full report. `docs/TEAM_GUIDE.md` is the short version, with the viva script and likely questions. |
| Run the tests | `python -m pytest -q` (full install). Tests that need data you do not have are skipped and say why. |

To use `python` commands yourself, activate the environment first:

- Windows: `.venv\Scripts\activate`
- Linux / macOS: `source .venv/bin/activate`

## What is in this folder

| Folder | What |
|---|---|
| `src/seam/` | The project's code |
| `scripts/` | Every experiment, plus `analyse_video.py` and `check_install.py` |
| `docs/` | The website (`docs/index.html`), its figures, and the team guide |
| `models/mediapipe/` | MediaPipe's face, hand and pose models, so Python-side tracking works offline |
| `artifacts/fer/`, `artifacts/export/` | Our three trained emotion models (PyTorch and ONNX) |
| `artifacts/audit`, `m3`, `m4`, `m5a` ... | The result files behind every number in the report |
| `paper/` | The experiment log and the claims ledger |

## What is not in this folder, and why

- **The sign-language videos and their annotations** (ASLLRP, EmoSign, WLASL). Their terms
  do not allow us to pass them on. You can run the website, analyse your own videos and read
  every result without them; you cannot re-run the experiments that need them.
- **The SMPL-X body model.** It is licensed to each person individually (free registration at
  <https://smpl-x.is.tue.mpg.de>). The animated avatar clips contain that model's mesh, so
  they are only included if whoever made this copy chose to and you hold the licence too.
  Without them the avatar section shows rendered frames.

## If something goes wrong

| What you see | What to do |
|---|---|
| "The full install needs Python 3.11 or 3.12" | Install Python 3.12 from python.org (it can sit beside a newer Python). On Windows tick "Add python.exe to PATH", then run setup again. Or use `--lite`. |
| Windows says the script is from an unknown publisher | Click **More info**, then **Run anyway**. It only creates the `.venv` folder here. |
| The install stops partway | Check the internet connection and run the setup file again; it continues where it stopped. |
| The page says the camera is blocked | Allow the camera for `127.0.0.1` in the browser's address bar, then reload. |
| "Address already in use" | Something else uses port 8000. Run `run_demo.bat 9000` or `./run_demo.sh 9000`. |
| The page loads but nothing tracks | The first load needs internet for the MediaPipe models. Reload once you are online. |

Run `python scripts/check_install.py` at any time to see what is and is not working.
