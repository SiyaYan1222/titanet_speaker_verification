# TitaNet Speaker Verification - Local Deployment

This repo is a local-ready version of the Hugging Face Space:

- Space: `nithinraok/titanet-speaker-verification`
- Model: `nvidia/speakerverification_en_titanet_large`
- UI: Gradio
- Backend: NVIDIA NeMo + PyTorch

The app compares two speech samples and returns whether they are likely from the same speaker. It supports two input modes:

1. **Microphone** - record two samples in the browser.
2. **Upload File** - upload two audio files.

> Recommended environment: **Linux or Windows WSL2 Ubuntu**. Native Windows can work, but NeMo/audio dependencies are usually easier in Linux/WSL2.

---

## Repository Structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── .gitkeep
└── scripts/
    ├── setup_linux_wsl.sh
    ├── run_linux_wsl.sh
    ├── setup_windows_native.ps1
    └── run_windows_native.ps1
```

---

## 1. Enter WSL2 from Windows

From **Windows Terminal**, **PowerShell**, or **CMD**:

```powershell
wsl
```

If you have more than one WSL distro:

```powershell
wsl -l -v
```

Then enter the specific distro, for example:

```powershell
wsl -d Ubuntu
```

or:

```powershell
wsl -d Ubuntu-22.04
```

Exit WSL:

```bash
exit
```

---

## 2. Check NVIDIA GPU in WSL2

Inside WSL2 Ubuntu:

```bash
nvidia-smi
```

Then check PyTorch CUDA availability after installation:

```bash
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
PY
```

The app also runs on CPU, but GPU is faster.

---

## 3. Linux / WSL2 Setup

From your workspace folder:

```bash
cd ~/Workspace
```

Clone your repo after you push it:

```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_FOLDER>
```

Or, if you are already inside the folder, run:

```bash
chmod +x scripts/setup_linux_wsl.sh scripts/run_linux_wsl.sh
./scripts/setup_linux_wsl.sh
```

Manual setup equivalent:

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv ffmpeg libsndfile1 git git-lfs

git lfs install

python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 4. Run on Linux / WSL2

```bash
source venv/bin/activate
python app.py
```

Or:

```bash
./scripts/run_linux_wsl.sh
```

Open in the Windows browser:

```text
http://127.0.0.1:7860
```

Usually this also works from Windows:

```text
http://localhost:7860
```

---

## 5. Run with LAN Access

By default, the app starts on `127.0.0.1:7860`.

To make it available to other devices on the same network:

```bash
source venv/bin/activate
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 python app.py
```

Then access it from another device using:

```text
http://<YOUR_MACHINE_IP>:7860
```

Find your WSL/Linux IP:

```bash
hostname -I
```

---

## 6. Native Windows Setup

WSL2 is recommended. Use this only if you want to try native Windows.

Install first:

- Python 3.10
- Git
- Git LFS
- FFmpeg
- Microsoft C++ Build Tools

From PowerShell inside the repo folder:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Or run:

```powershell
.\scripts\setup_windows_native.ps1
```

Run the app:

```powershell
.\venv\Scripts\Activate.ps1
python app.py
```

Open:

```text
http://127.0.0.1:7860
```

---

## 7. Requirements

`requirements.txt`:

```txt
gradio>=4.44.1
git+https://github.com/NVIDIA/NeMo.git@r2.0.0#egg=nemo_toolkit[asr]
```

The model is downloaded automatically on first run:

```python
model_name = "nvidia/speakerverification_en_titanet_large"
```

---

## 8. Push to GitHub or GitLab

Initialise git if this is a new local repo:

```bash
git init
git add .
git commit -m "Initial local TitaNet speaker verification app"
```

Add your remote:

```bash
git remote add origin <YOUR_REPO_URL>
```

Push:

```bash
git branch -M main
git push -u origin main
```

If the remote already exists:

```bash
git remote -v
git remote set-url origin <YOUR_REPO_URL>
```

If your GitLab default branch is protected and blocks the initial commit, ask a Maintainer/Owner to push the initial commit or temporarily adjust branch protection.

---

## 9. Troubleshooting

### `ModuleNotFoundError: No module named 'gradio'`

Install Gradio:

```bash
pip install gradio
```

### `nvidia-smi` does not work in WSL2

Update/install the NVIDIA driver on the Windows side, then restart WSL:

```powershell
wsl --shutdown
wsl
```

### Microphone does not work

Use the **Upload File** tab first. In WSL2, the browser runs in Windows and sends the recorded file to the WSL backend. Upload mode is usually more stable for testing.

### First run is slow

The TitaNet model is downloaded during the first launch. Later launches should be faster.

### CPU mode only

The app can run on CPU, but comparison may be slower. Check:

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
PY
```

---

## 10. Notes for Demo Use

For a quick lab/demo workflow:

1. Start app in WSL2.
2. Open `http://localhost:7860` in Windows browser.
3. Use **Upload File** first to verify the pipeline.
4. Then test **Microphone** mode.
5. For more stable speaker verification, use clean speech samples of similar duration and avoid background noise.
