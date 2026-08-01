# PlantAI Backend — Setup Guide

## 1. Install Python 3.12

```bash
# Update packages
sudo apt update
sudo apt install -y software-properties-common

# Add the deadsnakes repository
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# Install Python 3.12
sudo apt install -y python3.12 python3.12-venv python3.12-dev

# Make 'python' point to Python 3.12
sudo ln -sf /usr/bin/python3.12 /usr/bin/python

# Install pip for Python 3.12
python -m ensurepip --upgrade

# Upgrade pip and packaging tools
python -m pip install --upgrade pip setuptools wheel

# Verify
python --version        # → Python 3.12.x
python -m pip --version
```

---

## 2. Clone & Set Up the Project

```bash
# Navigate to the project directory
cd /opt/plantai/backend

# Create a virtual environment
python -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 3. Configure Environment Variables

```bash
# Copy the example env file
cp .env.example .env
```

Edit `.env` as needed. Key variables:

| Variable                | Default                              | Description                        |
|-------------------------|--------------------------------------|------------------------------------|
| `MODEL_PATH`            | `artifacts/model_b_combined.keras`   | Path to the Keras model file       |
| `CLASS_NAMES_PATH`      | `artifacts/class_names.json`         | Path to class names JSON           |
| `BACKBONE`              | `efficientnetb0`                     | Must match training notebook       |
| `IMG_SIZE`              | `224`                                | Input image size                   |
| `USE_TTA`               | `true`                               | Enable test-time augmentation      |
| `CONFIDENCE_THRESHOLD`  | `0.6`                                | Min confidence to return a result  |
| `TOP_K`                 | `3`                                  | Number of top predictions          |
| `MAX_UPLOAD_MB`         | `10`                                 | Max upload file size               |
| `HOST`                  | `0.0.0.0`                            | Server bind address                |
| `PORT`                  | `5001`                               | Server port                        |

---

## 4. Run Locally (Development)

For quick local testing you can use Flask's built-in dev server:

```bash
source venv/bin/activate
python wsgi.py
```

> **Note:** This starts a single-threaded development server.
> You'll see `WARNING: This is a development server.` — that's expected and fine for local testing.

---

## 5. Deploy with PM2 + Gunicorn (Production)

### Why not `pm2 start wsgi.py`?

Running `wsgi.py` directly starts Flask's built-in dev server, which is single-threaded,
not hardened, and not meant for production traffic. Use **Gunicorn** instead.

### Install PM2 (if not already)

```bash
sudo npm install -g pm2
```

### Start the app

```bash
cd /opt/plantai/backend

pm2 start /opt/plantai/backend/venv/bin/gunicorn \
  --name plantai \
  --interpreter /opt/plantai/backend/venv/bin/python \
  --cwd /opt/plantai/backend \
  -- --bind 0.0.0.0:5001 --timeout 180 --workers 2 wsgi:app
```

> **`--workers 2`**: Adjust based on your server's CPU cores. A common rule of thumb is `(2 × CPU cores) + 1`.

### Useful PM2 commands

```bash
# View logs
pm2 logs plantai --lines 100

# Restart the app
pm2 restart plantai

# Stop the app
pm2 stop plantai

# Remove the app from PM2
pm2 delete plantai

# Check status
pm2 status
```

### Persist across reboots

> **Important:** Don't mix `sudo` and non-`sudo` PM2 commands — they operate on
> different PM2 daemons (your user's vs root's).

```bash
# Save current process list
pm2 save

# Generate startup script (prints a sudo command — copy-paste and run it)
pm2 startup
```

---

## 6. Caddy Reverse Proxy (HTTPS)

Caddy handles automatic HTTPS certificates and proxies traffic to the backend.

### Edit the Caddyfile

```bash
sudo vim /etc/caddy/Caddyfile
```

Add the following site blocks:

```caddyfile
api.example.com {
    reverse_proxy 127.0.0.1:5000
}

ai.example.com {
    reverse_proxy 127.0.0.1:5001
}
```

### Reload Caddy

```bash
sudo systemctl reload caddy
```

> Caddy will automatically provision and renew TLS certificates for both domains via Let's Encrypt.

### Verify Caddy status

```bash
sudo systemctl status caddy
```

---

## 7. Verify the Server

```bash
# Health check (local)
curl http://localhost:5001/health

# Health check (via domain)
curl https://ai.example.com/health

# List available classes
curl https://ai.example.com/classes

# Test a prediction
curl -X POST https://ai.example.com/predict \
  -F "image=@/path/to/leaf_image.jpg"
```

---

## 8. Update & Redeploy

```bash
cd /opt/plantai/backend
git pull

# Activate venv and install any new dependencies
source venv/bin/activate
pip install -r requirements.txt

# Restart the app
pm2 restart plantai
```

---

## 9. Troubleshooting

### TensorFlow startup warnings

On startup you may see logs like:

```
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
oneDNN custom operations are on…
This TensorFlow binary is optimized to use available CPU instructions…
```

These are **not errors** — they are TensorFlow's internal info messages:

| Message | Meaning |
|---|---|
| `absl::InitializeLog()` | TF logs to stderr before its own logging system initializes — cosmetic noise |
| `oneDNN custom operations are on` | TF uses Intel's oneDNN library for faster CPU math — a good thing |
| `optimized to use available CPU instructions (AVX2, AVX512F…)` | TF is leveraging your server's advanced CPU instructions — also good |
| `rebuild TensorFlow with the appropriate compiler flags` | Informational — the pip package is fine as-is |

> **Note:** You'll see two copies of each message because Gunicorn runs 2 workers, each loading TensorFlow independently.

#### Silence them

Set these environment variables before starting the process:

```bash
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
```

- `TF_CPP_MIN_LOG_LEVEL=3` — suppresses INFO, WARNING, and ERROR level C++ logs from TensorFlow
- `TF_ENABLE_ONEDNN_OPTS=0` — disables the oneDNN optimization startup notification

To make these permanent across server reboots, add them to `~/.bashrc`:

```bash
echo 'export TF_CPP_MIN_LOG_LEVEL=3' >> ~/.bashrc
echo 'export TF_ENABLE_ONEDNN_OPTS=0' >> ~/.bashrc
source ~/.bashrc
```

They are also defined in `ecosystem.config.js` and set as defaults in `plant_disease/config.py`.
