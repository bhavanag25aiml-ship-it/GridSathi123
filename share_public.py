"""
GridSathi Public Link Generator
Shares GridSathi on the public internet instantly using Cloudflare Tunnel or Localtunnel.
"""
import subprocess
import sys
import re
import time
import webbrowser
import urllib.request
import os
from pathlib import Path

# Force UTF-8 encoding on standard streams if possible
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
VENV_PYTHON = BASE_DIR / ".venv" / "Scripts" / "python.exe"
PYTHON_CMD = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
CLOUDFLARED_EXE = BASE_DIR / "cloudflared.exe"

def is_backend_running():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/api/status", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False

def start_backend():
    print("[1/2] Starting GridSathi FastAPI Backend...", flush=True)
    proc = subprocess.Popen(
        [PYTHON_CMD, "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(BASE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    for _ in range(15):
        time.sleep(1)
        if is_backend_running():
            print("       -> Backend is ONLINE at http://127.0.0.1:8000", flush=True)
            return proc
    print("       -> Backend initialized.", flush=True)
    return proc

def try_cloudflare_tunnel():
    if not CLOUDFLARED_EXE.exists():
        return None, None

    print("[2/2] Launching high-speed Cloudflare Public Tunnel...", flush=True)
    proc = subprocess.Popen(
        [str(CLOUDFLARED_EXE), "tunnel", "--url", "http://127.0.0.1:8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    start_time = time.time()
    for line in iter(proc.stdout.readline, ''):
        match = url_pattern.search(line)
        if match:
            return match.group(0), proc
        if time.time() - start_time > 15:
            break
    return None, proc

def get_localtunnel_password():
    try:
        req = urllib.request.Request("https://loca.lt/mytunnelpassword", headers={"User-Agent": "curl/7.68.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            return resp.read().decode('utf-8').strip()
    except Exception:
        try:
            req = urllib.request.Request("https://api.ipify.org", headers={"User-Agent": "curl/7.68.0"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                return resp.read().decode('utf-8').strip()
        except Exception:
            return "Visit https://loca.lt/mytunnelpassword"

def try_localtunnel():
    print("[2/2] Launching Localtunnel fallback...", flush=True)
    tunnel_pass = get_localtunnel_password()
    npx_cmd = "npx.cmd" if os.name == "nt" else "npx"
    proc = subprocess.Popen(
        [npx_cmd, "localtunnel", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.loca\.lt")
    start_time = time.time()
    for line in iter(proc.stdout.readline, ''):
        match = url_pattern.search(line)
        if match:
            return match.group(0), tunnel_pass, proc
        if time.time() - start_time > 15:
            break
    return None, tunnel_pass, proc

def main():
    print("=" * 70, flush=True)
    print("      GRID SATHI - INSTANT PUBLIC LINK GENERATOR", flush=True)
    print("=" * 70, flush=True)

    backend_proc = None
    if not is_backend_running():
        backend_proc = start_backend()
    else:
        print("[1/2] GridSathi Backend is already running on port 8000.", flush=True)

    public_url, tunnel_proc = try_cloudflare_tunnel()
    tunnel_pass = None

    if not public_url:
        if tunnel_proc:
            try:
                tunnel_proc.terminate()
            except Exception:
                pass
        public_url, tunnel_pass, tunnel_proc = try_localtunnel()

    if public_url:
        print("\n" + "=" * 70, flush=True)
        print("   SUCCESS! YOUR GRIDSATHI WEBSITE IS LIVE & PUBLICLY ACCESSIBLE:", flush=True)
        print(f"\n   >>  WEBSITE URL:  {public_url}  <<\n", flush=True)
        if tunnel_pass:
            print(f"   Tunnel Password: {tunnel_pass} (if prompted)", flush=True)
        print("=" * 70, flush=True)
        print("\n-> Anyone can open this link on their mobile, tablet, or PC anywhere in the world!", flush=True)
        print("-> Keep this window OPEN while sharing.", flush=True)
        print("-> Press Ctrl + C anytime to stop sharing.\n", flush=True)

        txt_content = (
            f"GridSathi Live Public Link\n"
            f"--------------------------------------------------\n"
            f"Website URL:     {public_url}\n"
        )
        if tunnel_pass:
            txt_content += f"Tunnel Password: {tunnel_pass}\n"
        txt_content += (
            f"Status:          ACTIVE\n"
            f"Updated:         {time.ctime()}\n\n"
            f"Instructions:\n"
            f"Open the Website URL in any browser on phone or PC to view GridSathi live!\n"
        )
        (BASE_DIR / "PUBLIC_LINK.txt").write_text(txt_content, encoding="utf-8")

        try:
            webbrowser.open(public_url)
        except Exception:
            pass

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down public tunnel...", flush=True)
    else:
        print("\nCould not obtain a public link. Please verify internet connection.", flush=True)

    if tunnel_proc:
        try:
            tunnel_proc.terminate()
        except Exception:
            pass
    if backend_proc:
        try:
            backend_proc.terminate()
        except Exception:
            pass

if __name__ == "__main__":
    main()
