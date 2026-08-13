import os
import sys
import socket
import subprocess
import webbrowser
import time
import threading

def find_free_port(start_port=8000):
    port = start_port
    while port < 9000:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except socket.error:
                port += 1
    return start_port

def run():
    print("=" * 60)
    print("         OMNIML BRAIN - WEB SERVICE LAUNCHER")
    print("=" * 60)
    
    workspace_root = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(workspace_root, ".venv")
    
    # 1. Create Virtual Environment if not exists
    if not os.path.exists(venv_dir):
        print("[*] Creating Python virtual environment (this may take a moment)...")
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
        print("[+] Virtual environment created!")
    
    # Locate pip and uvicorn binaries
    if sys.platform == "win32":
        pip_path = os.path.join(venv_dir, "Scripts", "pip.exe")
        uvicorn_path = os.path.join(venv_dir, "Scripts", "uvicorn.exe")
    else:
        pip_path = os.path.join(venv_dir, "bin", "pip")
        uvicorn_path = os.path.join(venv_dir, "bin", "uvicorn")

    # 2. Check if requirements are installed
    requirements_file = os.path.join(workspace_root, "backend", "requirements.txt")
    print("[*] Ensuring dependencies are installed...")
    subprocess.run([pip_path, "install", "-r", requirements_file], check=True)
    print("[+] Dependencies verified!")

    # Find free port automatically to avoid "Address already in use" binding conflicts
    port = find_free_port(8000)
    server_url = f"http://127.0.0.1:{port}"

    # 3. Start Uvicorn Server
    print(f"[*] Starting FastAPI server on {server_url} ...")
    
    # Launch browser after a short delay to give the server time to bind to the port
    def launch_browser():
        time.sleep(1.5)
        print("[*] Opening web interface in browser...")
        webbrowser.open(server_url)

    # We run the uvicorn command
    browser_thread = threading.Thread(target=launch_browser)
    browser_thread.daemon = True
    browser_thread.start()

    try:
        subprocess.run([uvicorn_path, "backend.main:app", "--host", "127.0.0.1", "--port", str(port)], check=True)
    except KeyboardInterrupt:
        print("\n[+] Server stopped by user. Goodbye!")
    except Exception as e:
        print(f"\n[-] Server exited with error: {str(e)}")

if __name__ == "__main__":
    run()
