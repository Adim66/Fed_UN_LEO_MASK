import subprocess
import sys

num_clients = 4

processes = []
for i in range(num_clients):
    print(f"Lancement client #{i+1}")
    p = subprocess.Popen([
        sys.executable, "client.py",
        "--client_id", str(i),
        "--num_clients", str(num_clients)
    ])
    processes.append(p)

for p in processes:
    p.wait()
