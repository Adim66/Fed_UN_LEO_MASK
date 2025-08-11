import numpy as np
import re
def compute_forget_vector(client_id: str, log_file: str, n_subs: int, tau: float = 0.1):
    s = np.zeros(n_subs)
    
    with open(log_file, "r") as f:
        lines = f.readlines()
    print(f"Processing log file: {log_file} for client {client_id}")    
    print("lines:", lines)
    for line in lines:
        # Ligne contenant le client voulu
        if f"Client {client_id}" in line:
            # Cherche les sous-indices
            match = re.search(r"Sub-Indices:\s*([0-9,]*)", line)
            if match:
                indices_str = match.group(1)
                if indices_str:  # Pas vide
                    indices = [int(i.strip()) for i in indices_str.split(",") if i.strip().isdigit()]
                    if indices:
                        weight = 1 / len(indices)
                        for i in indices:
                            if 0 <= i < n_subs:
                                s[i] += weight

    # Normalisation
    if np.sum(s) > 0:
        s_hat = s / np.sum(s)
    else:
        s_hat = s  # vecteur nul

    influential_indices = [i for i, v in enumerate(s_hat) if v > tau]

    return influential_indices, s_hat

