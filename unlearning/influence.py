import re

def influences(client_id: str, log_file: str, influential_indices: list, alpha=1.0, beta=1.0, gamma=0.0):
    with open(log_file, "r") as f:
        lines = f.readlines()

    # Structure: round -> cluster_id -> list of (client_id, budget, sub_indices)
    cluster_clients = {}
    current_round = None
    current_cluster_id = None

    for line in lines:
        round_match = re.match(r"=== Round (\d+) ===", line)
        if round_match:
            current_round = int(round_match.group(1))
            cluster_clients[current_round] = {}
            continue

        cluster_match = re.match(r"Cluster (\d+):", line)
        if cluster_match:
            current_cluster_id = int(cluster_match.group(1))
            cluster_clients[current_round][current_cluster_id] = []
            continue

        client_match = re.match(r"\s*Client ([a-f0-9]+), Budget: ([0-9.]+), Sub-Indices: (.+)", line)
        if client_match:
            cid, budget, subs = client_match.groups()
            subs = subs.strip()
            sub_indices = []
            if subs != "N/A":
                sub_indices = [int(s.strip()) for s in subs.split(",") if s.strip().isdigit()]
            cluster_clients[current_round][current_cluster_id].append((cid, float(budget), sub_indices))

    # Calcul influence pour chaque sous-structure influencée
    f_K_i = {}

    for i in influential_indices:
        sum_contrib = 0.0
        num_CKi = 0
        budget_K = None

        for rnd in cluster_clients:
            for cluster_id, clients in cluster_clients[rnd].items():
                for cid, budget, sub_indices in clients:
                    if cid == client_id and i in sub_indices:
                        if budget_K is None:
                            budget_K = budget
                        num_CKi += 1

                        # Trouver les clients l ∈ Lij
                        L_ij = [l for l in clients if i in l[2]]

                        # Calcul delta (l ∼ K): ici = 1 car même cluster
                        delta_sum = sum([alpha * 1 for l in L_ij])
                        size_Lij = len(L_ij)
                        if size_Lij == 0:
                            continue
                        term = 1 - (delta_sum / size_Lij)
                        sum_contrib += term

        if num_CKi == 0:
            f_K_i[i] = 0.0  # aucune influence détectée
        else:
            contrib_cluster = 0.0  # par défaut
            total = (1 / num_CKi) * sum_contrib + beta * (budget_K or 0.0) + gamma * contrib_cluster
            f_K_i[i] = total

    return f_K_i
