import random
import time
import flwr as fl
import numpy as np
import torch.nn as nn
from typing import List, Optional, Tuple, Dict
from collections import defaultdict
import os

from unlearning.forget_vector import compute_forget_vector
from unlearning.influence import influences


# Fonction simple pour clusteriser sans sklearn en 3 groupes
def cluster_clients_by_budget(budgets: List[float], all_clients: List[fl.server.client_proxy.ClientProxy]) -> Dict[int, List[fl.server.client_proxy.ClientProxy]]:
    clusters = {0: [], 1: [], 2: []}
    for client, b in zip(all_clients, budgets):
        if b < 0.33:
            clusters[0].append(client)
        elif b < 0.66:
            clusters[1].append(client)
        else:
            clusters[2].append(client)
    return clusters
def clusters_to_save(budgets: List[float], all_clients: List[fl.server.client_proxy.ClientProxy]) -> Dict[int, List[fl.server.client_proxy.ClientProxy]]:
    clusters = {0: [], 1: [], 2: []}
    for client, b in zip(all_clients, budgets):
        if b < 0.33:
            clusters[0].append((client,b))
        elif b < 0.66:
            clusters[1].append((client,b))
        else:
            clusters[2].append((client,b))
    return clusters

# Exemple de paramètres initiaux (liste de numpy arrays)
def get_initial_parameters() -> fl.common.Parameters:
    model = SimpleModel()
    # Convertir tous les poids en ndarrays
    weights = [param.detach().numpy() for param in model.state_dict().values()]
    return fl.common.ndarrays_to_parameters(weights)



def split_model_into_substructures(ndarrays: List[np.ndarray], num_parts: int):
    part_size = len(ndarrays) // num_parts
    substructures = []

    for i in range(num_parts):
        start = i * part_size
        end = len(ndarrays) if i == num_parts - 1 else (i + 1) * part_size
        substructures.append(ndarrays[start:end])
    return substructures



NUM_SUBSTRUCTURES = 10 
UNLEARN_ROUND=5
unlearn_threshold = 0.7  # Seuil pour considérer une influence comme faible
CLIENT_TO_UNLEARN= ""  # Client à désapprendre""
SUB_WEEK=[] # Nombre de sous-structures à créer
class SimpleModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(784, 256),  # Layer 0
            nn.ReLU(),            # Layer 1
            nn.Linear(256, 128),  # Layer 2
            nn.ReLU(),            # Layer 3
            nn.Linear(128, 10)    # Layer 4
        )
class CustomStepStrategy(fl.server.strategy.FedAvg):

    def configure_fit(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        all_clients = list(client_manager.all().values())
        if not all_clients:
            print("[Warning] Aucun client disponible pour ce round.")
            return []
        global CLIENT_TO_UNLEARN, SUB_WEEK, unlearn_threshold
        # Générer budgets aléatoires [0,1]
        budgets = [random.random() for _ in all_clients]
        print(f"[Round {server_round}] Budgets générés : {budgets}")
        # Clustering
        clusters = cluster_clients_by_budget(budgets, all_clients)
        #hadhouma clusters besh n7otouhom fi el file for unlearning 
        log_clusters = clusters_to_save(budgets, all_clients)
        #print(f"Taille des paramètres : {len(parameters)}")
        print(f"\n[Round {server_round}] Budgets clients :")
        for i, (client, budget) in enumerate(zip(all_clients, budgets)):
            print(f"  Client {i} - id={client.cid} budget={budget:.2f}")

        print("Clusters:")
        for cid, cluster in clusters.items():
            print(f"  Cluster {cid}: {len(cluster)} clients")

        # Regrouper tous les clients non vides pour entraînement (ou sélection par cluster ici)
        selected_clients = []
        for cluster_clients in clusters.values():
            if cluster_clients:
                selected_clients.extend(cluster_clients)

        if not selected_clients:
            print("[Warning] Aucun client sélectionné après clustering.")
            return []
        # fy clusters 3anna tous les clients per cluster 
        # Préparer FitIns lenna besh ysir el identifications de ai selon le budget , on prend chaque cluster on selectionne w pour cluster i , on devisie w en des aj puis en les envioue selon fitns(params**)
        fit_instructions: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]] = []
        print("paramssssssssssssssssssssssssssssssssssssssssssssssssssssssss")
        

        weights = fl.common.parameters_to_ndarrays(parameters)
        print(f"[DEBUG] Round weights: {[w.shape for w in weights]}")


        print(f"client id {all_clients[0].cid}")
        print("budgets", len(budgets))
        substructures = split_model_into_substructures(weights, NUM_SUBSTRUCTURES)
        print(f"[DEBUG] Nombre total de sous-structures générées : {len(substructures)}")
        print(f"[DEBUG] Exemple de forme des tenseurs dans le bloc 0 : {[t.shape for t in substructures[0]]}")
        if (server_round == UNLEARN_ROUND):
                    print("Unlearning round reached, computing forget vector...")
                    influential_indices, _ = compute_forget_vector(all_clients[0].cid,"logs/cluster_log.txt", NUM_SUBSTRUCTURES)
                    print(f"Influential indices for client {all_clients[0].cid}: {influential_indices}")
                    fki = influences(all_clients[0].cid, "logs/cluster_log.txt", influential_indices)
                    print(f"Influence vector for client {all_clients[0].cid}: {fki}")
                    weakly_influenced = [i for i in influential_indices if fki[i] < unlearn_threshold]
                    SUB_WEEK=weakly_influenced
                    CLIENT_TO_UNLEARN = all_clients[0].cid
                    print(f"Client {CLIENT_TO_UNLEARN} will unlearn substructures: {SUB_WEEK}")
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        for cluster_id, cluster_clients in clusters.items():
             if not cluster_clients:
                continue

             num_clients = len(cluster_clients)
             client_shares = []
             print("***************************************************")
            # Calcul du nombre de sous-structures que chaque client peut gérer
             for i, client in enumerate(cluster_clients):
                 budget = budgets[all_clients.index(client)]
                 share = max(1, int((1 - budget) * NUM_SUBSTRUCTURES))  # au moins 1 bloc
                 client_shares.append((client, share))

        # Répartition des blocs entre clients (sans chevauchement)
             current_idx = 0
             for client, share in client_shares:
    
                     
                 if client.cid == CLIENT_TO_UNLEARN and UNLEARN_ROUND <server_round <=9:
                     
                     continue 
                 
                 if current_idx >= NUM_SUBSTRUCTURES:
                   break  # plus rien à attribuer

                 end_idx = min(current_idx + share, NUM_SUBSTRUCTURES)
                 config = {
   "cluster_id": cluster_id,
    "sub_indices": ",".join(map(str, list(range(current_idx, end_idx))))

}               
                 print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")
                 print(config) #lenna ba3ed el round 5 besh n7otou les sub indices weakly influenced llkol el clients
                 if server_round >=UNLEARN_ROUND :
                      config = {
    "cluster_id": cluster_id,
    "sub_indices": ",".join(str(n) for n in SUB_WEEK)

}                     
                      print("sub indices weakly influenced", SUB_WEEK)

                 if client.cid == CLIENT_TO_UNLEARN :
                                 print("RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR :")
                                 print("round", server_round)
                                 config = {
    "cluster_id": cluster_id,
    "sub_indices": ",".join(["0"])

}         
                 fit_instructions.append((client, fl.common.FitIns(parameters, config)))
                 for nodes in log_clusters.values():
                     for i in range(len(nodes)):
                            node_client, budget = nodes[i][0], nodes[i][1]
                            if node_client.cid == client.cid:
                # Remplacer le tuple (client, budget) par un triplet (client, budget, sub_indices)
                             nodes[i] = (node_client, budget, config["sub_indices"])
                             break  # une fois trouvé, inutile de continuer
                             
                 print("cluster log",log_clusters)     
                             
                             
                     
            # Fusionner les blocs attribués à ce client
                 
                # print(f"Client {client.cid} - Cluster {cluster_id} : {len(assigned_blocks)} blocs attribués")

                # print(config)
                 
                 current_idx = end_idx
        #print(f"[DEBUG] Nombre total de clients sélectionnés pour fit: {len(fit_instructions)}")
       
# Chemin du fichier de log (crée un dossier 'logs' si besoin)
        os.makedirs("logs", exist_ok=True)
        log_file_path = "logs/cluster_log.txt"

        with open(log_file_path, "a") as f:
          f.write(f"\n=== Round {server_round} ===\n")
          for cluster_id, nodes in log_clusters.items():
             f.write(f"Cluster {cluster_id}:\n")
             for entry in nodes:
                client_obj, budget = entry[0], entry[1]
                sub_indices = entry[2] if len(entry) > 2 else "N/A"
                f.write(f"  Client {client_obj.cid}, Budget: {budget:.2f}, Sub-Indices: {sub_indices}\n")

        return fit_instructions
   # The following block is commented out because it is unused and causes indentation errors.
   # def configure_fit(
   #     self,
   #     server_round: int,
   #     parameters: fl.common.Parameters,
   #     client_manager: fl.server.client_manager.ClientManager,
   # ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
   #
   #     # Récupérer tous les clients connectés
   #     all_clients = list(client_manager.all().values())
   #
   #     # Configuration simple pour chaque client
   #     fit_config = {
   #         "round": server_round,
   #         "epochs": 1,
   #     }
   #
   #     # Créer une liste d'instructions de fit pour tous les clients
   #     fit_instructions: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]] = []
   #     for client in all_clients:
   #         ins = fl.common.FitIns(parameters, fit_config)
   #         fit_instructions.append((client, ins))
   #
   #     return fit_instructions
# Exemple : forme des poids du modèle global (PyTorch/Keras -> à adapter selon ton modèle réel)
    GLOBAL_MODEL_SHAPE_LIST = [
    (784, 128),
    (128,),
    (128, 64),
    (64,),
    (64, 10),
    (10,)
]


    def aggregate_fit(self, server_round, results, failures):
        print("d5alnaaaaaaaaaaaaaaaaaaaa ll agggggggg")
        if not results:
            return None, {}

        clusters_results = {}
        print(f"Clients reçus : {len(results)}")

    # Regroupement par cluster
        for _, fit_res in results:
            sub_indices = fit_res.metrics.get("sub_indices")
            cluster_id = fit_res.metrics.get("cluster_id")
            weights = fl.common.parameters_to_ndarrays(fit_res.parameters)
            
            if cluster_id not in clusters_results:
             clusters_results[cluster_id] = []
            clusters_results[cluster_id].append((weights, fit_res.num_examples, sub_indices))

    # Agrégation par cluster (moyenne pondérée)
        aggregated_clusters = {}
        cluster_sizes = {}

        for cluster_id, cluster_data in clusters_results.items():
        # Initialiser la somme pondérée des poids du cluster
            weighted_sum = [np.zeros_like(w) for w in cluster_data[0][0]]
            total_examples = 0

            for weights, num_examples, _ in cluster_data:
                for i, w in enumerate(weights):
                 weighted_sum[i] += w * num_examples
                 total_examples += num_examples

        # Calcul de la moyenne pondérée
        averaged_weights = [w / total_examples for w in weighted_sum]
        aggregated_clusters[cluster_id] = averaged_weights
        cluster_sizes[cluster_id] = total_examples

    # Agrégation globale : moyenne pondérée des clusters selon leur taille
        total_examples_global = sum(cluster_sizes.values())
        global_weight_sum = [np.zeros_like(w) for w in aggregated_clusters[next(iter(aggregated_clusters))]]

        for cluster_id, cluster_weights in aggregated_clusters.items():
           cluster_weight = cluster_sizes[cluster_id] / total_examples_global
           for i, w in enumerate(cluster_weights):
             global_weight_sum[i] += w * cluster_weight

        aggregated_parameters = fl.common.ndarrays_to_parameters(global_weight_sum)
        return aggregated_parameters, {}
    
    def aggregate_evaluate(self, server_round, results, failures):
      if not results:
          return None, {}

      total_examples = 0
      weighted_loss_sum = 0.0
      weighted_accuracy_sum = 0.0
      print(f"actvated evaluation: {len(results)}")

      for id, evaluate_res in results:
        num_examples = evaluate_res.num_examples
        metrics = evaluate_res.metrics

        if id.cid == CLIENT_TO_UNLEARN:
                with open("logs/first_client_eval.txt", "a") as f:
                    f.write(f"Round {server_round} - Loss: {evaluate_res.loss:.4f}, "
                            f"Accuracy: {metrics.get('accuracy',0):.4f}\n")
        loss = evaluate_res.loss
        accuracy = metrics.get("accuracy", 0.0)

        weighted_loss_sum += loss * num_examples
        weighted_accuracy_sum += accuracy * num_examples
        total_examples += num_examples

      avg_loss = weighted_loss_sum / total_examples
      avg_accuracy = weighted_accuracy_sum / total_examples

    # Tu peux retourner la loss globale et les métriques agrégées
      return avg_loss, {"accuracy": avg_accuracy}




    

    

   
if __name__ == "__main__":
    initial_parameters = get_initial_parameters()
    weights = fl.common.parameters_to_ndarrays(initial_parameters)  
    print(f"Taille des paramètres : {len(weights)}")
    # 🔸 Afficher un exemple de poids initiaux
    


    strategy = CustomStepStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=2,
        initial_parameters=initial_parameters,
    )
        
  
    print("Démarrage du serveur Flower avec stratégie personnalisée...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )
