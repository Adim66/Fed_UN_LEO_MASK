from typing import List, Tuple, Dict
import flwr as fl
from sympy import evaluate  
import model 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import logging
import argparse


logging.basicConfig(level=logging.DEBUG)
# ========== Données d'entraînement factices ==========


def load_full_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalisation classique MNIST
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    print(f"Train dataset size: {len(train_dataset)}"
          f", Test dataset size: {len(test_dataset)}")
    return train_dataset, test_dataset

def load_data(train_dataset, client_id, num_clients, batch_size=1):
    # Calculer indices pour découper le dataset en partitions égales
    data_len = len(train_dataset)
    indices_per_client = data_len // num_clients
    start_idx = client_id * indices_per_client
    end_idx = start_idx + indices_per_client if client_id != num_clients - 1 else data_len

    client_indices = list(range(start_idx, end_idx))

    client_subset = Subset(train_dataset, client_indices)
    client_loader = DataLoader(client_subset, batch_size=batch_size, shuffle=True)
    print(f"Client id {client_id} a {len(client_subset)} exemples.")
    return client_loader


# ========== Utilitaires pour manipuler les poids ==========
def set_weights(model: nn.Module, weights: List[np.ndarray]):
    with torch.no_grad():
        for param, w in zip(model.parameters(), weights):
            param.data = torch.tensor(w, dtype=param.dtype)

def get_weights(model: nn.Module) -> List[np.ndarray]:
    return [param.detach().numpy() for param in model.parameters()]

# ========== Entraînement local ==========
def train(model, dataloader, epochs=1):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.08)
    loss_fn = nn.CrossEntropyLoss()
    i=0
    for _ in range(epochs):
        j=0
        for x, y in dataloader:
            x = x.view(x.size(0), -1)
            optimizer.zero_grad()
            output = model(x)
            j+=1
            
            loss = loss_fn(output, y)
            loss.backward()

            optimizer.step()
        print(f"-----------------Training batch-------------------- {j}")
        
def evaluation(model, dataloader):
    model.eval()  # mode évaluation (désactive dropout, batchnorm etc.)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.view(x.size(0), -1)  # n3awed el shape mte3 el input
            outputs = model(x)
            loss = loss_fn(outputs, y)
            total_loss += loss.item() * x.size(0)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

# ========== Client Flower ==========
class BudgetClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, num_clients: int) :
        self.client_id = client_id
        self.num_clients = num_clients
        print(f"Client ID: {self.client_id}, Total Clients: {self.num_clients}")
        self.train_dataset, self.test_dataset = load_full_mnist()
        self.local_data = load_data(self.train_dataset, self.client_id, self.num_clients)




    





    def build_full_model(self):
     model = nn.Sequential(
        nn.Linear(784, 128),   # poids (784,128) + biais (128,)
        nn.ReLU(),
        nn.Linear(128, 64),    # poids (128,64) + biais (64,)
        nn.ReLU(),
        nn.Linear(64, 10)      # poids (64,10) + biais (10,)
        # Pas d'activation finale car CrossEntropyLoss attend logits
     )
     return model

    def fit(self, parameters, config):
    # 1. Récupérer la sous-structure envoyée par le serveur (ex: "0,1,2") et la convertir en liste d'entiers
     sub_indices = list(map(int, config.get("sub_indices", "").split(",")))
     if sub_indices == []:
           return parameters, len(self.local_data), {
        "cluster_id": config.get("cluster_id", "unknown"),
        "sub_indices": config.get("sub_indices", ""),
        "id": self.client_id
    }
     print(f"params de types :{type(parameters)}")
   
    # 2. Construire un modèle complet avec l'architecture globale (toutes les couches)
     model = self.build_full_model()

    # 3. Appliquer les poids globaux reçus à ce modèle (chaque client reçoit le même modèle complet)
     set_weights(model, parameters)

    # 4. Geler (freeze) un certain pourcentage de couches cachées selon la sous-structure reçue
     self.freeze_layers_by_percentage(model, sub_indices)
     print(f"données de ce client est : {len(self.local_data.dataset)} exemples")
     train(model, self.local_data, epochs=config.get("epochs", 1))

  

    # 7. Extraire les poids entraînés du modèle après entraînement
     updated_parameters = get_weights(model)
     print("------------------------------------------------------------------------")
     print(f"updated est de type: {type(updated_parameters)}")
     print(f"type of elements : {type(updated_parameters[0])}")
     print(f"[DEBUG] Après entraînement - nombre de tensors: {len(updated_parameters)}")
    # 8. Retourner :
    #    - les nouveaux poids,
    #    - le nombre d'exemples utilisés,
    #    - les métadonnées utiles (cluster_id, sous-structure utilisée)
     return updated_parameters, len(self.local_data), {
        "cluster_id": config.get("cluster_id", "unknown"),
        "sub_indices": config.get("sub_indices", ""),
        "id": self.client_id
    }
    def evaluate(self, parameters, config):
    # 1. Construire le modèle complet
      model = self.build_full_model()
    
    # 2. Charger les poids globaux dans le modèle
      set_weights(model, parameters)
    
    # 3. Mettre le modèle en mode évaluation
      model.eval()
    
    # 4. Utiliser ta fonction d’évaluation locale
      loss, accuracy = evaluation(model, self.local_data)
    
    # 5. Retourner la loss, le nombre d’exemples, et les métriques (par ex. accuracy)
      return loss, len(self.local_data.dataset), {"accuracy": accuracy}

    def freeze_layers_by_percentage(self,model, sub_indices):
        total_subparts = 10
        percent = len(sub_indices) / total_subparts  # ex: 0.2 si 2 sur 10

    # Obtenir les couches (modules) du modèle sauf entrée et sortie (exemple)
        children = list(model.children())  # récupère la liste des sous-modules

    # Supposons que la 1ère couche est d'entrée, la dernière de sortie
        hidden_layers = children[1:-1]
        num_to_keep = int(len(hidden_layers) * percent)

    # On garde les premières num_to_keep couches cachées entraînables
        for i, layer in enumerate(children):
          if i == 0 or i == len(children) - 1:
              for param in layer.parameters():
                 param.requires_grad = True
          elif i - 1 < num_to_keep:
              for param in layer.parameters():
                 param.requires_grad = True
          else:
              for param in layer.parameters():
                 param.requires_grad = False
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--num_clients", type=int, default=4)
    args = parser.parse_args()

    client_instance = BudgetClient(client_id=args.client_id, num_clients=args.num_clients)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client_instance)