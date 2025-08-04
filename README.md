# Federated SubNet (FedSN) - Implémentation Simple

## Description

Ce projet propose une implémentation simple de **Federated SubNet (FedSN)**, une méthode de *federated learning* où chaque client entraîne une sous-partie (substructure) du modèle global. L’objectif est de réduire la charge computationnelle côté client tout en conservant une bonne performance globale.

---

## Workflow - Masking et Freezing

La clé de cette approche repose sur un mécanisme de **masking** et **freezing** des couches du réseau neuronal :

1. **Réception du modèle global complet** :  
   Chaque client reçoit le modèle global complet (tous les paramètres).

2. **Définition d’un masque sur les couches** :  
   Pour chaque client, une liste `sub_indices` indique les couches ou sous-structures qu’il doit entraîner. Par exemple, un client peut n’entraîner que les couches 2 à 4.

3. **Masquage des couches non concernées (Freezing)** :  
   Toutes les couches **hors du sous-ensemble ciblé** sont mises en mode *gelé* (`requires_grad=False`), ce qui empêche la mise à jour de leurs poids pendant l’entraînement local.

4. **Entraînement local** :  
   Le client effectue le fine-tuning uniquement sur les couches actives (non gelées).

5. **Retour des paramètres mis à jour** :  
   Après l’entraînement, le client renvoie au serveur uniquement les sous-parties mises à jour du modèle.

6. **Agrégation côté serveur** :  
   Le serveur reconstitue le modèle global en agrégeant les sous-structures mises à jour provenant de tous les clients.

---

## Avantages de cette méthode

- **Allègement du calcul client** : chaque client n’entraîne qu’une partie du modèle, ce qui réduit la consommation de ressources locales (CPU/GPU, mémoire).  
- **Flexibilité** : la stratégie `sub_indices` peut être adaptée selon la capacité ou budget computationnel de chaque client.  
- **Simplicité d’intégration** : on conserve un modèle global de taille fixe, évitant les complexités liées à la gestion de modèles partiels indépendants.

---

## Structure du code

- **client.py** : contient la logique pour recevoir le modèle, appliquer le masking/freezing, entraîner la sous-structure, et renvoyer les mises à jour.  
- **serveur.py** : orchestre la fédération, distribue les sous-structures selon des critères (budget client, clustering), et agrège les mises à jour reçues.  
- **utils.py** : fonctions utilitaires pour manipulation des paramètres, masquage, et reconstruction du modèle.

---

## Instructions d’utilisation

1. Lancer le serveur fédéré :  
   ```bash
   python serveur.py
