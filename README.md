# Federated SubNet (FedSN) - Simple Implementation

## Description

This project provides a simple implementation of **Federated SubNet (FedSN)**, a federated learning method where each client trains a **sub-part (substructure)** of the global model. The goal is to reduce the computational load on the client side while maintaining good overall performance.

---

## Workflow – Masking and Freezing

The core idea of this approach relies on **masking** and **freezing** layers of the neural network:

1. **Receive the full global model:**  
   Each client receives the complete global model (all parameters).

2. **Define a mask on layers:**  
   For each client, a list of `sub_indices` indicates which layers or substructures they should train. For example, a client may only train layers 2 to 4.

3. **Freeze the layers outside the subset:**  
   All layers **outside the targeted subset** are frozen (`requires_grad=False`), preventing their weights from being updated during local training.

4. **Local training:**  
   The client fine-tunes only the active (unfrozen) layers.

5. **Return updated parameters:**  
   After training, the client sends back only the updated substructure parameters.

6. **Server-side aggregation:**  
   The server reconstructs the global model by aggregating updated substructures from all clients.

---

## Advantages of this Method

- **Reduced client computation:** each client trains only part of the model, lowering local resource consumption (CPU/GPU, memory).  
- **Flexibility:** the `sub_indices` strategy can be adapted based on each client’s computational budget.  
- **Simplicity of integration:** maintains a fixed-size global model, avoiding complexities of managing independent partial models.

---

## Code Structure

- **client.py:** handles receiving the model, applying masking/freezing, training the substructure, and returning updates.  
- **server.py:** orchestrates federation, distributes substructures based on client budgets or clustering, and aggregates received updates.  
- **utils.py:** utility functions for parameter manipulation, masking, and model reconstruction.

---

## Usage Instructions

1. Start the federated server:  
   ```bash
   python server.py
