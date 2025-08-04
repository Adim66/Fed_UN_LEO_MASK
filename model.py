import torch
import torch.nn as nn

def get_parameters():
    return [
        [0.0] * 10  # Un vecteur de poids factice
    ]

def set_parameters(parameters):
    pass  # Rien à faire ici pour un modèle factice

def train_model(parameters, epochs):
    # Simule un apprentissage local
    new_parameters = [[p + 0.1 for p in parameters[0]]]  # Juste une incrémentation
    return new_parameters, 10  # 10 exemples *
def evaluation(model, dataloader):
    model.eval()  # mode évaluation (désactive dropout, batchnorm etc.)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            outputs = model(x)
            loss = loss_fn(outputs, y)
            total_loss += loss.item() * x.size(0)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy
