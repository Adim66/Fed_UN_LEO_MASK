import torch.nn.functional as F

def gradient_inversion(model, data_K, optimizer):
    model.train()
    for x, y in data_K:
        optimizer.zero_grad()
        output = model(x)
        loss = -F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()

