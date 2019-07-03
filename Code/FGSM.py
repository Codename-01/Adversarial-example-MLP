import torch

def FGSM(X,Y,model,cost,epsilon):
    X.requires_grad = True
    outputs = model(X)
    model.zero_grad()
    loss = cost(outputs,Y)
    loss.backward()
    
    data_grad = X.grad.data
    perturbed_data = epsilon*data_grad.sign()+X
    return perturbed_data