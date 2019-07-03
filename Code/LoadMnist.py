import torch
from torchvision import datasets,transforms

def get_MNIST_data(): 
    totensor = transforms.ToTensor()
    data_train = datasets.MNIST(root = "./data/",train = True,transform = totensor,download = True)
    data_test = datasets.MNIST(root="./data/",transform = totensor,train = False) 
    return data_train,data_test

def get_load_data(data_train,data_test,BSize):
    data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size = BSize, shuffle = True)
    data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size = BSize,shuffle = True)
    return data_loader_train,data_loader_test