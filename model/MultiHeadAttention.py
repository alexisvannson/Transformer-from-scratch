import torch
import torch.nn as nn
import multiprocessing as mp
    

X = torch.randn(5, 5)

print(X.shape)
matrixQ = torch.randn(5, 5)
matrixK = torch.randn(5, 5)
matrixV = torch.randn(5, 5)

Q = torch.matmul(X, matrixQ)
K = torch.matmul(X, matrixK)
V = torch.matmul(X, matrixV)

#print(Q)
#print(K)
#print(V)

class FFN(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, out_dim=512):
        super().__init__
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim)
        self.linear2 = nn.Linear(in_features=self.hidden_dim, out_features=out_dim)
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return x
        
        
class Transformer(nn.Module):
    def __init__(self, X, matrixQ, matrixK, matrixV, h=8):
        super().__init__
        self.Q = torch.matmul(X, matrixQ)
        self.K = torch.matmul(X, matrixK)
        self.V = torch.matmul(X, matrixV)
        self.h = h
        self.linear_layer = FFN()
        
    def DotProductAttention(self):
        coefs = nn.Softmax(torch.matmul(self.Q, torch.transpose(self.K, 0, 1)), dim=1) #lacks division by sqrt(dk)
        X = torch.matmul(coefs, self.V)
        return X

    def get_projections(self):
        torch.stack(self.linear(self.Q))
        torch.stack(self.linear(self.K))
        torch.stack(self.linear(self.V))
        
    def MultiHeadAttention(self):

        with mp.Pool() as pool:
            pool.map(self.get_projections(), range(self.h))
        
        with mp.Pool() as pool:
            pool.map(self.DotProductAttention(), range(self.h)) # does that work with dimentions?!
            
        torch.concat   #to be fixed
        self.linear_layer()
        
    def addNorm(self):
        pass
    def EncoderBlock(self,x ):
        self.MultiHeadAttention()
        self.addNorm()
        self.linear_layer()
        self.addNorm()
        
    #captum 
    #interpreto
        
    
    
