import torch
import torch.nn as nn
import multiprocessing as mp
    

X = torch.randn(10, 512)

W = torch.randn(512, 1536)
Wq = torch.randn(512, 64)
Wk = torch.randn(512, 64)
Wv = torch.randn(512, 64)



# shape 10 x 64
Q = torch.matmul(X, Wq)
K = torch.matmul(X, Wk)
V = torch.matmul(X, Wv)

FusedQKVTensor = torch.matmul(X, W) # shape 10 x 1536
FusedQKVTensor = FusedQKVTensor.chunk(3, dim=-1) # shape 10 x  3 x 512
FusedQKVTensor = FusedQKVTensor.reshape(10, 3, 8, 64)
#[QKV=3, Heads=8, SeqLen=100, HeadDim=64]

#linear projection 8 times (done once with matrix multiplication)

#concatenate all => shape 10 x 512 then multiply by Wo to get new X


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
        x =  self.linear2(x)
        return x
        
        
class Transformer(nn.Module):
    def __init__(self, X, matrixQ, matrixK, matrixV, h=8):
        super().__init__
        self.Q = torch.matmul(X, matrixQ)
        self.K = torch.matmul(X, matrixK)
        self.V = torch.matmul(X, matrixV)
        self.h = h
        self.linear_layer = nn.Linear()
    
    def updateX(self, X):
        self.X = X
    def updateQ(self, Q):
            self.Q = Q
    def updateK(self, K):
            self.K = K
    def updateV(self, V):
            self.V = V
            
    def DotProductAttention(self, Q, K, V):
        coefs = nn.Softmax(torch.matmul(Q, torch.transpose(K, 0, 1)), dim=1) #lacks division by sqrt(dk)
        X = torch.matmul(coefs, V)
        return X

    def get_projections(self):
        torch.stack(self.linear(self.Q))
        torch.stack(self.linear(self.K))
        torch.stack(self.linear(self.V))
        
    def MultiHeadAttention(self, Q, K, V):
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
        
    
    
