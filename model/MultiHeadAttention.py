import torch
import torch.nn as nn
from math import sqrt, sin, cos
import torch.nn.functional as F

#requires_grad_=True
class FFN(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, out_dim=512):
        super().__init__()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x =  self.linear2(x)
        return x

class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.dk = 64
        
    def forward(self, Q, K, V):
        coefs = F.softmax(torch.matmul(Q, torch.transpose(K, 0, 1)) * (1 / sqrt(self.dk)), dim=1) # softmax applied horizontally
        X = torch.matmul(coefs, V)
        return X

class MultiHeadAttention(nn.Module):
    def __init__(self, W, Wo):
        super().__init__()
        self.linear = nn.Linear(in_features=64, out_features=64)
        self.DotProductAttention = DotProductAttention()
        self.heads = 8
        self.embedding_dim= 512
        self.W = W
        self.Wo = Wo
    
    def get_projections(self, Q, K, V):
        print("linear ", self.linear(Q).shape, self.linear(K).shape, self.linear(V).shape,)
        return self.linear(Q), self.linear(K), self.linear(V)
    
        
    def forward(self, X):
        FusedQKVTensor = torch.matmul(X, self.W) # shape: (tokens, 1536)
        
        token_number = FusedQKVTensor.shape[0]
        # Unpack the tuple into separate Q, K, and V tensors, each shape (tokens, 512)
        Q, K, V = FusedQKVTensor.chunk(3, dim=-1)  

        # Reshape them for the 8 heads, then transpose each one so the Head dimension is first => shape: (8, tokens, 64)
        Q = Q.reshape(token_number, self.heads, self.embedding_dim // self.heads).transpose(0, 1) 
        K = K.reshape(token_number, self.heads, self.embedding_dim // self.heads).transpose(0, 1)
        V = V.reshape(token_number, self.heads, self.embedding_dim // self.heads).transpose(0, 1)
        
        projectedq1, projectedk1, projectedv1 = self.get_projections(Q[0], K[0], V[0]) #10, 64
        
        #parallelize later
        final_tensor = self.DotProductAttention(projectedq1, projectedk1, projectedv1) #[10, 64]
        
        for i in range(1, self.heads):
            print(i)
            projectedQ, projectedK, projectedV = self.get_projections(Q[i], K[i], V[i])
            current = self.DotProductAttention(projectedQ, projectedK, projectedV)
            final_tensor = torch.cat((final_tensor, current), dim=1)
            print(final_tensor.shape)
        
        return torch.matmul(final_tensor, self.Wo)

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, W, Wo):
        super().__init__()
        self.W = W
        self.Wo = Wo    
        self.multiheadAttention = MultiHeadAttention(self.W, self.Wo)
    
    def forward(self,  X):
        seq_len = X.shape[0]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
        return self.MultiHeadAttention(X) + causal_mask
        
class Encoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.FFN = FFN()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.MultiHeadAttention = MultiHeadAttention()
    
    def AddAndNorm(self, initialX, finalX):
        X = initialX + finalX
        X = self.layer_norm(X)
        return X
        
    def EncoderBlock(self,X, W, Wo):
        x = self.AddAndNorm(X, self.MultiHeadAttention(X, W, Wo))
        x = self.AddAndNorm(x, self.FFN(x))
        return X
    
    def forward(self, x):
        pass
            
class PositionalEncoding:
    def __init__(self):
        self.d_model = 64
    
    def getPositionalEncoding(self, pos, i): #could be replace by pytorch dict
        if (i % 2 == 0):
            return sin(pos/ (10000 ** (2*i / self.d_model)))
        else:
            return cos(pos/ (10000 ** (2*i / self.d_model)))
        
    def AddPositionalEncodings(self, words_embeddings, pos):
        for word_embedding in words_embeddings:
            for i in range(len(word_embedding)):
                word_embedding[i] += self.getPositionalEncoding(pos, i)
        return words_embeddings
        
    
class Transformer(nn.Module):
    def __init__(self):
        pass
















