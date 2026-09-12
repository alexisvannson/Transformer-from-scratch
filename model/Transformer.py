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
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.heads = 8
        self.embedding_dim = embedding_dim
        self.W = nn.Parameter(torch.randn(embedding_dim,  3 * embedding_dim), requires_grad=True)
        self.Wo = nn.Parameter(torch.randn(embedding_dim, embedding_dim), requires_grad=True)
        self.linear = nn.Linear(in_features=embedding_dim//self.heads, out_features=embedding_dim//self.heads)
        self.DotProductAttention = DotProductAttention()
    
    def get_projections(self, Q, K, V):
        print("linear ", self.linear(Q).shape, self.linear(K).shape, self.linear(V).shape,)
        return self.linear(Q), self.linear(K), self.linear(V)
    
    def reshapeProjectedFusedQKVTensor(self, X):
        FusedQKVTensor = torch.matmul(X, self.W) # shape: (tokens, 1536)
                
        token_number = FusedQKVTensor.shape[0]
        # Unpack the tuple into separate Q, K, and V tensors, each shape (tokens, 512)
        Q, K, V = FusedQKVTensor.chunk(3, dim=-1)  

        # Reshape them for the 8 heads, then transpose each one so the Head dimension is first => shape: (8, tokens, 64)
        Q = Q.reshape(token_number, self.heads, self.embedding_dim//self.heads).transpose(0, 1) 
        K = K.reshape(token_number, self.heads, self.embedding_dim//self.heads).transpose(0, 1)
        V = V.reshape(token_number, self.heads, self.embedding_dim // self.heads).transpose(0, 1)
        return Q, K, V
        
    def forward(self, X):
        Q, K, V = self.reshapeProjectedFusedQKVTensor(X)
        
        #parallelize later
        final_tensor = self.DotProductAttention(Q[0], K[0], V[0]) #[10, 64]
        
        for i in range(1, self.heads):
            current = self.DotProductAttention(Q[i], K[i], V[i])
            final_tensor = torch.cat((final_tensor, current), dim=1)
        
        return torch.matmul(final_tensor, self.Wo)

class FinalMultiHeadAttention(MultiHeadAttention):
    def __init__(self, embedding_dim=512):
        super().__init__(embedding_dim)
    
    def forward(self, X):
        Q, K, V = self.reshapeProjectedFusedQKVTensor(X)
                
        #parallelize later
        final_tensor = self.DotProductAttention(Q[0], K[0], V[0]) #[10, 64]
        
        for i in range(1, self.heads):
            current = self.DotProductAttention(Q[i], K[i], V[i])
            final_tensor = torch.cat((final_tensor, current), dim=1)
        
        return torch.matmul(final_tensor, self.Wo), K, V

class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.FFN = FFN()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.MultiHeadAttention = MultiHeadAttention()
    
    def AddAndNorm(self, initialX, finalX):
        X = initialX + finalX
        print(initialX.dtype, finalX.dtype)
        X = self.layer_norm(X)
        return X
        
    def forward(self, X):
        X = self.AddAndNorm(X, self.MultiHeadAttention(X))
        X = self.AddAndNorm(X, self.FFN(X))
        return X

class FinalEncoderBlock(EncoderBlock):
    def __init__(self, embedding_dim=512):
        super().__init__(embedding_dim)
        self.MultiHeadAttention = FinalMultiHeadAttention()
        
    def forward(self, X):
        finalX, K, V = self.MultiHeadAttention(X)
        X = self.AddAndNorm(X, finalX)
        X = self.AddAndNorm(X, self.FFN(X))
        return X, K, V
    
class Encoder(nn.Module):
    def __init__(self, n_blocks=6):
        super().__init__()
        EncoderBlocks = [EncoderBlock() for _ in range(n_blocks - 1)]
        self.EncoderBlocks = nn.Sequential(*EncoderBlocks)
        self.final_block = FinalEncoderBlock()
        
    def forward(self, X):
        X = self.EncoderBlocks(X)
        X, K, V = self.final_block(X)
        return X, K, V


class MaskedMultiHeadAttention(MultiHeadAttention):
    def __init__(self, embedding_dim=512):
        super().__init__(embedding_dim)
        self.dk = 64
    def getMask(self, dim):
        mask = torch.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                if (i < j):
                    mask[i][j] = float('-inf')
        return mask
    
    def MaskedDotProductAttention(self, Mask, Q, K, V):
            coefs = F.softmax((torch.matmul(Q, torch.transpose(K, 0, 1)) * (1 / sqrt(self.dk))) + Mask, dim=1) # softmax applied horizontally
            X = torch.matmul(coefs, V)
            return X
        
    def forward(self,  X):
        causal_mask = self.getMask(X.shape[0])
        Q, K, V = self.reshapeProjectedFusedQKVTensor(X)
            
        #parallelize later
        final_tensor = self.MaskedDotProductAttention(causal_mask, Q[0], K[0], V[0])

        for i in range(1, self.heads):
            current = self.MaskedDotProductAttention(causal_mask, Q[i], K[i], V[i])
            final_tensor = torch.cat((final_tensor, current), dim=1)

        X = torch.matmul(final_tensor, self.Wo) 
        
        return X, Q
           
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


class CrossAttention(MultiHeadAttention):   
    def __init__(self):
        super().__init__()
    
    def forward(self, X, Q, K, V ):
        #parallelize later
        final_tensor = self.DotProductAttention(Q[0], K[0], V[0]) #[10, 64]
        
        for i in range(1, self.heads):
            current = self.DotProductAttention(Q[i], K[i], V[i])
            final_tensor = torch.cat((final_tensor, current), dim=1)
        
        return torch.matmul(final_tensor, self.Wo)

class DecoderBlock(nn.Module):
    def __init__(self, K, V, embedding_dim=512):
        super().__init__()
        self.K = K
        self.V = V
        self.FFN = FFN()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.MaskedMultiHeadAttention = MaskedMultiHeadAttention()
        self.CrossAttention = CrossAttention()
            
    def AddAndNorm(self, initialX, finalX):
        X = initialX + finalX
        X = self.layer_norm(X)
        return X

    def forward(self, X):
        X1, Q = self.MaskedMultiHeadAttention(X)
        #print("Maskedattentionresult", X1)
        X = self.AddAndNorm(X,  X1)
        #print("AddNorm", X)
        X = self.AddAndNorm(X, self.CrossAttention(X, Q, self.K, self.V))
        X = self.AddAndNorm(X, self.FFN(X))
        print("decoderblock result", X)
        return X

class Decoder(nn.Module):
    def __init__(self, K, V, n_blocks=6):
        super().__init__()
        DecoderBlocks = [DecoderBlock(K, V) for _ in range(n_blocks)]
        self.DecoderBlocks = nn.Sequential(*DecoderBlocks)
        
    def forward(self, X):
        X = self.DecoderBlocks(X)
        return X

class Transformer(nn.Module):
    def __init__(self, out_dim,  emmbedding_dim=512):
        super().__init__()
        self.encoder = Encoder()
        self.linear = nn.Linear(emmbedding_dim, out_dim)
    
    def forward(self, X):
        X, K, V = self.encoder(X)
        decoder =  Decoder(K, V)
        X = decoder(X)
        X = self.linear(X)
        return F.softmax(X, dim=1)
        
