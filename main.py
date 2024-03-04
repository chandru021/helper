import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):

    def __init__(self , vec_dim : int , voc_size: int):
        super().__init__()
        self.vec_dim = vec_dim
        self.voc_size = voc_size
        self.embedding = nn.Embedding(voc_size , vec_dim) 

    def get_vector(x : int):
        return self.embedding(x) * math.sqrt(self.vec_dim) 
    
class Positional_Encoding(nn.Module):

    def __init__(self , vec_dim : int , seq_len : int , dropout : float):
        super().__init__()
        self.vec_dim = vec_dim 
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        #positional encoding  

        pe = torch.zeros(seq_len , vec_dim)

        position = torch.arange(0 , seq_len , dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, vec_dim , 2).float() * (-math.log(10000.0)/ vec_dim))

        pe[:,0::2] = torch.sin(positio n * div_term)
        pe[: ,1 :: 2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe' , pe)

    def forward(self , x) : 
        
        x = x + (self.pe[: , : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNorm(nn.Module):

    def __init__(self , epsilon:float=10**-6):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(0))

    def forward(self,x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1 , keepdim=True)

        return self.alpha * (x - mean) / (std + self.epsilon) + self.bias


class FeedForward(nn.Module):

    def __init__(self , vec_dim : int , d_ff : int , dropout : float):
        super.__init__()
        self.inputLayer = nn.Linear(vec_dim , d_ff)
        self.dropout = nn.Dropout(dropout)
        self.outputLayer = nn.Linear(d_ff , vec_dim)
        

    def forward(self , x):
        # batch , seq_len , vec_dim

        return self.outputLayer(self.dropout(torch.relu(self.inputLayer(x))))


class MultiHeadAttentionLayer(nn.Module):

    def __init__(self , vec_dim : int , heads : int , dropout : float):
        super().__init__()
        self.vec_dim = vec_dim 
        self.dropout = nn.Dropout(dropout)
        self.heads = heads

        assert vec_dim % heads == 0 , "vec_dim is not divisible by heads"

        self.dim_k = vec_dim // heads

        self.w_k = nn.Linear(vec_dim , vec_dim)
        self.w_v = nn.Linear(vec_dim , vec_dim)
        self.w_q = nn.Layer(vec_dim , vec_dim)
        self.w_o = nn.Layer(vec_dim , vec_dim)


    @staticmethod
    def attention(query , key , value , mask , dropout : nn.Dropout):
        d_k = query.shape[-1]

        atention_score = (query @ key.transpose(-2,-1)) / math.sqrt(d_k) 

        if mask is not None :
            atention_score.masked_fill_(mask == 0 , -1e9)
        
        if dropout is not None : 
            atention_score = dropout(atention_score)

        return (atention_score @ value), atention_score

    

    def forward(self, q, k, v, mask):

        query = self.w_q(q)
        key = self.w_k(k)  
        value = self.w_v(v)
         
        query = query.veiw(query.shape[0] , query.shape[1] , self.heads , self.dim_k).transpose(1 , 2)
        key = key.view(key.shape[0] , key.shape[1], self.heads , self.dim_k).transpose(1 , 2)
        value = value.view(value.shape[0] , value.shape[1] , self.heads, self.dim_k).transpose(1,2)

         
        x , self.atention_score = MultiHeadAttentionLayer.attention(query, key , value , mask , self.dropout)


        x = x.transpose(1 , 2).contiguous().view(x.shape[0] , -1 , self.d_k * self.heads)

        return self.w_o(x)


class ResidualConnection(nn.Module):

    def __init__(self , dropout : float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x , subLayer):
        return x + self.dropout(subLayer(self.norm(x)))
    


class EncoderBlock(nn.Module):

    def __init__(self, self_atention_block : MultiHeadAttentionLayer, feed_forward_block : FeedForward , dropout: float):
        super().__init__()
        self.self_atention_block = self_atention_block
        self.feed_forward_block = feed_forward_block
        self.res_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x , src_mask):

        x = self.res_connections[0](x , lambda x : self.self_atention_block(x , x , x , src_mask))
        x = self.res_connections[1](x , self.feed_forward_block(x))
        return x 
    
class Encoder(nn.Module):

    def __init__(self , layers : nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()  

    def forward(self , x , mask):

        for layer in self.layers :
            x = layer( x , mask)

        return self.norm(x) 


class DecoderBlock(nn.Module):

    def __init__(self , self_attention_block : MultiHeadAttentionLayer , feed_forward_block : FeedForward , cross_attention_block : MultiHeadAttentionLayer, dropout : float):
        super().__init__()
        self.self_attention_block  = self_attention_block
        self.feed_forward_block = feed_forward_block 
        self.cross_attention_block = cross_attention_block

        self.residualConnection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self , x , encoder_output , src_mask , target_mask):
        x = self.residualConnection[0](x , lambda x : self.self_attention_block(x, x, x, target_mask))
        x = self.residualConnection[1](x , lambda x : self.cross_attention_block(x , encoder_output, encoder_output , src_mask))
         
