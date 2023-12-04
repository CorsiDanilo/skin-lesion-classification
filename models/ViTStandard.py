import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        pos_enc = torch.zeros((max_len, embed_dim))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer('positional_encoding', pos_enc)

    def forward(self, x):
        B, L, E = x.shape #batch size, sequence length, embedding dimension
        return x + self.positional_encoding[:, :L, :]

'''
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.patch_embedding = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.image_size[0] and W == self.image_size[1] #H and W must match
        x = self.patch_embedding(x)
        x = x.view(B, self.embed_dim, -1).transpose(1, 2) #Flatten patches
        return x
'''

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, d_model):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = d_model

        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size) #Calculate the number of patches
        self.projection = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size) # Define linear projection for each patch

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.image_size[0] and W == self.image_size[1] #H and W must match
        patches = self.projection(x) #Linear projection to obtain patches
        patches = patches.view(B, self.embed_dim, self.num_patches).transpose(1, 2) #Reshape patches to (B, d_model, num_patches)

        return patches

class Attention(nn.Module):
    def __init__(self, attn_dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)  # Get the size of the key
        attn = torch.einsum('BHLD, BHMD -> BHLM', query, key)  # Compute the dot product of the query and key, and scale it
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn/math.sqrt(d_k), dim=-1))
        output = torch.einsum('BHLL, BHLD -> BHLD', attn, value)  # Compute the weighted sum of the values
        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = Attention()

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hidden, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_hidden) # position-wise
        self.w2 = nn.Linear(d_hidden, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w2(F.relu(self.w1(x)))
        x = self.dropout(x)
        x += residual
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class TransformerEncoder(nn.Module):
    def __init__(self, img_size, in_channels, d_model, patch_size, n_layers, n_head, d_k, d_v, d_inner, dropout=0.1, scale_emb=False):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, return_attns=False):
        enc_slf_attn_list = []
        enc_output = self.patch_embedding(src_seq)
        enc_output = self.dropout(self.positional_encoding(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output

class ViT_standard(nn.Sequential):
    def __init__(self, in_channels, n_head, patch_size, d_model, img_size, n_layers, n_classes):
        super(ViT_standard, self).__init__()
        #self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, emb_size)
        #self.positional_encoding = PositionalEncoding(emb_size)
        d_k = d_v = d_model // n_head
        d_inner = d_model
        self.transformer_encoder = TransformerEncoder(img_size, in_channels, d_model, patch_size, n_layers, n_head, d_k, d_v, d_inner)
        self.fc = nn.Linear(d_model, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

