import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encoding import PositionalEncoding1d


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.attn(x2, x2, x2, attn_mask=mask)[0])
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ff(x2))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.attn1 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.attn2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, e_outputs, src_mask, tgt_mask):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.attn1(x2, x2, x2, attn_mask=tgt_mask)[0])
        x2 = self.norm2(x)
        x = x + self.dropout2(self.attn2(x2, e_outputs, e_outputs, attn_mask=src_mask)[0])
        x2 = self.norm3(x)
        x = x + self.dropout3(self.ff(x2))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_blocks, n_heads, d_ff, dropout, pe1d=False):
        super().__init__()
        self.n_blocks = n_blocks
        self.pe = PositionalEncoding1d(d_model, dropout=dropout) if pe1d else nn.Identity()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask):
        x = self.pe(src)
        x = x.transpose(0, 1)
        for i in range(self.n_blocks):
            x = self.layers[i](x, mask)
        return self.norm(x).transpose(0, 1)


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_blocks, n_heads, d_ff, dropout, pe1d=False):
        super().__init__()
        self.n_blocks = n_blocks
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding1d(d_model, dropout=dropout) if pe1d else nn.Identity()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, e_outputs, src_mask, tgt_mask):
        x = self.embed(tgt)
        x = x.transpose(0, 1)
        x = self.pe(x)
        for i in range(self.n_blocks):
            x = self.layers[i](x, e_outputs, src_mask, tgt_mask)
        return self.norm(x).transpose(0, 1)


class Transformer(nn.Module):
    def __init__(self, tgt_vocab, d_model, n_blocks, n_heads, d_ff, dropout):
        super().__init__()
        self.encoder = TransformerEncoder(d_model, n_blocks, n_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(tgt_vocab, d_model, n_blocks, n_heads, d_ff, dropout)
        self.out = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt, src_mask, tgt_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(tgt, e_outputs, src_mask, tgt_mask)
        output = self.out(d_output)
        return output
