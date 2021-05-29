"""
baseline model (no attn, no row encoded rnn,
                zero initialization for decoder hidden states)

cnn: stanford
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Im2LatexModel(nn.Module):
    def __init__(self, out_size, emb_size,
                 enc_rnn_h, dec_rnn_h, n_layer=1):
        super(Im2LatexModel, self).__init__()

        # follow the original paper's table2: CNN specification
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0)),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0)),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),

            # vanilla
            nn.MaxPool2d((2, 1), (2, 1), (0, 0)),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),

            # vanilla
            nn.MaxPool2d((1, 2), (1, 2), (0, 0)),

            nn.Conv2d(512, 512, 3, 1, 0),
            nn.ReLU()
        )

        self.rnn_decoder = nn.LSTMCell(enc_rnn_h+emb_size, dec_rnn_h)
        self.embedding = nn.Embedding(out_size, emb_size)

        # enc_rnn_h*2 is the dimension of context
        self.W_c = nn.Linear(dec_rnn_h+2*enc_rnn_h, enc_rnn_h)
        self.W_out = nn.Linear(enc_rnn_h, out_size)

        self.dec_rnn_h = dec_rnn_h

    def forward(self, imgs, formulas):
        """args:
        imgs: [B, C, H, W]
        formulas: [B, MAX_LEN]

        return:
        logits: [B, MAX_LEN, VOCAB_SIZE]
        """
        # encoding
        row_enc_out = self.encode(imgs) # [B, H', W', 512]
        B, H, _, _ = row_enc_out.shape

        # zero init decoder's states
        h_0 = torch.zeros(B, self.dec_rnn_h)
        c_0 = torch.zeros(B, self.dec_rnn_h)
        dec_states = (h_0, c_0)

        context_0 = row_enc_out.mean(dim=[1, 2])

        # init_O
        O_t = torch.tanh(self.W_c(torch.cat([h_0, context_0], dim=1)))

        max_len = formulas.size(1)
        logits = []
        for t in range(max_len):
            tgt = formulas[:, t:t+1]
            # ont step decoding
            dec_states, O_t, logit = self.step_decoding(dec_states, O_t,
                                                        row_enc_out, tgt)
            logits.append(logit)
        logits = torch.stack(logits, dim=1)  # [B, MAX_LEN, out_size]
        return logits

    def encode(self, imgs):
        encoded_imgs = self.cnn_encoder(imgs)  # [B, 512, H', W']
        encoded_imgs = encoded_imgs.permute(0, 2, 3, 1)  # [B, H', W', 512]
        B, H, W, out_channels = encoded_imgs.size()
        encoded_imgs = encoded_imgs.view(B, H, W, -1)  # [B, H, W, enc_rnn_h*2]
        return encoded_imgs

    def step_decoding(self, dec_states, O_t, enc_out, tgt):
        """Runing one step decoding"""

        prev_y = self.embedding(tgt).squeeze(1)  # [B, emb_size]
        inp = torch.cat([prev_y, O_t], dim=1)  # [B, emb_size+enc_rnn_h]
        h_t, c_t = self.rnn_decoder(inp, dec_states)

        context_t = enc_out.mean(dim=[1, 2]) # = context_0 # [B, enc_rnn_h*2]

        O_t = self.W_c(torch.cat([h_t, context_t], dim=1)).tanh() # [B, enc_rnn_h]

        # calculate logit
        logit = F.softmax(self.W_out(O_t), dim=1)  # [B, out_size]

        return (h_t, c_t), O_t, logit
