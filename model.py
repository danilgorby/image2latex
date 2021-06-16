import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchvision.models as models
from positional_encoding import PositionalEncoding2d
from transformer import TransformerEncoder, TransformerDecoder
from make_vocab import PAD_TOKEN
from utils import get_pad_mask, get_subsequent_mask

INIT = 1e-2


class Im2LatexTransformerModel(nn.Module):
    """
    Naive transformer-based model, based on Harvard Im2Latex, but with LSTM encoder-decoder replaced with transformer
    """
    def __init__(self, out_size, n_blocks, n_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.cnn_encoder = self.cnn_encoder = nn.Sequential(
                nn.Conv2d(3, 512, 3, 1, 0),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d((1, 2), (1, 2), (0, 0)),
                nn.Conv2d(512, 256, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d((2, 1), (2, 1), (0, 0)),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d((2, 2), (2, 2), (0, 0)),
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d((2, 2), (2, 2), (1, 1))
            )
        d_model = 64
        self.pos_encoder = PositionalEncoding2d(d_model, dropout)
        self.trans_encoder = TransformerEncoder(d_model, n_blocks, n_heads, d_ff, dropout, pe1d=False)
        self.trans_decoder = TransformerDecoder(out_size, d_model, n_blocks, n_heads, d_ff, dropout, pe1d=True)

    def forward(self, imgs, formulas):
        """args:
        imgs: [B, C, H, W]
        formulas: [B, MAX_LEN]

        return:
        logits: [B, MAX_LEN, VOCAB_SIZE]
        """

        # encoding
        src_mask = None
        e_outputs = self.encode(imgs, src_mask)  # [B, H'*W', 64]
        # decoding
        formulas_mask = get_pad_mask(formulas) & get_subsequent_mask(formulas)
        d_output = self.trans_decoder(formulas, e_outputs, src_mask, formulas_mask)
        output = self.out(d_output)
        return output

    def encode(self, imgs, src_mask=None):
        encoded_imgs = self.cnn_encoder(imgs)  # [B, 64, H', W']
        encoded_imgs = encoded_imgs.permute(0, 2, 3, 1)  # [B, H', W', 64]

        B, H, W, out_channels = encoded_imgs.size()

        assert out_channels == 64, 'Wrong number of out channels'

        encoded_imgs = self.pos_encoder(encoded_imgs)
        encoded_imgs = encoded_imgs.contiguous().view(B, H*W, out_channels)
        e_outputs = self.trans_encoder(encoded_imgs, mask=src_mask)  # [B, H'*W', 64]
        return e_outputs


class Im2LatexModel(nn.Module):
    def __init__(self, out_size, emb_size,
                 enc_rnn_h, dec_rnn_h,
                 cnn, attn, pos_enc,
                 dec_init, n_layer=1):

        super(Im2LatexModel, self).__init__()

        self.dec_rnn_h = dec_rnn_h
        self.pos_enc = pos_enc

        if cnn == 'stanford':
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
            emb_size_rnn_enc = 512
        elif cnn == 'harvard':
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(3, 512, 3, 1, 0),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d((1, 2), (1, 2), (0, 0)),
                nn.Conv2d(512, 256, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d((2, 1), (2, 1), (0, 0)),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d((2, 2), (2, 2), (0, 0)),
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d((2, 2), (2, 2), (1, 1))
            )
            emb_size_rnn_enc = 64
            if pos_enc != 'rnn_enc':
                self.cnn_encoder = nn.Sequential(self.cnn_encoder,
                                                 nn.Conv2d(emb_size_rnn_enc, 512, 1, 1, 0)
                                                )
        elif cnn == 'densenet1':
            # 2 blocks
            model = models.densenet161(pretrained=False)
            self.cnn_encoder = torch.nn.Sequential(*(list(model.children())[0][:8]))
            emb_size_rnn_enc = 384
            if pos_enc != 'rnn_enc':
                self.cnn_encoder = nn.Sequential(self.cnn_encoder,
                                                 nn.Conv2d(emb_size_rnn_enc, 512, 1, 1, 0)
                                                )
        elif cnn == 'densenet2':
            # 3 blocks
            model = models.densenet161(pretrained=False)
            self.cnn_encoder = torch.nn.Sequential(*(list(model.children())[0][:-2]))
            emb_size_rnn_enc = 1056
            if pos_enc != 'rnn_enc':
                self.cnn_encoder = nn.Sequential(self.cnn_encoder,
                                                 nn.Conv2d(emb_size_rnn_enc, 512, 1, 1, 0)
                                                )
        else:
            raise ValueError('No such cnn architecture.')

        if pos_enc == 'rnn_enc':
            self.pos_encoder = nn.LSTM(emb_size_rnn_enc, enc_rnn_h,
                                       bidirectional=True,
                                       batch_first=True)

            # a trainable initial hidden state V_h_0 for each row
            self.V_h_0 = nn.Parameter(torch.Tensor(n_layer * 2, enc_rnn_h))
            self.V_c_0 = nn.Parameter(torch.Tensor(n_layer * 2, enc_rnn_h))
            init.uniform_(self.V_h_0, -INIT, INIT)
            init.uniform_(self.V_c_0, -INIT, INIT)
        elif pos_enc == 'spacial2d_enc':
            self.pos_encoder = PositionalEncoding2d(512)  # выглядит как хардкодинг, надо все как-то в переменную типа out_cahnnels завернуть
        elif pos_enc == 'none':
            self.pos_encoder = None
        else:
            raise ValueError(f'There is no {pos_enc} positional encoding options. Possible positional encoding options'
                             f'are: rnn_enc, spacial2d_enc, None')

        self.rnn_decoder = nn.LSTMCell(enc_rnn_h+emb_size, dec_rnn_h)
        self.embedding = nn.Embedding(out_size, emb_size)

        # enc_rnn_h*2 is the dimension of context
        self.W_c = nn.Linear(dec_rnn_h+2*enc_rnn_h, enc_rnn_h)
        self.W_out = nn.Linear(enc_rnn_h, out_size)

        # Attention mechanism
        self.attn = attn
        if attn > 0:
            self.beta_SW = nn.Parameter(torch.Tensor(dec_rnn_h))
            init.uniform_(self.beta_SW, -INIT, INIT)
            self.W_h_SW = nn.Linear(dec_rnn_h, dec_rnn_h) # bias=False?
            self.W_v_SW = nn.Linear(enc_rnn_h*2, dec_rnn_h)
            if attn == 2:
                self.beta_CW = nn.Parameter(torch.Tensor(dec_rnn_h))
                init.uniform_(self.beta_CW, -INIT, INIT)
                self.W_h_CW = nn.Linear(dec_rnn_h, dec_rnn_h) # bias=False?
                self.W_v_CW = nn.Linear(enc_rnn_h*2, dec_rnn_h)

        self.dec_init = dec_init
        if dec_init == 1:
            # dec hidden state init
            self.dec_rnn_h = dec_rnn_h
            self.W_h0 = nn.Linear(512, dec_rnn_h) # !!!
            self.W_c0 = nn.Linear(512, dec_rnn_h) # emb_size_rnn_enc

    def forward(self, imgs, formulas):
        """args:
        imgs: [B, C, H, W]
        formulas: [B, MAX_LEN]

        return:
        logits: [B, MAX_LEN, VOCAB_SIZE]
        """

        # encoding
        row_enc_out, hiddens = self.encode(imgs)  # [B, H', W', 512]
        # init decoder's states
        dec_states, O_t = self.init_decoder(row_enc_out, hiddens)

        max_len = formulas.size(1)
        logits = []
        for t in range(max_len):
            tgt = formulas[:, t:t+1]
            # ont step decoding
            dec_states, O_t, logit = self.step_decoding(
                dec_states, O_t, row_enc_out, tgt)
            logits.append(logit)
        logits = torch.stack(logits, dim=1)  # [B, MAX_LEN, out_size]
        return logits

    def encode(self, imgs):
        encoded_imgs = self.cnn_encoder(imgs)  # [B, 64, H', W']
        encoded_imgs = encoded_imgs.permute(0, 2, 3, 1)  # [B, H', W', 64]

        (h, c) = (None, None)
        B, H, W, out_channels = encoded_imgs.size()
        if self.pos_enc == 'rnn_enc':
            # Prepare data for Row Encoder
            # poccess data like a new big batch
            encoded_imgs = encoded_imgs.contiguous().view(B*H, W, out_channels)
            # prepare init hidden for each row
            init_hidden_h = self.V_h_0.unsqueeze(
                1).expand(-1, B*H, -1).contiguous()
            init_hidden_c = self.V_c_0.unsqueeze(
                1).expand(-1, B*H, -1).contiguous()
            init_hidden = (init_hidden_h, init_hidden_c)

            # Row Encoder
            row_enc_out, (h, c) = self.pos_encoder(encoded_imgs, init_hidden)
            # row_enc_out [B*H, W, enc_rnn_h]
            # hidden: [2, B*H, enc_rnn_h]
            row_enc_out = row_enc_out.view(B, H, W, -1)  # [B, H, W, enc_rnn_h]
            h, c = h.view(2, B, H, -1), c.view(2, B, H, -1)
            return row_enc_out, (h, c)
        else:
            if self.pos_enc == 'spatial2d_enc':
                encoded_imgs = self.pos_encoder(encoded_imgs)
            encoded_imgs = encoded_imgs.view(B, H, W, -1)  # [B, H, W, enc_rnn_h*2]
            return encoded_imgs, (h, c)

    def step_decoding(self, dec_states, O_t, enc_out, tgt):
        """Runing one step decoding"""

        prev_y = self.embedding(tgt).squeeze(1)  # [B, emb_size]
        inp = torch.cat([prev_y, O_t], dim=1)  # [B, emb_size+enc_rnn_h]
        h_t, c_t = self.rnn_decoder(inp, dec_states)

        if self.attn == 0:
            context_t = enc_out.mean(dim=[1, 2])  # = context_0 # [B, enc_rnn_h*2]
        elif self.attn == 1:
            context_t, attn_scores = self._get_SW_attn(enc_out, dec_states[0])  # [B, enc_rnn_h*2]
        elif self.attn == 2:
            context_t, attn_scores = self._get_CW_attn(enc_out, dec_states[0])  # [B, H, W, enc_rnn_h*2]
            context_t, attn_scores = self._get_SW_attn(context_t, dec_states[0]) # [B, enc_rnn_h*2]
        else:
            raise ValueError('possible values for attn are 0, 1, 2')

        O_t = self.W_c(torch.cat([h_t, context_t], dim=1)).tanh() # [B, enc_rnn_h]

        # calculate logit
        logit = F.softmax(self.W_out(O_t), dim=1)  # [B, out_size]

        return (h_t, c_t), O_t, logit

    def _get_SW_attn(self, enc_out, prev_h):
        """Attention mechanism
        args:
            enc_out: row encoder's output [B, H, W, enc_rnn_h]
            prev_h: the previous time step hidden state [B, dec_rnn_h]
        return:
            context: this time step context [B, H, W, enc_rnn_h]
            attn_scores: Attention scores
        """
        # spatial-wise attn
        B, H, W, _ = enc_out.size()
        linear_prev_h = self.W_h_SW(prev_h).view(B, 1, 1, -1)
        linear_prev_h = linear_prev_h.expand(-1, H, W, -1)
        e = torch.sum(
            self.beta_SW * torch.tanh(
                linear_prev_h +
                self.W_v_SW(enc_out)
            ),
            dim=-1
        )  # [B, H, W]

        alpha = F.softmax(e.view(B, -1), dim=-1).view(B, H, W)
        attn_scores = alpha.unsqueeze(-1)
        context = torch.sum(attn_scores * enc_out,
                            dim=[1, 2])  # [B, enc_rnn_h]

        return context, attn_scores

    def _get_CW_attn(self, enc_out, prev_h):
        """Attention mechanism
        args:
            enc_out: row encoder's output [B, H, W, enc_rnn_h]
            prev_h: the previous time step hidden state [B, dec_rnn_h]
        return:
            context: this time step context [B, enc_rnn_h]
            attn_scores: Attention scores
        """
        # channel-wise attn
        B, H, W, _ = enc_out.size()
        linear_prev_h = self.W_h_CW(prev_h).view(B, 1, 1, -1)
        linear_prev_h = linear_prev_h.expand(-1, H, W, -1)

        e = torch.sum(
            self.beta_CW * torch.tanh(
                linear_prev_h +
                self.W_v_CW(enc_out)
            ),
            dim=[1, 2]
        )  # [B, C]

        alpha = F.softmax(e, dim=-1)
        attn_scores = alpha.unsqueeze(1).unsqueeze(1)
        context = attn_scores * enc_out  # [B, H, W, enc_rnn_h]
        return context, attn_scores

    def init_decoder(self, enc_out, hiddens):
        """args:
            enc_out: the output of row encoder [B, H, W, enc_rnn_h]
            hidden: the last step hidden of row encoder [2, B, H, enc_rnn_h]
          return:
            h_0, c_0  h_0 and c_0's shape: [B, dec_rnn_h]
            init_O : the average of enc_out  [B, enc_rnn_h]
            for decoder
        """

        h, c = hiddens
        B, H, _, _ = enc_out.shape
        device = enc_out.device
        if h is None:
            # no rnn enc
            if self.dec_init == 0:
                # zero init
                h = torch.zeros(B, self.dec_rnn_h)  # h_0
                c = torch.zeros(B, self.dec_rnn_h)  # c_0
                h, c = h.to(device), c.to(device)
            elif self.dec_init == 1:
                # non-zero init (может можно по-другому)
                v = enc_out.mean(dim=[1, 2])
                h = self.W_h0(v).tanh()  # h_0
                c = self.W_c0(v).tanh()  # c_0
        else:
            h, c = self._convert_hidden(h), self._convert_hidden(c)
        context_0 = enc_out.mean(dim=[1, 2])
        init_O = torch.tanh(self.W_c(torch.cat([h, context_0], dim=1)))
        return (h, c), init_O

    def _convert_hidden(self, hidden):
        """convert row encoder hidden to decoder initial hidden"""
        hidden = hidden.permute(1, 2, 0, 3).contiguous()
        # Note that 2*enc_rnn_h = dec_rnn_h
        hidden = hidden.view(hidden.size(
            0), hidden.size(1), -1)  # [B, H, dec_rnn_h]
        hidden = hidden.mean(dim=1)  # [B, dec_rnn_h]

        return hidden
