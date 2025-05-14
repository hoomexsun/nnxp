# src/models/cnn_attn.py
import torch
import torch.nn as nn
import math
from .base import XlitModel
from .positional_encoding import PositionalEncoding


class ConvEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, kernel_size, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.pe = PositionalEncoding(embed_dim)
        self.conv = nn.Conv1d(embed_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T)
        embedded = self.embedding(x)  # (B, T, E)
        embedded = self.pe(embedded)
        embedded = self.dropout(embedded)
        conv_input = embedded.transpose(1, 2)  # (B, E, T)
        conv_output = self.conv(conv_input).transpose(1, 2)  # (B, T, H)
        return conv_output  # (B, T, H)


class ConvAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.scale = math.sqrt(enc_dim)

    def forward(self, query, encoder_outputs):
        # query: (B, T_dec, D)   encoder_outputs: (B, T_enc, D)
        scores = torch.bmm(query, encoder_outputs.transpose(1, 2))  # (B, T_dec, T_enc)
        attn_weights = torch.softmax(scores / self.scale, dim=2)
        context = torch.bmm(attn_weights, encoder_outputs)  # (B, T_dec, D)
        return context, attn_weights


class ConvDecoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, kernel_size, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.pe = PositionalEncoding(embed_dim)
        self.conv = nn.Conv1d(embed_dim, hidden_dim * 2, kernel_size, padding=kernel_size // 2)
        self.attn = ConvAttention(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, encoder_outputs):
        # tgt: (B, T)
        embedded = self.embedding(tgt)  # (B, T, E)
        embedded = self.pe(embedded)
        embedded = self.dropout(embedded)
        conv_input = embedded.transpose(1, 2)  # (B, E, T)
        conv_output = self.conv(conv_input).transpose(1, 2)  # (B, T, 2H)

        # Gated Linear Unit
        H = conv_output.size(-1) // 2
        out, gate = conv_output[:, :, :H], torch.sigmoid(conv_output[:, :, H:])
        out = out * gate  # (B, T, H)

        context, _ = self.attn(out, encoder_outputs)  # (B, T, H)
        combined = torch.cat([out, context], dim=2)  # (B, T, 2H)
        output = self.fc_out(combined)  # (B, T, vocab)
        return output


class CNNSeq2SeqAttn(XlitModel):
    def __init__(self, model_conf: dict, device: torch.device):
        super().__init__(model_conf, device)
        self.encoder = ConvEncoder(
            input_dim=model_conf["idim"],
            embed_dim=model_conf["embed_dim"],
            hidden_dim=model_conf["hidden_dim"],
            kernel_size=model_conf["kernel_size"],
            dropout=model_conf["dropout"],
        )

        self.decoder = ConvDecoder(
            output_dim=model_conf["odim"],
            embed_dim=model_conf["embed_dim"],
            hidden_dim=model_conf["hidden_dim"],
            kernel_size=model_conf["kernel_size"],
            dropout=model_conf["dropout"],
        )

        self.teacher_forcing_ratio = model_conf.get("teacher_forcing_ratio", 0.5)

    def forward(self, x, y=None, max_len=None):
        batch_size = x.size(0)
        max_len = max_len or self.max_len
        target_len = y.size(1) if y is not None else max_len
        y_vocab_size = self.decoder.fc_out.out_features

        encoder_outputs = self.encoder(x)  # (B, T_src, H)
        outputs = torch.zeros(batch_size, target_len, y_vocab_size, device=self.device)

        inp = (
            y[:, 0]
            if y is not None
            else torch.full((batch_size,), self.sos_token, dtype=torch.long, device=self.device)
        )
        tgt_seq = torch.zeros(batch_size, target_len, dtype=torch.long, device=self.device)
        tgt_seq[:, 0] = inp

        for t in range(1, target_len):
            decoder_input = tgt_seq[:, :t]  # growing tgt sequence
            logits = self.decoder(decoder_input, encoder_outputs)  # (B, t, vocab)
            output_t = logits[:, -1, :]  # take last time step
            outputs[:, t] = output_t
            top1 = output_t.argmax(1)

            if y is not None:
                teacher_force = torch.rand(batch_size, device=self.device) < self.teacher_forcing_ratio
                inp = torch.where(teacher_force, y[:, t], top1)
            else:
                inp = top1

            tgt_seq[:, t] = inp

            if self.eos_token is not None and (inp == self.eos_token).all():
                break

        return outputs
