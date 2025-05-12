import torch
import torch.nn as nn

from .base import XlitModel
from .positional_encoding import PositionalEncoding

class CNNEncoder(nn.Module):
    def __init__(self, idim: int, embed_dim: int, num_kernels: int, kernel_size: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(idim, embed_dim)
        self.pe = PositionalEncoding(embed_dim)
        self.conv = nn.Conv1d(embed_dim, num_kernels, kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embedded = self.pe(embedded)
        embedded = self.dropout(embedded)
        conv_input = embedded.transpose(1, 2)  # (batch_size, embed_dim, seq_len)
        conv_output = self.conv(conv_input)  # (batch_size, num_kernels, seq_len)
        conv_output = conv_output.transpose(1, 2)  # (batch_size, seq_len, num_kernels)
        return conv_output

class CNNDecoder(nn.Module):
    def __init__(self, odim: int, embed_dim: int, num_kernels: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(odim, embed_dim)
        self.rnn = nn.GRU(embed_dim + num_kernels, num_kernels, batch_first=True)
        self.fc_out = nn.Linear(num_kernels, odim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, context, hidden):
        # inp: (batch_size,)
        embedded = self.dropout(self.embedding(inp)).unsqueeze(1)  # (batch_size, 1, embed_dim)
        # context: (batch_size, num_kernels)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)  # (batch_size, 1, embed + context)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))  # output: (batch_size, 1, num_kernels)
        output = self.fc_out(output.squeeze(1))  # (batch_size, odim)
        return output, hidden.squeeze(0)

class CNNSeq2Seq(XlitModel):
    def __init__(self, model_conf: dict, device: torch.device):
        super().__init__(model_conf, device)
        self.encoder = CNNEncoder(
            model_conf["idim"],
            model_conf["embed_dim"],
            model_conf["hidden_dim"],
            model_conf["kernel_size"],
            model_conf["dropout"],
        )

        self.decoder = CNNDecoder(
            model_conf["odim"],
            model_conf["embed_dim"],
            model_conf["hidden_dim"],
            model_conf["dropout"],
        )

        self.teacher_forcing_ratio = model_conf.get("teacher_forcing_ratio", 0.5)

    def forward(self, x, y=None, max_len=None):
        batch_size = x.size(0)
        max_len = max_len or self.max_len
        target_len = y.size(1) if y is not None else max_len
        y_vocab_size = self.decoder.fc_out.out_features

        encoder_output = self.encoder(x)  # (batch_size, seq_len, hidden_dim)
        context = encoder_output.mean(dim=1)  # global context

        outputs = torch.zeros(batch_size, target_len, y_vocab_size, device=self.device)

        inp = (
            y[:, 0] if y is not None else
            torch.full((batch_size,), self.sos_token, dtype=torch.long, device=self.device)
        )
        hidden = context  # use context as initial hidden state

        for t in range(1, target_len):
            output, hidden = self.decoder(inp, context, hidden)
            outputs[:, t] = output
            top1 = output.argmax(1)

            if y is not None:
                use_teacher = torch.rand(batch_size, device=self.device) < self.teacher_forcing_ratio
                inp = torch.where(use_teacher, y[:, t], top1)
            else:
                inp = top1
                if self.eos_token is not None and (inp == self.eos_token).all():
                    break

        return outputs
