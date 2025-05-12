from typing import Optional
import torch
from torch import nn


from src.models.base import XlitModel


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.rnn = nn.LSTM(
            embed_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, hidden, cell


class Attention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len, _ = encoder_outputs.size()
        hidden = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        combined = torch.cat([hidden, encoder_outputs], dim=2)
        energy = torch.tanh(self.attn(combined))
        attention = self.v(energy).squeeze(2)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.rnn = nn.LSTM(
            embed_dim + hidden_dim,
            hidden_dim,
            num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hidden_dim)

    def forward(
        self,
        inp: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        inp = inp.unsqueeze(1)  # (batch_size, 1)
        embedded = self.dropout(self.embedding(inp))  # (batch_size, 1, embed_dim)
        # (batch_size, seq_len)
        attn_weights = self.attention(hidden, encoder_outputs, mask)
        # (batch_size, 1, hidden_dim)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        # (batch_size, 1, embed+hidden)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.fc_out(
            torch.cat((output.squeeze(1), context.squeeze(1)), dim=1)
        )  # (batch_size, output_dim)
        return prediction, hidden, cell


class AttnSeq2Seq(XlitModel):
    def __init__(self, model_conf: dict, device: torch.device):
        super().__init__(model_conf, device)
        self.encoder = Encoder(
            model_conf["idim"],
            model_conf["embed_dim"],
            model_conf["hidden_dim"],
            model_conf["elayers"],
            model_conf["dropout"],
        ).to(device)

        self.decoder = Decoder(
            model_conf["odim"],
            model_conf["embed_dim"],
            model_conf["hidden_dim"],
            model_conf["dlayers"],
            model_conf["dropout"],
        ).to(device)

        self.teacher_forcing_ratio = model_conf.get("teacher_forcing_ratio", 0.5)

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:

        max_len = max_len or self.max_len
        batch_size = x.size(0)
        y_vocab_size = self.decoder.fc_out.out_features
        target_len = y.size(1) if y is not None else max_len

        assert target_len is not None, "max_len must be provided when y is None"

        mask = x != self.pad_token  # (batch_size, seq_len)
        encoder_outputs, hidden, cell = self.encoder(x)
        outputs = torch.zeros(batch_size, target_len, y_vocab_size, device=self.device)

        inp = (
            y[:, 0]
            if y is not None
            else torch.full(
                (batch_size,), self.sos_token, dtype=torch.long, device=self.device
            )
        )

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(
                inp, hidden, cell, encoder_outputs, mask
            )
            outputs[:, t] = output
            top1 = output.argmax(1)

            if y is not None:
                teacher_force = (
                    torch.rand(1, device=self.device) < self.teacher_forcing_ratio
                )
                inp = torch.where(
                    teacher_force.unsqueeze(1), y[:, t].unsqueeze(1), top1.unsqueeze(1)
                ).squeeze(1)
            else:
                inp = top1
                if self.eos_token is not None and (inp == self.eos_token).all():
                    break

        return outputs
