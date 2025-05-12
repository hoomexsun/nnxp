import torch
from torch import nn

from .base import XlitModel

class LSTMSeq2Seq(XlitModel):
    def __init__(self, model_conf: dict, device: torch.device):
        super().__init__(model_conf, device)
        self.embedding = nn.Embedding(model_conf["idim"], model_conf["embed_dim"])
        self.encoder = nn.LSTM(
            model_conf["embed_dim"],
            model_conf["hidden_dim"],
            num_layers=model_conf["nlayers"],
            dropout=model_conf["dropout"],
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            model_conf["embed_dim"],
            model_conf["hidden_dim"],
            num_layers=model_conf["nlayers"],
            dropout=model_conf["dropout"],
            batch_first=True,
        )
        self.output_layer = nn.Linear(model_conf["hidden_dim"], model_conf["odim"])
        self.dropout = nn.Dropout(model_conf["dropout"])
        self.teacher_forcing_ratio = model_conf.get("teacher_forcing_ratio", 0.5)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, max_len: int = None):
        batch_size = x.size(0)
        embedded = self.dropout(self.embedding(x))
        _, (hidden, cell) = self.encoder(embedded)

        if y is not None:
            target_len = y.size(1)
            outputs = torch.zeros(batch_size, target_len, self.output_layer.out_features, device=self.device)
            inp = y[:, 0]
        else:
            target_len = max_len or self.max_len
            assert target_len is not None, "max_len must be provided when y is None"
            outputs = torch.zeros(batch_size, target_len, self.output_layer.out_features, device=self.device)
            inp = torch.full((batch_size,), self.sos_token, dtype=torch.long, device=self.device)

        for t in range(1, target_len):
            embedded_input = self.dropout(self.embedding(inp)).unsqueeze(1)
            output, (hidden, cell) = self.decoder(embedded_input, (hidden, cell))
            output = self.output_layer(output.squeeze(1))
            outputs[:, t] = output

            top1 = output.argmax(1)

            if y is not None and torch.rand(1).item() < self.teacher_forcing_ratio:
                inp = y[:, t]
            else:
                inp = top1

        return outputs
