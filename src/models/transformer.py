import torch
import torch.nn as nn

from .base import XlitModel
from .positional_encoding import PositionalEncoding
    

class TransformerSeq2Seq(XlitModel):
    def __init__(self, model_conf: dict, device: torch.device):
        super().__init__(model_conf, device)

        self.embed_dim = model_conf["embed_dim"]
        self.encoder_embed = nn.Embedding(model_conf["idim"], self.embed_dim)
        self.decoder_embed = nn.Embedding(model_conf["odim"], self.embed_dim)

        self.pos_encoder = PositionalEncoding(self.embed_dim)
        self.pos_decoder = PositionalEncoding(self.embed_dim)

        self.transformer = nn.Transformer(
            d_model=self.embed_dim,
            nhead=model_conf["num_heads"],
            num_encoder_layers=model_conf["num_encoder_layers"],
            num_decoder_layers=model_conf["num_decoder_layers"],
            dim_feedforward=model_conf["dim_feedforward"],
            dropout=model_conf["dropout"],
            batch_first=True,
        )

        self.generator = nn.Linear(self.embed_dim, model_conf["odim"])
        self.dropout = nn.Dropout(model_conf["dropout"])
        self.teacher_forcing_ratio = model_conf.get("teacher_forcing_ratio", 0.5)

    def forward(self, src, tgt=None, max_len=None):
        batch_size, src_len = src.shape
        max_len = max_len or self.max_len
        tgt_len = tgt.size(1) if tgt is not None else max_len

        src_mask = None
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len).to(self.device)

        src_emb = self.dropout(self.pos_encoder(self.encoder_embed(src)))
        memory = self.transformer.encoder(src_emb, mask=src_mask)

        outputs = torch.zeros(batch_size, tgt_len, self.generator.out_features).to(self.device)

        ys = (
            tgt[:, 0] if tgt is not None else
            torch.full((batch_size,), self.sos_token, dtype=torch.long, device=self.device)
        )
        ys = ys.unsqueeze(1)

        for t in range(1, tgt_len):
            tgt_emb = self.dropout(self.pos_decoder(self.decoder_embed(ys)))
            out = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask[:t, :t])
            output = self.generator(out[:, -1])
            outputs[:, t] = output
            top1 = output.argmax(1).unsqueeze(1)

            if tgt is not None:
                use_teacher = torch.rand(batch_size, device=self.device) < self.teacher_forcing_ratio
                next_input = torch.where(use_teacher, tgt[:, t], top1.squeeze(1))
            else:
                next_input = top1.squeeze(1)
                if self.eos_token is not None and (next_input == self.eos_token).all():
                    break

            ys = torch.cat([ys, next_input.unsqueeze(1)], dim=1)

        return outputs
