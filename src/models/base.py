import torch
import torch.nn as nn

class XlitModel(nn.Module):
    def __init__(self, model_conf: dict, device: torch.device) -> None:
            super().__init__()
            self.device = device
            self.max_len = model_conf["max_len"]
            self.sos_token = model_conf["sos_token"]
            self.eos_token = model_conf["eos_token"]
            self.pad_token = model_conf["pad_token"]
            
