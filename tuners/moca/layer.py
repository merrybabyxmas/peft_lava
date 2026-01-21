import torch
import torch.nn as nn
from peft.tuners.tuners_utils import BaseTunerLayer


class MoCAAdapter(nn.Module):
    """
    VAE-style MoCA adapter.
    Input  : Tensor (..., D)
    Output : Tensor (..., D)
    Never modifies the output type (LoRA-style safety).
    """
    def __init__(self, hidden_size, rank):
        super().__init__()
        self.D = hidden_size
        self.r = rank

        self.ln = nn.LayerNorm(hidden_size)

        # VAE-style projections
        self.W_mu = nn.Linear(hidden_size, rank, bias=False)
        self.W_logvar = nn.Linear(hidden_size, rank, bias=False)

        # reconstruction
        self.W_o = nn.Linear(rank, hidden_size, bias=False)

        self.scale = nn.Parameter(torch.tensor(1e-3))

        # init
        nn.init.kaiming_uniform_(self.W_mu.weight, a=0)
        nn.init.kaiming_uniform_(self.W_logvar.weight, a=0)
        nn.init.zeros_(self.W_o.weight)

    def forward(self, h):
        if not isinstance(h, torch.Tensor):
            return h

        h_norm = self.ln(h)

        mu = self.W_mu(h_norm)
        logvar = self.W_logvar(h_norm)

        if self.training:
            eps = torch.randn_like(mu)
            z = mu + torch.exp(0.5 * logvar) * eps
        else:
            z = mu

        delta = self.W_o(z) * self.scale

        return h + delta



class MoCALayer(BaseTunerLayer, nn.Module):
    adapter_layer_names = ("moca",)

    def __init__(self, base_layer, adapter_name, hidden_size, rank):
        nn.Module.__init__(self)
        BaseTunerLayer.__init__(self)

        if not isinstance(base_layer, nn.Linear):
            raise TypeError("MoCALayer can ONLY wrap nn.Linear layers.")

        self.base_layer = base_layer

        # freeze base layer parameters
        for p in self.base_layer.parameters():
            p.requires_grad = False

        # adapter
        self.moca = nn.ModuleDict({
            adapter_name: MoCAAdapter(hidden_size, rank)
        })

        self._active_adapter = adapter_name

        # MUST USE THIS:
        self._disable_adapters = False     # ⭕ correct

    def forward(self, x, *args, **kwargs):
        # base forward
        h = self.base_layer(x)

        # preserve LoRA safety: h must stay Tensor
        if not isinstance(h, torch.Tensor):
            return h

        if self.disable_adapters:
            return h

        # apply adapters
        for name in self.active_adapters:
            adapter = self.moca[name]
            h = adapter(h)

        return h
    
    def set_adapter(self, adapter_names):
        self._active_adapter = adapter_names[0] if isinstance(adapter_names, list) else adapter_names

    @property
    def active_adapters(self):
        return [self._active_adapter]
