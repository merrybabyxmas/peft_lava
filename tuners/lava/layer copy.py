import torch
import torch.nn as nn
from peft.tuners.tuners_utils import BaseTunerLayer


# ==========================================
#   VAE-Style Lava Adapter
# # ==========================================
# class LavaAdapter(nn.Module):
#     is_adapter = True

#     def __init__(self, hidden_size, rank):
#         super().__init__()
#         self.ln = nn.LayerNorm(hidden_size)

#         self.W_mu = nn.Linear(hidden_size, rank)
#         self.W_logvar = nn.Linear(hidden_size, rank)
#         self.W_o = nn.Linear(rank, hidden_size)

#         # nn.init.kaiming_uniform_(self.W_mu.weight)
#         # nn.init.kaiming_uniform_(self.W_logvar.weight)
#         # nn.init.zeros_(self.W_o.weight)


#         nn.init.xavier_uniform_(self.W_mu.weight)
#         nn.init.xavier_uniform_(self.W_logvar.weight)
#         nn.init.xavier_uniform_(self.W_o.weight)
        
#         # nn.init.zeros_(self.W_o.weight)

#         self.scale = nn.Parameter(torch.tensor(1e-2))

#     def forward(self, h):
#         if not isinstance(h, torch.Tensor):
#             return h

#         h = self.ln(h)
#         mu = self.W_mu(h)
#         logvar = self.W_logvar(h)

#         if self.training:
#             eps = torch.randn_like(mu)
#             z = mu + torch.exp(0.5 * logvar) * eps
#         else:
#             z = mu

#         delta = self.W_o(z) * self.scale
#         return h + delta


class LavaAdapter(nn.Module):
    is_adapter = True

    def __init__(self, hidden_size, rank):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)
        
        self.W_mu = nn.Linear(hidden_size, rank)
        self.W_logvar = nn.Linear(hidden_size, rank)
        self.W_o = nn.Linear(rank, hidden_size)

        nn.init.xavier_uniform_(self.W_mu.weight)
        nn.init.xavier_uniform_(self.W_logvar.weight)
        nn.init.xavier_uniform_(self.W_o.weight)

        self.scale = nn.Parameter(torch.tensor(1e-2))

    def forward(self, h):
        if not isinstance(h, torch.Tensor):
            return h

        raw_h = h
        h = self.ln(h)

        mu = self.W_mu(h)
        logvar = torch.clamp(self.W_logvar(h), -5, 5)

        if self.training:
            eps = torch.randn_like(mu)
            std = torch.exp(0.5 * logvar)
            z = mu + std * eps
        else:
            z = mu

        # delta WITHOUT LN (more stable)
        delta = self.W_o(z) * self.scale
        
        return raw_h + delta






# ==========================================
#        LavaLayer Wrapper for Linear
# ==========================================
class LavaLayer(BaseTunerLayer, nn.Module):
    is_adapter = True
    adapter_layer_names = ("lava",)

    def __init__(self, base_layer: nn.Linear, adapter_name: str, rank: int):
        nn.Module.__init__(self)
        BaseTunerLayer.__init__(self)

        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LavaLayer can only wrap nn.Linear.")

        self.base_layer = base_layer

        # freeze original linear
        for p in self.base_layer.parameters():
            p.requires_grad = False

        out_dim = base_layer.out_features

        # adapter dictionary
        self.lava = nn.ModuleDict({
            adapter_name: LavaAdapter(out_dim, rank)
        })

        # state
        self._active_adapters = [adapter_name]
        self._disable_adapters = False

    # ======================================================
    #           Adapter switching API
    # ======================================================
    def set_adapter(self, adapter_names):
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]
        self._active_adapters = adapter_names

    @property
    def active_adapters(self):
        return self._active_adapters

    @property
    def disable_adapters(self):
        return self._disable_adapters

    @disable_adapters.setter
    def disable_adapters(self, v: bool):
        self._disable_adapters = v

    # ======================================================
    #                Forward Pass
    # ======================================================
    def forward(self, x, *args, **kwargs):
        h = self.base_layer(x)

        if not isinstance(h, torch.Tensor):
            return h

        if self.disable_adapters:
            return h

        for name in self.active_adapters:
            adapter = self.lava[name]
            h = adapter(h)

        return h
