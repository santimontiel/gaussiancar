import torch.nn as nn


class SegHead(nn.Module):
    def __init__(
        self,
        dim_last,
        multi_head,
        outputs,
    ):
        super().__init__()

        self.multi_head = multi_head
        self.outputs = outputs

        dim_total = 0
        dim_max = 0
        for _, (start, stop, _) in outputs.items():
            assert start < stop

            dim_total += stop - start
            dim_max = max(dim_max, stop)

        assert dim_max == dim_total
        if multi_head:
            layer_dict = {}
            for k, (start, stop, act_fn) in outputs.items():
                match act_fn:
                    case "relu":
                        act_fn = nn.ReLU(inplace=True)
                    case "identity":
                        act_fn = nn.Identity()
                    case _:
                        raise ValueError(f"Unsupported activation function: {act_fn}")
                layer_dict[k] = nn.Sequential(
                    nn.Conv2d(dim_last, dim_last, 3, padding=1, bias=False),
                    nn.BatchNorm2d(dim_last),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dim_last, stop - start, 1),
                    act_fn,
                )
            self.to_logits = nn.ModuleDict(layer_dict)
        else:
            self.to_logits = nn.Sequential(
                nn.Conv2d(dim_last, dim_last, 3, padding=1, bias=False),
                nn.BatchNorm2d(dim_last),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_last, dim_max, 1),
                nn.Identity(),
            )

    def forward(self, x):
        if self.multi_head:
            return {k: v(x) for k, v in self.to_logits.items()}
        else:
            x = self.to_logits(x)
            return {k: x[:, start:stop] for k, (start, stop) in self.outputs.items()}
