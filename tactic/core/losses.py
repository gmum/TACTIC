import torch
import einops

class CrossEntropyLossExtraBatch(torch.nn.CrossEntropyLoss):
    def __init__(self, anom_weight, *args, **kwargs):
        assert 0 < anom_weight < 1
        weight = torch.tensor([1-anom_weight, anom_weight]).to("cuda")
        super().__init__(weight=weight, *args, **kwargs)

    def forward(self, input, target):
        """
        Input has shape (batch_size, num_samples, num_classes)
        Target has shape (batch_size, num_samples)

        Compared to the original CrossEntropyLoss, accepts (batch_size, num_samples) as batch
        """

        input = einops.rearrange(input, 'b s c -> (b s) c')
        target = einops.rearrange(target, 'b s -> (b s)')

        return super().forward(input, target)



