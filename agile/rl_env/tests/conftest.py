import torch

# Isaac Lab 3.0 simulation values expose a ProxyArray `.torch` view. Unit-test
# mocks use ordinary tensors, so provide the same read-only view in this test tree.
if not hasattr(torch.Tensor, "torch"):
    torch.Tensor.torch = property(lambda self: self)
