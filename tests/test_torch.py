import torch

print(f"PyTorch version: {torch.__version__}")

# check if MPS (Metal Performance Shaders) is available (for M1/M2/M3/M4 Macs)
if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
    print(f"MPS available: {torch.mps.is_available()}")
    if torch.mps.is_available():
        print("You can use Apple Silicon GPU acceleration!")
        print("Device: mps")
else:
    print("Running on CPU only (standard Mac configuration)")

# test the basic tensor operations
print("\nTesting basic tensor operations:")
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print(f"a + b = {a + b}")
print(f"a * b = {a * b}")

# test creating random matrix and computing its dot product
print("\nTesting matrix operations:")
m1 = torch.randn(3, 4)
m2 = torch.randn(4, 2)
print(f"Matrix m1 shape: {m1.shape}")
print(f"Matrix m2 shape: {m2.shape}")
m3 = torch.matmul(m1, m2)
print(f"m1 @ m2 shape: {m3.shape}")
print(f"m1 @ m2 = \n{m3}")

print("\nPyTorch is working correctly!")
