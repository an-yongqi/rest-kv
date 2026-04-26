import torch

def ema(input_tensor, dim, alpha=None):
    if alpha is None:
        alpha = 2. / (input_tensor.size(dim) + 1)
    # Initialize cumulative tensor with the first element along dim
    cumulative_tensor = input_tensor.select(dim, 0).unsqueeze(dim).clone()
    for i in range(1, input_tensor.size(dim)):
        current_slice = input_tensor.select(dim, i).unsqueeze(dim)
        cumulative_tensor = alpha * current_slice + (1 - alpha) * cumulative_tensor

    return cumulative_tensor.squeeze(dim)

# Test
def test_ema():
    # Input data
    input_tensor = torch.tensor([[[10.0, 20.0, 30.0, 40.0, 50.0]]])

    print("Input Tensor:")
    print(input_tensor.shape)

    # Compute EMA, returns only the last value
    result = ema(input_tensor, dim=2, alpha=0.9)
    
    print("\nFinal EMA Value:")
    print(result)

test_ema()
