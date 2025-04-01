import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)

lower_bound = 33.0936
upper_bound = 40.1418

random_tensor = torch.empty(41, 726, 31, 1047, device=device).uniform_(lower_bound, upper_bound)

random_tensor_bf16 = random_tensor.to(torch.bfloat16)

min_value = torch.min(random_tensor_bf16)

print(f"张量中的最小值: {min_value}")

print(random_tensor_bf16.shape)
print(random_tensor_bf16.dtype)
print(f"计算设备: {random_tensor_bf16.device}")