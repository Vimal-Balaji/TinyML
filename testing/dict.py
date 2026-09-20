import torch

data = torch.load("student_int8.pth", map_location="cpu")

print("Keys in the state_dict:", data["stem.0.weight"][1])
