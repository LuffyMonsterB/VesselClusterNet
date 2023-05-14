import torch

x = torch.randn(3, 4, 5)
x_list = x.tolist()
for i in range(len(x_list)):
    sub_x = torch.tensor(x_list[i])
    print(sub_x.shape)