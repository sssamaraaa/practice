import torch


# =============================================== Задание 2: Автоматическое дифференцирование ===============================================
# 2.1
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = torch.tensor(3.0, requires_grad=True)
f = x**2 + y**2 + z**2 + 2*x*y*z
print(f"f(x, y, z) = {f.item()}") # item возвращает скаляр, без item вернется тензор tensor(26., grad_fn=<AddBackward0>)
f.backward() 
print(f"df/dx = {x.grad.item()}, df/dy = {y.grad.item()}, df/dz = {z.grad.item()}")

"""
Аналитическое решение: 
df/dx = 2*x + 2*y*z = 2*1 + 2*2*3 = 14,
df/dy = 2*y + 2*x*z = 2*2 + 2*1*3 = 10,
df/dz = 2*z + 2*x*y = 2*3 + 2*1*2 = 10.
Значения совпадают с результатами, полученными с помощью PyTorch.
"""


# 2.2
X = torch.tensor([1.5, 2.5, 3.5], requires_grad=False)
W = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)
y_true = torch.tensor([1.0, 2.0, 3.0])
y_pred = W * X + b
MSE = torch.mean((y_true - y_pred) ** 2)
print(f"MSE = {MSE.item()}")
gradients = torch.autograd.grad(MSE, [W, b])
print(f"df/dW = {gradients[0].item()}, df/db = {gradients[1].item()}")


# 2.3
E = torch.tensor(2.0, requires_grad=True)
func1 = E**2 + 1
func2 = torch.sin(func1)
func2.backward(retain_graph=True) # НЕ очищаем граф для последующего вычисления градиента через autograd
print(f"df/dE = {E.grad.item()}")
print(f"df/dE (через autograd) = {torch.autograd.grad(func2, E)[0].item()}") 