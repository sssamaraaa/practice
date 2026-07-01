import torch


# =============================================== Задание 1: Создание и манипуляции с тензорами ===============================================
# 1.1
t1 = torch.rand(3, 4) # тензор 3х4, заполненный случайными числами из диапазона[0, 1)
t2 = torch.zeros(2,3,4) # тензор 2х3х4, заполненный нулями
t3 = torch.ones(5, 5) # тензор 3х4, заполненный единицами
t4 = torch.arange(16).reshape(4, 4) # тензор 4х4, заполненный числами от 0 до 15
print(t1, t2, t3, t4, sep="\n\n")


# 1.2
A = torch.arange(1, 13).reshape(3, 4)
B = torch.arange(10, 22).reshape(4, 3)
print(f"Транспонированная матрица A:\n{A.T}\n")
print(f"Матричное умножение A и B:\n{torch.matmul(A, B)}\n")
print(f"Элементное умножение A и B трансп.:\n{A * B.T}\n")
print(f"Сумма всех элементов A:\n{torch.sum(A)}\n")


# 1.3
C = torch.arange(1, 126).reshape(5, 5, 5)
print(f"Первая строка: {C[0]}")
print(f"Последний столбец: {C[:, :, -1]}")
print(f"Подматрица 2x2 из центра тензора: {C[2:4, 2:4, 2:4]}")
print(f"Все элементы с четными индексами: {C[::2, ::2, ::2]}")


# 1.4
D = torch.arange(1, 25)
print(f"Shape 2x12: {D.reshape(2, 12)}")
print(f"Shape 3x8: {D.reshape(3, 8)}")
print(f"Shape 4x6: {D.reshape(4, 6)}")
print(f"Shape 2x3x4: {D.reshape(2, 3, 4)}")
print(f"Shape 2x2x2x3: {D.reshape(2, 2, 2, 3)}")