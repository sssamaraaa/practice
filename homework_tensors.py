import torch
import time


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


# =============================================== Задание 3: Сравнение производительности CPU vs CUDA ===============================================
# 3.1
F = torch.rand(64, 1024, 1024)
G = torch.rand(128, 512, 512)
H = torch.rand(256, 256, 256)


# 3.2
def check_time(device, func, T1, T2):
    """
    Замеряет время выполнения func(T1, T2) на указанном устройстве.
    Parameters:
        device : str ("cuda" или "cpu").
        T1, T2 : torch.Tensor (входные тензоры).
    Returns:
        float (время выполнения в секундах).
    Raises:
        ValueError (если device не "cpu" и не "cuda").
        TypeError (если T1 или T2 не являются torch.Tensor).
    """
    if not isinstance(T1, torch.Tensor) or not isinstance(T2, torch.Tensor):
        raise TypeError("T1 and T2 must be torch.Tensor objects.") 
    
    if device == "cuda":
        T1 = T1.to(device)
        T2 = T2.to(device)  
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        func(T1, T2)
        end_time.record()
        torch.cuda.synchronize()

        return start_time.elapsed_time(end_time) / 1000.0

    elif device == "cpu":
        start_time = time.time()
        func(T1, T2)
        end_time =time.time()

        return end_time - start_time

    else:
        raise ValueError("Invalid device. Use 'cpu' or 'cuda'.")
    

def transpose_op(T1, _):
    return T1.transpose(-1, -2)

def sum_op(T1, _):
    return torch.sum(T1)

# 3.3
operations = [
    ("Матричное умножение", torch.matmul),
    ("Сложение", torch.add),
    ("Поэлементное умножение", torch.mul),
    ("Транспонирование", transpose_op),
    ("Сумма элементов", sum_op),
]

tensors = [
    ("F (64x1024x1024)", F),
    ("G (128x512x512)", G),
    ("H (256x256x256)", H),
]

for tensor_name, T in tensors:
    print(f"\n{'='*64}")
    print(tensor_name)
    print(f"{'Операция':25} {'CPU (мс)':>12} {'GPU (мс)':>12} {'Ускорение':>12}")

    for op_name, op in operations:
        cpu_time = check_time("cpu", op, T, T)

        if torch.cuda.is_available():
            gpu_time = check_time("cuda", op, T, T)
            speedup = cpu_time / gpu_time
            gpu_ms = gpu_time * 1000
            speed = f"{speedup:.2f}x"
        else:
            gpu_ms = "-"
            speed = "-"

        print(
            f"{op_name:25}"
            f"{cpu_time*1000:12.2f}"
            f"{gpu_ms if gpu_ms=='-' else f'{gpu_ms:.2f}':>12}"
            f"{speed:>12}"
        )


# 3.4 
"""
РЕЗУЛЬТАТ:
================================================================
F (64x1024x1024)
Операция                      CPU (мс)     GPU (мс)    Ускорение
Матричное умножение            555.37       24.67      22.52x
Сложение                        71.19        6.70      10.62x
Поэлементное умножение          80.78        2.37      34.03x
Транспонирование                 0.00        0.00       0.00x
Сумма элементов                 10.97        1.19       9.23x

================================================================
G (128x512x512)
Операция                      CPU (мс)     GPU (мс)    Ускорение
Матричное умножение            193.27        4.24      45.57x
Сложение                        37.74        1.06      35.63x
Поэлементное умножение          30.80        1.07      28.73x
Транспонирование                 0.00        0.00       0.00x
Сумма элементов                  4.99        0.39      12.76x

================================================================
H (256x256x256)
Операция                      CPU (мс)     GPU (мс)    Ускорение
Матричное умножение             37.90        1.19      31.97x
Сложение                         8.98        0.57      15.68x
Поэлементное умножение           8.98        0.57      15.82x
Транспонирование                 0.00        0.00       0.00x
Сумма элементов                  3.99        0.23      17.29x

1. Наибольшее ускорение получила операция матричного умножения, потому что это самая вычислительно затратная операция, 
с которой gpu справлятся быстрее благодаря большому количеству параллельных вычислительных ядер. 
Выигрыш в ускорении операций поэлементного умножения и сложения меньше,
потому что эти операции менее ресурсоемкие, их скорость в основном ограничивается скоростью доступа к памяти. 
Тип девайса не влияет на скорость транспонирования, тк эта операция не затрагивает данные в тензоре, она только меняет его форму (не копирует данные, а создает новое представление).

2. Основная причина того, что иногда операции на GPU могут быть медленнее, чем на CPU, заключается в накладных расходах на передачу данных между CPU и GPU.
Для небольших матриц накладные расходы даже могут превышать выигрыш от параллельного вычисления.

3. В общем случае при увеличении объема вычислений GPU используется эффективнее, однако выигрыш зависит не только от размера,
но и от формы матриц, а также от выбранных библиотекой CUDA алгоритмов вычислений. Поэтому однозначно сказать, как влияют размеры матриц на ускорение - нельзя.

4. Передача данных между CPU и GPU - это, в первую очередь, физический процесс, поэтому он гораздо медленее, чем внутренние вычисления на CPU или GPU.
(В своем примере я не считала время передачи данных между CPU и GPU, это уже отдельный бенчмарк). Из-за этого, в пайплайнах принято ОДИН раз переносить данные на GPU 
(взяли батч - сразу перенесли на GPU - выполнили все вычисления (в общем виде)).
"""         