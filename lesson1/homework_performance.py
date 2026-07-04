import torch
import time


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