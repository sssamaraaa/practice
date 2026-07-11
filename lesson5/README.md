# Задание 1 (стандартные аугментации)

### Результат применения аугментаций
![alt text](results/task1/Гароу.png)
![alt text](results/task1/Генос.png)
![alt text](results/task1/Сайтама.png)
![alt text](results/task1/Соник.png)
![alt text](results/task1/Татсумаки.png)

# Задание 2 (кастомные аугментации)

### Результат применения аументаций
![alt text](results/task2/Гароу.png)
![alt text](results/task2/Генос.png)
![alt text](results/task2/Сайтама.png)
![alt text](results/task2/Соник.png)
![alt text](results/task2/Татсумаки.png)

# Задание 3 (анализ датасета)

### Количество изображений по персонажам

| Персонаж | Количество | 
|----------|-----------|
| Гароу | 30 | 
| Генос | 30 | 
| Сайтама | 30 | 
| Соник | 30 | 
| Татсумаки | 30 | 
| Фубуки | 30 | 

### Размеры изображений

| Параметр | Значение | 
|----------|---------|
| Минимальная ширина | 210 | 
| Максимальная ширина | 736 | 
| Средняя ширина | 538.89 | 
| Минимальная высота | 240 | 
| Максимальная высота | 1308 | 
| Средняя высота | 623.56 | 

### Графики

|                                   | График    | 
|-----------------------------------|-----------|
| Гистограмма по классам            | ![alt text](results/task3/class_distribution.png)          |
| Распределение размеров изображений|     ![alt text](results/task3/image_sizes.png)      |

# Задание 4 (пайплайн аугментаций)

### Light
![alt text](results/task4/light/Гароу.png)
![alt text](results/task4/light/Генос.png)
![alt text](results/task4/light/Сайтама.png)
![alt text](results/task4/light/Соник.png)
![alt text](results/task4/light/Татсумаки.png)

### Medium
![alt text](results/task4/medium/Гароу.png)
![alt text](results/task4/medium/medium/Генос.png)
![alt text](results/task4/medium/Сайтама.png)
![alt text](results/task4/medium/Соник.png)
![alt text](results/task4/medium/Татсумаки.png)

### Heavy
![alt text](results/task4/heavy/Гароу.png)
![alt text](results/task4/heavy/Генос.png)
![alt text](results/task4/heavy/Сайтама.png)
![alt text](results/task4/heavy/Соник.png)
![alt text](results/task4/heavy/Татсумаки.png)

# Задание 5 (эксперимент с размерами)

### Графики

|                                   | График    | 
|-----------------------------------|-----------|
| Память           |    ![alt text](results/task5/memory.png)      |
| Время|    ![alt text](results/task5/time.png)       |

# Задание 6 (дообучение ResNet18)

### График обучения

![alt text](results/task6/training_history.png)

### Лучшие показатели (10 эпоха)

| Метрика | Значение |
|---------|---------:|
| Train Accuracy | 0.9889 |
| Validation Accuracy | 0.8517 |
| Train Loss | 0.0541 |
| Validation Loss | 0.5411 |