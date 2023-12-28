import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def task1():
    print("Задание 1, решить уравнение")
    a = 0.3
    b = -21.17
    y = np.power(a+1.5, 1/3) + np.power(a - b, 8) - b/np.arcsin(np.power(abs(a), 2))
    print("y =", y)
    pass

def task2():
    print("Задание 2, найти оценки уровня регрессии")

    N = 12
    X = np.column_stack((np.ones(12), np.arange(N, N+12), np.random.randint(60, 83, size=12)))
    Y = np.random.uniform(13.5, 18.6, size=12)

    XT = X.T
    XTX = np.dot(XT, X)
    XTX_obratn = np.linalg.inv(XTX)
    XTY = np.dot(XT, Y)

    A = np.dot(XTX_obratn, XTY)
    print("Вектор оценок А: ", A)

    Y_predict = A[0] + A[1]*X[:, 1] + A[2]*X[:, 2]

    print("Предсказанные значения Y", Y_predict)
    print("Исходные значения Y", Y)

    pass

def task3():
    print("Задание 3, работа с пандас")

    dataset = pd.read_csv("test.csv")
    thousFromDataset = dataset.head(1000)
    missing_values = thousFromDataset.isnull().sum()
    thousFromDataset.loc[:, "Rooms"] = thousFromDataset["Rooms"].fillna(thousFromDataset["Rooms"].mean())

    plt.boxplot(thousFromDataset["Rooms"].apply(np.log1p))
    plt.xlabel("Rooms")
    plt.ylabel("Value")
    plt.title("Boxplot")
    plt.show()

    plt.hist(thousFromDataset["Rooms"].apply(np.log1p), bins=20)
    plt.xlabel("Rooms")
    plt.ylabel("Frequency")
    plt.title("Histogram")

    thousFromDataset["Rooms"].fillna(thousFromDataset["Rooms"].mean(), inplace=True)

    thousFromDataset["Rooms"] = thousFromDataset["Rooms"].clip(lower=0, upper=100)

    room_counts = thousFromDataset["Rooms"].value_counts()

    pivot_table = pd.pivot_table(thousFromDataset, values="Id", index="DistrictId", columns="Rooms", aggfunc=len)

    dataset_processed = thousFromDataset.dropna()
    dataset_processed["column_name"] = dataset_processed["column_name"].clip(lower=0, upper=100)

    dataset_processed.to_csv("changed_csv.csv", index=False)
    pass

def task4():
    print("Задание 4, работа с матплотлибом")

    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)

    # Функция z = x + y^2
    def func1(x, y):
        return x + y ** 2

    # Функция z = -x + y^3 - 2y^2
    def func2(x, y):
        return -x + y ** 3 - 2 * y ** 2

    # Функция z = x + y^2 - x*y
    def func3(x, y):
        return x + y ** 2 - x * y

    # Функция z = x*y / (x + y)
    def func4(x, y):
        return x * y / (x + y)

    Z1 = func1(X, Y)
    Z2 = func2(X, Y)
    Z3 = func3(X, Y)
    Z4 = func4(X, Y)

    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')

    ax1.plot_surface(X, Y, Z1, cmap='viridis')
    ax1.set_title('z = x + y^2')

    ax2.plot_surface(X, Y, Z2, cmap='viridis')
    ax2.set_title('z = -x + y^3 - 2y^2')

    ax3.plot_surface(X, Y, Z3, cmap='viridis')
    ax3.set_title('z = x + y^2 - x*y')

    ax4.plot_surface(X, Y, Z4, cmap='viridis')
    ax4.set_title('z = x*y / (x + y)')

    plt.tight_layout()
    plt.show()
    pass

def task5():
    t = np.arange(2, 3.05, 0.05)

    f_x = np.log(np.abs(1.3 + t)) - np.exp(t)

    print("Аргументы:", t)
    print("Значения функции:", f_x)

    max_value = np.max(f_x)
    min_value = np.min(f_x)
    mean_value = np.mean(f_x)

    array_length = len(f_x)

    if array_length % 2 == 0:
        sorted_array = np.sort(f_x)[::-1]
    else:
        sorted_array = np.sort(f_x)

    plt.plot(t, f_x, 'o-', label='f(x)')
    plt.xlabel('Аргумент')
    plt.ylabel('Значение функции')
    plt.title('График функции f(x)')
    plt.grid(True)
    plt.legend()

    plt.axhline(y=mean_value, color='r', linestyle='--', label='Среднее значение')
    plt.legend()

    print("Наибольшее значение:", max_value)
    print("Наименьшее значение:", min_value)
    print("Среднее значение:", mean_value)
    print("Количество элементов массива:", array_length)

    plt.show()

while True:
    ch = int(input("Введите номер задания ЛР: "))
    if ch == 1:
        task1()
    elif ch == 2:
        task2()
    elif ch == 3:
        task3()
    elif ch == 4:
        task4()
    elif ch == 5:
        task5()
