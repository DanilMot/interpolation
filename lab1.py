import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
class AutoDiffNum:
    def __init__(self, _re: float, _im: float):
        self._re = _re
        self._im = _im

    def __add__(self, dual):
        if isinstance(dual, AutoDiffNum):
            return AutoDiffNum(self._re + dual._re, self._im + dual._im)
        else:
            return AutoDiffNum(self._re + dual, self._im)

    def __sub__(self, dual):
        if isinstance(dual, AutoDiffNum):
            return AutoDiffNum(self._re - dual._re, self._im - dual._im)
        else:
            return AutoDiffNum(self._re - dual, self._im)

    def __mul__(self, dual):
        if isinstance(dual, AutoDiffNum):
            return AutoDiffNum(self._re * dual._re, self._re * dual._im + self._im * dual._im)
        else:
            return AutoDiffNum(self._re * dual, self._im * dual)

    def __pow__(self, n):
        return AutoDiffNum(self._re**n, n * self._re**(n-1) * self._im)

    def imag(self):
        return self._im

def task_2(filename: str):
    arr = [x for x in np.loadtxt(filename)]
    plt.scatter([x[0] for x in arr], [y[1] for y in arr])
    plt.xlabel('Ось Х')
    plt.ylabel('Ось Y')
    plt.show()
    return arr


def new_sparse_set(array: list, m: int):
    sparse_set = array[0::m]
    return sparse_set


def get_coef(h: int, arr: list):
    a = [k for k in arr]

    # задаем матрицу со строгим диагональным преобладанием, матрица A
    matrix_A = np.zeros((len(arr), len(arr)))
    matrix_A[0][0] = 1
    matrix_A[-1][-1] = 1

    for i in range(1, len(arr) - 1):
        matrix_A[i][i - 1] = h
        matrix_A[i][i] = 2 * (h+h)
        matrix_A[i][i + 1] = h
    # задаем матрицу b
    matrix_B = np.array([0] + [(3 * (a[i + 2] - a[i + 1]) / h) - (3 * (a[i + 1] - a[i]) / h)
                               for i in range(0, len(arr) - 2)] + [0])

    # Решение СЛАУ методом прогонки
    n = len(arr)
    P = np.zeros(n)
    Q = np.zeros(n)
    # Начальные значения
    P[0] = -matrix_A[0, 1] / matrix_A[0, 0]
    Q[0] = matrix_B[0] / matrix_A[0, 0]
    # Прямой ход
    for i in range(1, n-1):
        denominator = matrix_A[i, i] + matrix_A[i, i - 1] * P[i - 1]
        P[i] = -matrix_A[i, i + 1] / denominator
        Q[i] = (matrix_B[i] - matrix_A[i, i - 1] * Q[i - 1]) / denominator
    # Конечные значения
    P[n-1] = 0
    Q[n-1] = (matrix_A[n-1][n-2] * P[n-2] - matrix_B[n-1]) / (-matrix_A[n-1][n-1] - matrix_A[n-1][n-2] * P[n-2])
    matrix_c = np.zeros(n)
    matrix_c[n - 1] = Q[n - 1]
    # Обратная прогонка
    for i in range(n - 2, -1, -1):
        matrix_c[i] = P[i] * matrix_c[i + 1] + Q[i]

    d = [(matrix_c[i + 1] - matrix_c[i]) / (3 * h) for i in range(0, len(arr) - 1)] + [0]
    b = [1 / h * (a[i + 1] - a[i]) - h / 3 * (matrix_c[i + 1] + 2 * matrix_c[i])
         for i in range(0, len(arr) - 1)] + [0]
    all = np.transpose(np.vstack((a, b, matrix_c, d)))
    return all



def spline_points(coef: np.ndarray, n: int, m: int, h: float = 1):
    res = []
    for i in range(n):
        splain = coef[i // m][0] + coef[i // m][1] * (h * (i % m)) + coef[i // m][2] * (h * (i % m)) ** 2 + \
                 coef[i // m][3] * (h * (i % m)) ** 3
        res.append(splain)
    return res


def distance(x_i: list, y_i: list, xall: list, yall: list):
    n = len(xall)
    distance = np.zeros(n)
    for i in range(n):
        distance[i] = sqrt((xall[i] - x_i[i]) ** 2 + (yall[i] - y_i[i]) ** 2)
    return distance

def result_print(filename: str, data: np.ndarray):
    with open(filename, 'w') as f:
        f.write('a1\ta2\ta3\ta4\tb1\tb2\tb3\tb4\n')
        for s in data:
            data = '\t'.join(str(coef) for coef in s)
            f.write(data + '\n')

def result_plot_spline(spline_x: list, spline_y: list, x_r: list, y_r: list, x: list, y: list):
    fig, ax = plt.subplots(1, 1, figsize=(22, 22))
    ax.plot(spline_x, spline_y, linewidth=1, color='steelblue', label=f'Полученный сплайн, частый шаг t = 0.1')
    ax.scatter(x_r, y_r, s=0.5, color='red', label='Исходное множество точек P')
    ax.scatter(x, y, s=6, color='black', label=r'Прореженное множество точек $\hat P$')
    ax.legend()
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    plt.axis('equal')
    plt.grid()
    plt.show()

def diff(coef: np.ndarray):
    deriv = []
    for i in range(len(coef) - 1):
        a = AutoDiffNum(coef[i][0],0)
        b = AutoDiffNum(coef[i][1],0)
        c = AutoDiffNum(coef[i][2],0)
        d = AutoDiffNum(coef[i][3],0)
        t = AutoDiffNum(i, 1)
        t_i = AutoDiffNum(i,0)
        spline = a + b * (t - t_i) + c * (t - t_i)**2 + d * (t - t_i)**3
        deriv.append(spline.imag())
    return deriv

def show_tangents_and_normals(xall: list, yall: list, Gx: list, Gy: list, Rx: list, Ry: list, spline_x: list, spline_y: list):
    fig, ax = plt.subplots(1, 1, figsize=(22, 22))
    for i in range(len(Gx)):
        if i == 0:
            ax.plot([xall[i], xall[i] + 5 * Gx[i]], [yall[i], yall[i] + 10 * Gy[i]], linewidth=1, color='red',
                    label=r'Векторы первых производных $G(t)$ = $\frac{d}{dx}(\widetilde{x}, \widetilde{y})$')
            ax.plot([xall[i], xall[i] + 1 * Rx[i]], [yall[i], yall[i] + 1 * Ry[i]], linewidth=1,
                    color='black', label=r'Векторы нормалей $R(t)$ к $G(t)$')
        elif i % 2 == 0:
            ax.plot([xall[i], xall[i] + 5 * Gx[i]], [yall[i], yall[i] + 10 * Gy[i]],linewidth=1, color='red')
            ax.plot([xall[i], xall[i] + 1 * Rx[i]], [yall[i], yall[i] + 1 * Ry[i]],linewidth=1,color='black')
    ax.plot(spline_x, spline_y, linewidth=1, color='steelblue', label=f'Полученный сплайн, частый шаг t = 0.1')
    ax.legend()
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    plt.axis('equal')
    plt.show()


def lab1_advanced(sparse_x: list, sparse_y: list, coef_x: np.ndarray, coef_y: np.ndarray,new_spline_x: list,new_spline_y: list):
    Gx = diff(coef_x)
    Gy = diff(coef_y)
    coeffs = [-Gy[i] / Gx[i] for i in range(len(Gx))]
    Ry = [0.0001 for _ in range(len(Gx))]
    Rx = [coeffs[i] * Ry[i] for i in range(len(Ry))]

    for i in range(len(Gx)):
        length = sqrt(Rx[i] ** 2 + Ry[i] ** 2)
        Rx[i] = Rx[i] / (length * 5000)
        Ry[i] = Ry[i] / (length * 5000)
    show_tangents_and_normals(sparse_x, sparse_y, Gx, Gy, Rx, Ry,new_spline_x,new_spline_y)

def lab1_base(filename_in: str, factor: int, filename_out: str):
    # №2
    points = task_2(filename_in)
    start_x = [point[0] for point in points]
    start_y = [point[1] for point in points]

    # №3
    sparse_points = new_sparse_set(points, factor)
    sparse_x = [sparse_point[0] for sparse_point in sparse_points]
    sparse_y = [sparse_point[1] for sparse_point in sparse_points]

    # №4
    coef_x = get_coef(factor, sparse_x)
    coef_y = get_coef(factor, sparse_y)

    # №5
    x_for_distance = spline_points(coef_x, len(start_x), factor)
    y_for_distance = spline_points(coef_y, len(start_y), factor)
    distances = distance(x_for_distance, y_for_distance, start_x, start_y)
    print('Стандартное отклонение: {:.10f}'.format(distances.std()))
    print('Среднее отклонение: {:.10f}'.format(distances.mean()))

    # №6
    new_spline_x = spline_points(coef_x, len(start_x) * 10, factor * 10, h=0.1)
    new_spline_y = spline_points(coef_y, len(start_y) * 10, factor * 10, h=0.1)
    result_plot_spline(new_spline_x,new_spline_y,start_x,start_y,sparse_x,sparse_y)
    # №7
    result_print(filename_out, np.hstack((coef_x, coef_y)))
    # №8-11
    lab1_advanced(sparse_x,sparse_y,coef_x,coef_y,new_spline_x,new_spline_y)

if __name__ == '__main__':
    lab1_base('contours.txt', 10, 'coeffs.txt')