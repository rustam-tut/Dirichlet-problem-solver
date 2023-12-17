import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from typing import Callable, Union

from scipy.integrate import quad


class BaseSolver(ABC):

    def __init__(self, u: Callable, f: Callable, g: Callable, a1: float, a2: float, b1: float, b2: float):
        self.u = u
        self.f = f
        self.g = g
        self.a1 = a1
        self.a2 = a2
        self.b1 = b1
        self.b2 = b2
    
    @abstractmethod
    def solve(self):
        pass


class JacobiSolver(BaseSolver):

    def __init__(self, u: Callable, f: Callable, g: Callable, a1: float, a2: float, b1: float, b2: float, Nx=11, Ny=11, decr_ster_coef=1.5):
        super().__init__(u, f, g, a1, a2, b1, b2)
        self.Nx = Nx
        self.Ny = Ny
        self.decr_ster_coef = decr_ster_coef

    def __create_grid_function(self, func: Callable, nx, ny, hx, hy, sign=1.0):
        grid_func = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                grid_func[i, j] = sign * func(self.a1 + hx * i, self.b1 + hy * j)
        return grid_func
    
    def __create_U_init(self, nx, ny, hx, hy):
        U = np.zeros((nx, ny))
        x = np.linspace(self.a1, self.a2, nx)
        y = np.linspace(self.b1, self.b2, ny)
        for i in range(nx):
            U[i, 0] = self.g(x[i], self.b1)
            U[i, ny - 1] = self.g(x[i], self.b2)
        for j in range(ny):
            U[0, j] = self.g(self.a1, y[j])
            U[nx - 1, j] = self.g(self.a2, y[j])
        return U
    
    def __create_F_for_approx(self, nx, ny, hx, hy):
        F = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                yj = self.b1 + hy * j
                y1 = 0.5 * (2 * yj - hy)
                y2 = y1 + hy
                F[i, j] = (1 / hy) * \
                    quad(lambda y:  self.f(self.a1 + i * hx, y), y1, y2)[0]
        return F
    
    def __calc_h_grid(self, n, l, r):
        return abs((r - l) / float(n - 1))
    
    def solve(self, eps: float, show=True, info=True):
        nx = self.Nx
        ny = self.Ny
        error = 1000
        iterates = 0
        while True:
            hx = self.__calc_h_grid(nx, self.a1, self.a2)
            hy = self.__calc_h_grid(ny, self.b1, self.b2)
            U_k = self.__create_U_init(nx, ny, hx, hy)
            U_k1 = U_k.copy()
            U_ex = self.__create_grid_function(self.u, nx, ny, hx, hy)
            F = self.__create_grid_function(self.f, nx, ny, hx, hy)
            #F = self.__create_F_for_approx(nx, ny, hx, hy)
            c0 = 0.5 * hx ** 2 * hy ** 2 / (hx ** 2 + hy ** 2)
            c1 = 0.5 * hx ** 2 / (hx ** 2 + hy ** 2)
            c2 = 0.5 * hy ** 2 / (hx ** 2 + hy ** 2)
            iter_error = 1000
            iter_count = 0
            while (iter_error > 10e-15):
                iter_count += 1
                for i in range(1, nx - 1):
                    for j in range(1, ny - 1):
                        U_k1[i, j] = c0 * F[i, j] + c1 * (U_k[i, j - 1] + U_k[i, j + 1]) + c2 * (U_k[i - 1, j] + U_k[i + 1, j])
                iter_error = np.linalg.norm(U_k1 - U_k)
                U_k = U_k1.copy()
            error = np.linalg.norm(U_ex - U_k1)
            iterates += iter_count 
            if (info):
                print(f'Шаг по оси х: {hx}')
                print(f'Шаг по оси y: {hy}')
                print(f'Апостериорная погрешность: {iter_error}')
                print(f'Итераций для сетки на {nx * ny}: {iter_count}')
                print(f'Всего итераций: {iterates}')
                print(f'Погрешность точного и численного решений: {error}')
                print('------------------------------------------------------------\n')
            if (error < eps):
                break
            nx = int(nx * self.decr_ster_coef)
            ny = int(ny * self.decr_ster_coef)
        if (show):
            nodes = nx * ny
            create_plot_3d(self.u, self.a1, self.a2, 
                                self.b1, self.b2, 500, 500, 'Точное решение')
            create_plot_3d(self.u, self.a1, 
                                self.a2, self.b1, self.b2, nx, ny, f'Точное решение в расчетной сетке. {nodes} узлов.')
            create_plot_3d(U_k1, self.a1, 
                                self.a2, self.b1, self.b2, nx, ny, f'Численное решение в расчетной сетке. {nodes} узлов.')
        print(f'Для решения методом Якоби с точностью {eps} получили:')
        print(f'\t\tпогрешность точного и численного решения: {error}')
        print(f'\t\tитераций: {iterates}')


def create_plot_3d(func: Union[Callable, np.ndarray], x1, x2, y1, y2, m, n, title):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    x = np.linspace(x1, x2, m)
    y = np.linspace(y1, y2, n)
    x_m, y_m = np.meshgrid(x, y)
    z = None
    if isinstance(func, Callable):
        z = func(x_m, y_m)
    else:
        z = func
    plot_surf = ax.plot_surface(y_m, x_m, z, cmap='bwr')
    fig.colorbar(plot_surf, shrink=0.5, aspect=5)
    plt.title(title)
    plt.show()
            

        
        