import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.special import gamma
from scipy.stats import laplace
from datetime import datetime
from scipy.optimize import minimize
import sys

x0 = {
    (0, 1): {'a': -10, 'b': 15},
    (3, 2.5): {'a': -10, 'b': 15},
    (10, 0.4): {'a': -4, 'b': 20}
}

output = open('output.txt', 'w')

def read_param(path):
    input = []
    with open(path) as file:
        for line in file:
            input.append(float(line))
    return input

def convert_param(path):
    param = {}
    input = []
    input = read_param(path)
    if(len(input) != 3):
        print('Wrong input value')
        raise Exception
    else:
        param = {'m' : input[0],
                 'v' : input[1],
                 'alpha':input[2],
                }
    return param

def create_freq(N, param):
    x = np.zeros(N)
    for i in range(1, N - 1):
        x[i + 1] = (param['a'] * x[i] + param['b'] *
                    (x[i - 1]**2) + param['c']) % param['m']
    return x

def calc_period(x):
    x1 = x[999]
    x2 = x[998]
    T = 2
    x_new = []
    for i in range(len(x) - 3, 1, -1):
        if (x[i] != x1):
            T += 1
        else:
            if (x[i - 1] != x2):
                T += 1
            else:
                break
    if (T > 100):
        for j in range(i, len(x) - 1):
            x_new.append(x[j])
    return T, x_new

def create_table(df):
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    fig.tight_layout()

# Математическое ожидание
def mo(x):
    m = 0
    for i in range(len(x)):
        m += x[i]
    m /= len(x)
    return m

# Дисперсия
def disp(x, m):
    d = 0
    for i in range(len(x)):
        d += (x[i] - m)**2
    d /= (len(x) - 1)
    return d

# функция плотности распределения
def lapl_pdf(x, param):
    m = param['m']
    v = param['v']
    return (1 / 2*v) * math.exp(-(m - x)/ v)


# Определить границы a,b для заданного уровня точности
#   @pdf - плотность функции распределения
#   @cdf - функция распределения
#   @params - параметры распределения
def find_bounds(pdf, cdf, params, precision=0.98):
    mu, var = params
    # Точка a
    a = minimize(fun=lambda x: -pdf(x[0], mu, var),
                   x0=np.array([x0[(mu, var)]['a']]),
                   tol=1e-6,
                   constraints={'type': 'eq', 'fun': lambda x: cdf(x[0], mu, var) - (1-precision)/2}).x[0]

    # Точка b
    b = minimize(fun=lambda x: -pdf(x[0], mu, var),
                   x0=np.array([x0[(mu, var)]['b']]),
                   tol=1e-6,
                   constraints={'type': 'eq', 'fun': lambda x: cdf(x[0], mu, var) - (1+precision)/2}).x[0]

    return [a, b]


# Определить M на интервале
#   @pdf - плотность функции распределения
#   @params - параметры распределения
#   @x0 - начальное рпиближение
def fin_max_pdf(pdf, params, x0):
    mu, var = params

    x_max = minimize(fun=lambda x: -pdf(x[0], mu, var),
                   x0=np.array([x0]),
                   tol=1e-6).x[0]

    return pdf(x_max, mu, var)


# Построение графика плотности распределения
#   @pdf - плотность функции распределения
#   @params - параметры распределения
def plot_pdf(pdf, params, pointx=[], pointy=[]):
    mu, var = params

    # Точки графика
    x = np.arange(x0[(mu, var)]['a'] - 1, x0[(mu, var)]['b'] + 1.001, 0.001)
    y = [pdf(point, mu, var) for point in x]

    # Создание полотна
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    fig.suptitle('Плотность распределения Лапласа с параметрами: \u03BC = {0}; \u03C3 = {1}'.format(mu, var))

    # Ограничение значений осей
    ax.set_xlim(min(x), max(x))
    # plt.plot(pointx, pointy)

    # Cетка
    ax.grid()
    ax.plot(x, y, linewidth=3)

    plt.show()


# Функция отрисовки графиков теоретической функции плотности и эмпирической на одном по-лотне
#   @pdf - теоретическая функция плотности распределения
#   @params - параметры распределения
#   @sample - выборка
#   @interval - интервал моделирования (a,b)
def plot_pdf_theoretical_and_empirical(pdf, params, sample, interval):
    n = len(sample)
    mu, var = params
    a, b = interval[0], interval[1]

    x = np.arange(x0[(mu, var)]['a']-1, x0[(mu, var)]['b']+1.001, 0.001)
    # Теоретические значения
    y = [pdf(point, mu, var) for point in x]

    # Число интервалов
    K = int(5 * np.log10(n))

    # Эмпирические значения
    weights, bins_edges = np.histogram(sample, bins=K, range=(a, b))
    # weights, bins_edges = np.histogram(x, bins=K, range=(a, b))
    # Число попаданий переведем в высоты столбца гистограммы
    weights = np.array(weights) / n
    for i in range(len(weights)):
        weights[i] /= (bins_edges[i+1]-bins_edges[i])

    # Создание полотна
    fig = plt.figure(figsize=(8, 4))
    fig.suptitle('Распределение Лапласа с параметрами: \u03BC = {0}; \u03C3 = {1}\nДлина по-следовательности N = {2}'\
                 .format(mu, var, len(sample)))

    ax = fig.add_subplot(111)

    # Ограничение значений осей
    # ax.set_xlim(min(bins_edges), max(bins_edges))
    # ax.set_ylim(-0.5, 1)
    ax.set_xlim(min(x), max(x))
    # Cетка
    ax.grid()

    # Вывод эмпирической функции
    ax.hist(bins_edges[:-1], bins_edges, weights=weights)
    # Вывод теоретической функции
    ax.plot(x, y, linewidth=3)
    # Значения оси Ox
    # plt.xticks(bins_edges)
    # ax.tick_params(axis='x', rotation=90)
    plt.show()


# Проверка гипотеза о согласии смоделированной выборки с теоретическим распределением
#   @sample - выборка
#   @cdf - теоретическая функция распределения
#   @params - параметры теоретического распределения
#   @interval - интервал моделирования (a,b)
def chi_square_test(sample, cdf, params, interval, alpha=0.05):
    # Длина последовательности
    n = len(sample)
    mu, v = params
    a, b = interval
    # Число интервалов
    K = int(5*np.log10(n))

    # Частоты попадания в i-интервал
    weights, bins_edges = np.histogram(sample, bins=K, range=(a, b))
    weights = np.array(weights) / n

    # Значение статистики
    S = 0
    for i in range(K):
        P = cdf(bins_edges[i+1], mu, v) - cdf(bins_edges[i], mu, v)
        S += pow(weights[i] - P, 2) / P
    S *= n

    # Критическое значение статистики
    S_crit = stats.chi2.ppf(1-alpha, K-1)

    print('Критерий \u03c7\u00b2 Пирсона', file=output )
    print('Число интервалов K: {0}'.format(K), file=output )
    print('Значение статистики хи-квадрат: {0}'.format(np.round(S, 3)), file=output )
    print('Критическое значение: {0}'.format(np.round(S_crit, 3)), file=output )
    if S < S_crit:
        print('Гипотеза о согласии не отвергается\n', file=output )
    else:
        print('Гипотеза о согласии отвергается\n', file=output )

    # Решение через достигнутый уровень значимости
    p = integrate.quad(lambda x: pow(x, (K - 1) / 2 - 1) * np.exp(-x / 2), S, math.inf)
    p /= pow(2, (K - 1) / 2) * gamma((K - 1) / 2)

    print('Достигнутый уровень значимости: {0}\nУровень значимости: {1}'.format(np.round(p[0], 8), alpha), file=output )
    if p[0] > alpha:
        print('Гипотеза о согласии не отвергается\n\n', file=output )
    else:
        print('Гипотеза о согласии отвергается\n\n' , file=output)


# Вычисление значения I(z)
def I(z, v):
    result = 0
    for k in range(pow(2, 5)):
        result += pow(z/2, v+2*k)/(gamma(k+1)*gamma(k+v+1))
    return result


# Проверка гипотеезы о согласии с теоретическим распределением по критерию Крамера-Мизеса-Смирнова
#   @sample - выборка
#   @cdf - теоретическая функция распределения
#   @params - параметры теоретического распределения
def cramer_von_mises_test(sample, cdf, params):
    n = len(sample)
    S_critical = 0.4614

    # Сортировка выборки
    sample = np.sort(sample)

    # Значение статистики
    S = 1 / (12*n)
    for i in range(n):
        if len (params) < 2:
            S += pow(cdf(sample[i], params[0]) - (2*(i+1)-1)/(2*n), 2)
        else:
            S += pow(cdf(sample[i], params[0], params[1]) - (2*(i+1)-1)/(2*n), 2)


    print('Критерий \u03C9\u00b2-Крамера-Мизеса-Смирнова', file=output)
    print('Значение статистики: {0}'.format(np.round(S, 3)), file=output)
    print('Критическое значение: {0}'.format(S_critical), file=output)
    if S < S_critical:
        print('Гипотеза о согласии не отвергается\n', file=output)
    else:
        print('Гипотеза о согласии отвергается\n', file=output)

    # Решение через достигнутый уровень значимости
    a1 = 0
    for j in range(pow(2, 6)):
        temp = (gamma(j+0.5)*np.sqrt(4*j+1))/(gamma(0.5)*gamma(j+1))
        temp *= np.exp(-pow(4*j+1, 2)/(16*S))
        temp *= (I(pow(4*j+1, 2)/(16*S), -0.25) - I(pow(4*j+1, 2)/(16*S), 0.25))
        a1 += temp
    a1 /= np.sqrt(2*S)

    p = 1 - a1

    print('Достигнутый уровень значимости: {0}\nУровень значимости: {1}'.format(np.round(p, 4), 0.05), file=output)
    if p > 0.05:
        print('Гипотеза о согласии не отвергается\n\n\n', file=output)
    else:
        print('Гипотеза о согласии отвергается\n\n\n', file=output)



def main():
    np.random.seed(5)

    param = convert_param('CompSim_lab5/input.txt')

    # Параметры распределения
    mu = param['m']
    v = param['v']

    # Выбор границ интервала
    a, b = find_bounds(pdf=laplace.pdf, cdf=laplace.cdf, params=[mu, v], precision=0.98)

    # Выбор множества точек
    M = fin_max_pdf(pdf=laplace.pdf, params=[mu, v], x0=a)


    print('Интервал моделирования: a = {0}, b = {1}\nM = {2}\n'.format(np.round(a, 3), np.round(b, 3), np.round(M, 3)), file=output)
    N = [50, 200, 1000]

    # Генерирование равномерно распределенной случайной величины
    arr = np.random.uniform(0, 1, 1000000)

    x_array = []
    pointx = []
    pointy = []
    # plot_pdf(laplace.pdf, params=[mu, v], pointx=pointx, pointy=pointy)
    for n in N:
        i = 0
        # Начальная точка замера времени моделирования
        while len(x_array) < n:
            # Берем пару равномерно распределенных чисел
            alpha1, alpha2 = arr[i], arr[i+1]
            i += 1

            # Находим координаты точки
            x = a + alpha1 * (b - a)
            y = alpha2 * M
            pointx.append(x)
            pointy.append(y)


            # Принимаем или отвергаем новую точку
            if y < laplace.pdf(x, mu, v):
                x_array.append(x)

        # plot_pdf(pdf=laplace.pdf , params=[mu, v], pointx, pointy)
        print(f'N={n}', file=output)
        # print('N = {0}\nX = {{'.format(n, x_array), file=output)
        # for i in range(n):
            # print(np.round(x_array[i], 3), file=output)
        # print('}', file=output)
        print('Равномерно распределенных величин потребовалось: {0}\n'.format(len(arr)), file=output)

        # plot_pdf_theoretical_and_empirical(pdf=laplace.pdf, params=[mu, v], sample=x_array, interval=[a,b])

        # chi_square_test(sample=x_array, cdf=laplace.cdf, params=[mu, v], interval=[a,b])

        cramer_von_mises_test(sample=x_array, cdf=laplace.cdf, params=[mu, v])


main()
