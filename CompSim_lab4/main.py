import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime
import random
import scipy.integrate as integrate
from scipy.special import gamma
import scipy.stats as stats
import pandas as pd
import sys
from pandas.plotting import table

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
    if (len(input) != 9):
        print('Wrong input value')
        raise Exception
    else:
        param = {
            'a': input[0],
            'b': input[1],
            'c': input[2],
            'm': input[3],
            'K1': input[4],
            'r': input[5],
            'K2': input[6],
            'alpha': input[7],
            'l': input[8]
        }
    return param

def create_freq(N, param):
    x = np.zeros(N)
    for i in range(1, N - 1):
        x[i + 1] = (param['a'] * x[i] + param['b'] *
                    (x[i - 1]**2) + param['c']) % param['m']
        # x[i+1] = (2*x[i] + 3*(x[i-1]**2) + 4) % 100
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
def exp_pdf(x, param):
    l = param['l']
    return (1 / l) * math.exp(-(x / l))

# Функция распределения
def exp_cdf(x, params):
    l = params['l']
    return 1 - math.exp(-(x / l))

# Обратная функция
def inv_func(x, param):
    alpha = x  #exp_cdf(x, param)
    l = param['l']
    return -l * np.log(1 - alpha)

# Функция отрисовки графиков для указанных функций плотности и распределения
def plot_pdf_and_cdf(pdf, cdf, params, xlims=10, ylims={'pdf': 1, 'cdf': 1}):
    x = np.arange(0, xlims + 0.001, 0.001)
    y_pdf = [pdf(point, params) for point in x]
    y_cdf = [cdf(point, params) for point in x]

    fig, sub = plt.subplots(1, 2, figsize=(5, 5))
    fig.suptitle('Экспоненциальное распределение с параметром λ = {0}'.format(
        params['l']))
    titles = ['Функция плотности', 'Функция распределения']

    for ax, y, title, kind in zip(sub.flatten(), [y_pdf, y_cdf], titles,
                                  ['pdf', 'cdf']):
        # Органичение значений осей координат
        ax.set_xlim(0, xlims)
        ax.set_ylim(0, ylims[kind])
        # Разметка
        ax.grid()
        ax.set_title(title)
        ax.plot(x, y, color='green', linewidth=2)
    plt.show()

# Функция отрисовки графиков теоретической функции плотности и эмпирической на одном полотне
#   @pdf - теоретическая функция плотности распределения
#   @params - параметры распределения
#   @sample - выборка
#   @xlims - ограничения для оси X  виде вещественного числа
#   @ylims - ограничения для оси Y  виде вещественного числа
def plot_pdf_theoretical_and_empirical(pdf, params, sample, xlims=10, ylims=1):
    x = np.arange(0, xlims + 0.001, 0.001)
    # Теоретические значения
    y = [pdf(point, params) for point in x]
    # Эмпирические значения
    weights, bin_edges = np.histogram(sample, range=(0, xlims), normed=True)

    # Создание полотна
    fig = plt.figure(figsize=(7, 5))
    fig.suptitle(
        'Экспоненциальное распределение с параметром λ = {0}\nДлина последо-вательности N = {1}'
        .format(params['l'], len(sample)))
    ax = fig.add_subplot(111)

    # Ограничение значений осей
    ax.set_xlim(0, xlims)
    ax.set_ylim(0, ylims)
    # Cетка
    ax.grid()
    # Вывод эмпирической функции
    plt.hist(bin_edges[:-1], bin_edges, weights=weights)
    # Вывод теоретической функции
    ax.plot(x, y, linewidth=3)
    # Значения оси Ox
    plt.xticks(bin_edges)
    plt.show()

def drawHistogram(ys, spans):
    plt.hist(ys, spans)
    plt.show()

# Проверка гипотеза о согласии смоделированной выборки с теоретическим распреде-лением
#   @sample - выборка
#   @cdf - теоретическая функция распределения
#   @param - параметры теоретического распределения
def chi_square_test(sample, cdf, param):
    alpha = param['alpha']

    # Длина последовательности
    n = len(sample)

    # Число интервалов
    K = int(5 * np.log10(n))

    # Частоты попадания в i-интервал
    weights, bins_edges = np.histogram(sample,
                                       bins=K,
                                       range=(0, np.ceil(max(sample))))

    # Значение статистики
    S = 0
    for i in range(K):
        P = cdf(bins_edges[i + 1], param) - cdf(bins_edges[i], param)
        S += pow(weights[i] / len(sample) - P, 2) / P
    S *= n

    # Критическое значение статистики
    S_crit = stats.chi2.ppf(1 - alpha, K - 1)

    print('Критерий \u03c7\u00b2 Пирсона')
    print('Значение статистики хи-квадрат: {0}'.format(np.round(S, 3)))
    print('Критическое значение: {0}'.format(np.round(S_crit, 3)))
    if S < S_crit:
        print('Гипотеза о согласии не отвергается\n')
    else:
        print('Гипотеза о согласии отвергается\n')

    # Решение через достигнутый уровень значимости
    p = integrate.quad(lambda x: pow(x, (K - 1) / 2 - 1) * math.exp(-x / 2), S,
                       math.inf)
    p /= pow(2, (K - 1) / 2) * gamma((K - 1) / 2)

    print(
        'Достигнутый уровень значимости: {0}\nУровень значимости: {1}'.format(
            np.round(p[0], 4), alpha))
    if p[0] > alpha:
        print('Гипотеза о согласии не отвергается\n\n')
    else:
        print('Гипотеза о согласии отвергается\n\n')

# Непараметрический критерий согласия Омега-Андерсона-Дарлинга
#   @sample - выборка
#   @cdf - теоретическая функция распределения
#   @param - параметры теоретического распределения
def anderson_darling_test(sample, cdf, param):
    alpha = param['alpha']

    # Критические значения
    critical = {
        0.15: 1.6212,
        0.1: 1.933,
        0.05: 2.4924,
        0.025: 3.0775,
        0.01: 3.8781
    }

    n = len(sample)
    # Упорядочиваем выборку
    sample = np.sort(sample)

    # Значение статистики
    S = 0
    for i in range(n):
        F = cdf(sample[i], param)
        if F == 0.0:
            F = 0.0 + sys.float_info.epsilon

        S += ((2 *
               (i + 1) - 1) * np.log(F)) / (2 * n) + (1 - (2 * (i + 1) - 1) /
                                                      (2 * n)) * np.log(1 - F)
    S = -n - 2 * S

    print('Критерий \u03a9\u00b2-Андерсона-Дарлинга')
    print('Значение статистики S: {0}'.format(np.round(S, 3)))
    print('Критическое значение статистики a2(S): {0}'.format(critical[alpha]))
    if S < critical[alpha]:
        print('Гипотеза о согласии не отвергается\n')
    else:
        print('Гипотеза о согласии отвергается\n')

    # Решение через достигнутый уровень значимости
    a2 = 0
    for j in range(10):
        integral = integrate.quad(
            lambda y: np.exp(S / (8 * (pow(y, 2) + 1)) - (pow(
                (4 * j + 1) * np.pi * y, 2)) / (8 * S)), 0, math.inf)[0]
        a2 += (pow(-1, j) * gamma(j + 0.5) * (4 * j + 1) * np.exp(-pow(
            (4 * j + 1) * np.pi, 2) / (8 * S))) / (gamma(0.5) *
                                                   gamma(j + 1)) * integral
    a2 *= (np.sqrt(2 * np.pi) / S)
    p = 1 - a2
    print(
        'Достигнутый уровень значимости: {0}\nУровень значимости: {1}'.format(
            np.round(p, 4), alpha))
    if p > alpha:
        print('Гипотеза о согласии не отвергается\n\n\n')
    else:
        print('Гипотеза о согласии отвергается\n\n\n')

def main():
    param = convert_param('CompSim_lab4/input.txt')
    N = 50

    start_time = datetime.now()

    # Генерация равномерно распределенной величины на отрезке (0,1)
    x_array = create_freq(N, param) / 100
    # x_array = np.random.rand(N)

    # Генерация непрерывной случайной величины методом обратной функции
    for i in range(N):
        x_array[i] = inv_func(x_array[i], param)

    print('Время: ', datetime.now() - start_time)
    # plot_pdf_and_cdf(exp_pdf, exp_cdf, param)
    plot_pdf_theoretical_and_empirical(exp_pdf, param, x_array)

    # Параметрический критерий согласия
    chi_square_test(x_array, exp_cdf, param)

    # Непараметрический критерий слогласия
    anderson_darling_test(x_array, exp_cdf, param)

main()
