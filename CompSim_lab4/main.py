import numpy as np
import math
import matplotlib.pyplot as plt 
import random
import scipy.stats
import pandas as pd
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
    if(len(input) != 9):
        print('Wrong input value')
        raise Exception
    else:
        param = {'a' : input[0],
                 'b' : input[1],
                 'c' : input[2],
                 'm' : input[3],
                 'K1': input[4],
                 'r' : input[5],
                 'K2': input[6],
                 'alpha':input[7], 
                 'l' : input[8]
                }
    return param

def create_freq(N, param):
    x = np.zeros(N)
    for i in range(1, N-1):
        x[i+1] = (param['a']*x[i] + param['b']*(x[i-1] ** 2) + param['c']) % param['m']
        # x[i+1] = (2*x[i] + 3*(x[i-1]**2) + 4) % 100
    return x

def calc_period(x):
    x1 = x[999]
    x2 = x[998]
    T = 2
    x_new = []
    for i in range(len(x)-3, 1, -1):
        if(x[i]!=x1):
            T+=1
        else:
            if(x[i-1]!=x2):
                T+=1
            else:
                break
    if(T > 100):
        for j in range(i, len(x)-1):
            x_new.append(x[j])
    return T, x_new

def create_table(df):
    fig, ax = plot.subplots()
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
        d += (x[i] - m) ** 2
    d /= (len(x) - 1)
    return d

# функция плотности распределения 
def exp_pdf(x, param):
    l = param['l']
    return (1 / l) * math.exp(-(x / l))


# Функция распределения 
def exp_cdf(x, params):
    l = params['l']
    vych = (1 / l) * math.exp(-(x / l))
    return 1 - vych

# FIX ME
def inv_func(x, param): 
    a = exp_cdf(x, param)
    l = param['l']
    chis = l * (1 - a)
    chis = np.log(chis)
    return -l * chis


# Функция отрисовки графиков для указанных функций плотности и распределения
def plot_pdf_and_cdf(pdf, cdf, params, xlims=10, ylims={'pdf':1,'cdf':1}):
    x = np.arange(0, xlims + 0.001, 0.001)
    y_pdf = [pdf(point, params) for point in x]
    y_cdf = [cdf(point, params) for point in x]

    fig, sub = plt.subplots(1, 2, figsize=(5, 5))
    fig.suptitle('Экспоненциальное распределение с параметром \u03C3 = {0}'.format(params['l']))
    titles = ['Функция плотности', 'Функция распределения']

    for ax, y, title, kind in zip(sub.flatten(), [y_pdf, y_cdf], titles, ['pdf', 'cdf']):
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
    fig.suptitle('Экспоненциальное распределение с параметром \u03C3 = {0}\nДлина последо-вательности N = {1}'.format(params['l'], len(sample)))
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

def create_table(df):
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')   
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    fig.tight_layout()

def main():
    param = convert_param('KM/CompSim_lab4/input.txt')
    N = 1000
    x = create_freq(N, param)
    T, x = calc_period(x)
    # Генерация непрерывной случайной величины методом обратной функции
    x_array = np.empty(50)
    for i in range(50):
        x_array[i] = inv_func(x[i], param)
    plot_pdf_and_cdf(exp_pdf, exp_cdf, param)
    # plot_pdf_theoretical_and_empirical(exp_pdf, param, x_array)
    # plot_pdf_and_cdf(exp_pdf, exp_cdf, param)
    # plot_pdf_theoretical_and_empirical(exp_pdf, param, x_array)
    
    df = pd.DataFrame()
    df['Критерий'] = 1
    create_table(df)
    plt.show()

main()