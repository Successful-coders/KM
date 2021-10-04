import numpy as np
import math
import matplotlib.pyplot as plot 
import random
import scipy.stats
import pandas as pd

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
    if(len(input) != 8):
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
                }
    return param

def create_freq(N, param):
    x = np.zeros(N)
    for i in range(1, N-1):
        x[i+1] = (param['a']*x[i] + param['b']*(x[i-1] ** 2) + param['c']) % param['m']
        # x[i+1] = (2*x[i] + 3*(x[i-1]**2) + 4) % 100
    return x

def create_plot(x):
    plot.plot(x)
    plot.xlabel('x')
    plot.grid(True, which='both')
    plot.axhline(y=0, color='k')
    plot.axvline(x=0, color='k')
    plot.show()

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

def calc_Q(x):
    Q = 0
    for i in range(len(x)-1):
        if(x[i]>x[i+1]):
            Q += 1
    return Q

def test1(x, n, param):
    x_new = []
    for i in range(n):
        x_new.append(x[i])
    U = scipy.stats.norm.ppf(1 - param['alpha']/2)
    Q = calc_Q(x_new)
    Q_ = n / 2
    Q_nizh = Q - (U * np.sqrt(n))/2
    Q_verh = Q + (U * np.sqrt(n))/2
    test = False
    if(Q_ >= Q_nizh and Q_<= Q_verh):
        test = True
    return test

def create_table(val1, val2, val3, val4, val5, val6, columns):
    fig, ax = plot.subplots()
    values = np.array((val1, val2, val3, val4, val5, val6))
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=values, colLabels=columns, loc='center')

    fig.tight_layout()

    plot.show()


def test2(x, n, param):
    x_new = []
    for i in range(n):
        x_new.append(x[i])

    weights = np.ones_like(x_new) / n
    plot.ylabel('Частота')
    plot.xlabel('Интервалы')
    # Интервалы
    kint = plot.hist(x_new, int(param['K1']), weights=weights)

    v = 1 / int(param['K1'])
    # Мат ожидание
    M = np.mean(x_new)
    # Дисперсия
    D = np.var(x_new)

    U = scipy.stats.norm.ppf(1 - param['alpha']/2)
    # create dov int
    dov_int = np.zeros((int(param['K1']), 2))
    for i in range (int(param['K1'])):
        dov_int[i][0] = kint[0][i] - U * math.sqrt((int(param['K1']) - 1) / n) / int(param['K1'])
        dov_int[i][1] = kint[0][i] + U * math.sqrt((int(param['K1']) - 1) / n) / int(param['K1'])

    temp = []
    result = True
    for i in range(int(param['K1'])):
        if v > dov_int[i][0] and v < dov_int[i][1]:
            result = True
        else:
            result = False
    M_teor = int(param['m']) / 2
    D_teor = int(param['m']) ** 2 / 12
    # доверительные интервалы для мат. ожидания
    dov_int_M = np.zeros((2))
    dov_int_M[0] = M - U * math.sqrt(D) / math.sqrt(n)
    dov_int_M[1] = M + U * math.sqrt(D) / math.sqrt(n)

    # доверительные интервалы для дисперсии
    dov_int_D = np.zeros((2))

    dov_int_D[0] = (n - 1) * D / (scipy.stats.chi2.ppf(1 - param['alpha'] / 2, n-1))
    dov_int_D[1] = (n - 1) * D / (scipy.stats.chi2.ppf(param['alpha'] / 2, n-1))

    return result


def test3(x, n, param):
    t = int((n - 1) / param['r'])
    x_new = np.zeros((int(param['r']), t))
    for i in range(int(param['r'])):
        for j in range(t):
            x_new[i][j] = x[(i+1) * j + (i+1)]
    
    res1 = test1(x, n, param)
    res2 = test2(x, n, param)

    if(res1 and res2):
        return True
    else:
        return False

def chi_test(x, n, param):
    x_new = []
    for i in range(n):
        x_new.append(x[i])
    weights = np.ones_like(x_new) / len(x_new)
    S = 0
    P = 1 / int(param['K1'])
    plot.ylabel('Частота')
    plot.xlabel('Интервалы')
    kint = plot.hist(x_new, int(param['K1']), weights=weights)

    for i in range(int(param['K1'])):
        S = S + (kint[0][i] - P)**2 / P
    S = S * len(x)
    plot.show()
    if scipy.stats.chi2.ppf(1-param['alpha'], param['K1'] - 1) > S:
        return True
    else:
        return False

# функция распределения
F = lambda x, n: x/ (n - 1) 

def kolmogorov(x, n, param):
    x_new = []
    for i in range(n):
        x_new.append(x[i])

    x_new.sort()

    D_plus = []
    i = 1
    D_minus = []
    for x in x_new:
        D_minus.append(F (x, n) - (i - 1) / n)
        D_plus.append(i / n - F(x, n))
        i = i + 1

    D_p = max(D_plus)
    D_m = max(D_minus)
    D = max(D_m, D_p)
    # K = надо посчитать чему равно К
    S = (6 * n * D + 1) / (6 * np.sqrt(n))
    P = 1 - K
    if(P > param['alpha']):
        return True # нет оснований для отклонения
    else:
        S_krit = 1.3581
        if(S > S_krit):
            return False # отклоняется
        else:
            return True




def main():
    param = convert_param('KM\CompSim_lab2\input.txt')
    N = 1000
    gen = 2
    if (gen == 1):
        x = create_freq(N, param)
    else:
        x = [random.randint(param['a'], param['m']) for i in range(N)]
    T, x = calc_period(x)
    if(T < 100):
        print('Change parametrs')
    else:
        kolmogorov(x, 40, param)
        # test1(x, 100, param)

        # test2(x, 40, param)
        # test2(x, 100, param)

        # test3(x, 40 / int(param['r']), param)
        # test3(x, 100 / int(param['r']), param)

main()