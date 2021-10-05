import numpy as np
import math
import matplotlib.pyplot as plot 
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
        if(x[i] > x[i+1]):
            Q += 1
    return Q

def test1(x, n, param):
    x_new = []
    for i in range(n):
        x_new.append(x[i])
    U = scipy.stats.norm.ppf(1 - param['alpha']/2)
    Q = calc_Q(x_new)
    Q_ = n / 2
    Q_nizh = Q - ((U * math.sqrt(n))/2)
    Q_verh = Q + ((U * math.sqrt(n))/2)
    test = False
    df = pd.DataFrame()
    df['Нижнее знач. дов. интервала'] = np.round(pd.Series(Q_nizh), 5)
    df['Верхнее знач. дов. интервала'] = np.round(pd.Series(Q_verh), 5)
    df['n'] = pd.Series(n)
    df['Q_'] = pd.Series(Q_)
    df['Q'] = pd.Series(Q)
    if(Q_ >= Q_nizh and Q_ <= Q_verh):
        df['Test'] = '+'
        test = True
    else:
        df['Test'] = '-'
    print(df)
    fig, ax = plot.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')   
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    fig.tight_layout()
    plot.show()
    return test

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
        print(np.round((dov_int[i][0]), 5))
    df = pd.DataFrame()
    # df['Нижнее знач. дов. интервала'] = [np.round(pd.Series(dov_int[i][0]), 5)  for i in range(int(param['K1']))]
    # df['Верхнее знач. дов. интервала'] = [np.round(pd.Series(dov_int[i][1]), 5) for i in range(int(param['K1']))]

    # df['v теоретическое'] = str(np.round(v, 3))
    print(v)
    result = False
    temp = []
    for i in range(int(param['K1'])):
        if v > dov_int[i][0] and v < dov_int[i][1]:
            temp.append('+')
            result = True
        else:
            temp.append('-')
    df['v в интервале'] = temp
    # print(temp)

    df1 = pd.DataFrame()
    M_teor = int(param['m']) / 2
    D_teor = int(param['m']) ** 2 / 12
    df1['M_teor'] = pd.Series(M_teor)
    # доверительные интервалы для мат. ожидания
    dov_int_M = np.zeros((2))
    dov_int_M[0] = M - U * math.sqrt(D) / math.sqrt(n)
    dov_int_M[1] = M + U * math.sqrt(D) / math.sqrt(n)
    df1['Нижнее знач. дов. интервала M'] = np.round(pd.Series(dov_int_M[0]), 5)
    df1['Верхнее знач. дов. интервала M'] = np.round(pd.Series(dov_int_M[1]), 5)
    if M_teor > dov_int_M[0] and M_teor < dov_int_M[1]:
        result = True
        df1['M в интервале'] = pd.Series('+')
    else:
        result = False
        df1['M в интервале'] = pd.Series('-')

    df2 = pd.DataFrame()
    df2['D_teor'] = pd.Series(D_teor)
    # доверительные интервалы для дисперсии
    dov_int_D = np.zeros((2))
    dov_int_D[0] = (n - 1) * D / (scipy.stats.chi2.ppf(1 - param['alpha'] / 2, n-1))
    dov_int_D[1] = (n - 1) * D / (scipy.stats.chi2.ppf(param['alpha'] / 2, n-1))
    df2['Нижнее знач. дов. интервала D'] = np.round(pd.Series(dov_int_D[0]), 5)
    df2['Верхнее знач. дов. интервала D'] = np.round(pd.Series(dov_int_D[1]), 5)
    if D_teor > dov_int_D[0] and D_teor < dov_int_D[1]:
        result = True
        df2['D в интервале'] = pd.Series('+')
    else:
        result = False
        df2['D в интервале'] = pd.Series('-')
    
    print(f'Для n={n}')
    df.to_csv('KM/CompSim_lab2/out.csv', header=False, index=False)
    print(df)
    print(df1)
    print(df2)
    plot.show()
    return result


def test3(x, n, param):
    t = int((n - 1) / param['r'])
    x_new = np.zeros((int(param['r']), t))
    for i in range(int(param['r'])):
        for j in range(t):
            x_new[i][j] = x[(i+1) * j + (i+1)]
    
    res1 = test1(x, n, param)
    res2 = test2(x, n, param)
    result = False
    df = pd.DataFrame()
    if(res1):
        df['Test1'] = pd.Series('+')
    else:
        df['Test1'] = pd.Series('-')
    
    if(res2):
        df['Test2'] = pd.Series('+')
    else:
        df['Test2'] = pd.Series('-')
    
    if(res1 and res2):
        result = True
    fig, ax = plot.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')   
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    fig.tight_layout()
    plot.show() 
    return result

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
    print(f'Для {n}')
    print(f'S={S}')
    plot.show()
    ur = scipy.stats.chi2.ppf(1-param['alpha'], param['K1'] - 1)
    print(f'Уровень значимости = {ur}')
    if ur > S:
        print('Гипотеза по хи-квадрат не отклоняется')
        return True #гипотеза по хи-квадрат не отклоняется
    else:
        print('Гипотеза по хи-квадрат отклоняется')
        return False #гипотеза по хи-квадрат  отклоняется

# функция распределения
F = lambda x, n: x/ (n - 1) 
e = 2.72
def calc_K(S):
    K = 0
    for i in range(-100, 1000):
        K += ((-1) ** i) * math.exp((-2 * (i ** 2) * (S ** 2)))
    return K

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
    S = (6 * n * D + 1) / (6 * np.sqrt(n))
    K = calc_K(S)
    print(f'Для {n}')
    print(f'S={S}')
    P = 1 - K
    print(f'P={P}')
    if(P > param['alpha']):
        print('Гипотеза по Колмагорова не отклоняется')
        return True # нет оснований для отклонения
    else:
        # S_krit = 1.3581
        # if(S > S_krit):
        print('Гипотеза по Колмагорова отклоняется')
        return False # отклоняется
        # else:
        #     print('Гипотеза по Колмагорова не отклоняется')
        #     return True


def main():
    param = convert_param('KM/CompSim_lab2/input.txt')
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
        # test1(x, 40, param)
        # test1(x, 100, param)

        test2(x, 40, param)
        test2(x, 100, param)

        # test3(x, int(40 / int(param['r'])), param)
        # test3(x, int(100 / int(param['r'])), param)

        # chi_test(x, 40, param)
        # chi_test(x, 100, param)

        # kolmogorov(x, 40, param)
        # kolmogorov(x, 100, param)

main()