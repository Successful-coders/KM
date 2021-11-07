import numpy as np
import math
import matplotlib.pyplot as plot 
import random
import scipy.stats
import pandas as pd
from pandas.plotting import table
from scipy.stats import poisson
from scipy.stats import chi2

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
    x[0] = 0.01
    for i in range(N-1):
        x[i+1] = ((param['a']*x[i] + param['b']*(x[i-1] ** 2) + param['c']) % param['m']) 
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

def chi_test(p, l):
    # статистика
    alpha = 0.05
    S=0
    K = len(p)
    # теоретические вероятности
    # вероятность появления каждого уникального элемента
    P = [l ** k / math.factorial(k) * math.exp(-l) for k in range(K)] # р уже пронормированы по n
    for i in range(K):
        S = S + (p[i] - P[i])**2 / P[i]
    S = S * len(p)

    print('Статистика S =', np.round(S, 3))
    print('Статистика S крит. =', np.round(chi2.ppf(1-alpha, K - 1), 3))
    if chi2.ppf(1-alpha, K - 1) > S:
        print(np.round(S, 3), ' <', np.round(chi2.ppf(1-alpha, K - 1), 3),
        '=> гипотеза по хи-квадрат не отклоняется') 
        return 1
    else:
        print(np.round(S, 3), ' >', np.round(chi2.ppf(1-alpha, K - 1), 3),
        '=> гипотеза по хи-квадрат отклоняется') 
        return 0

def degree_law(k):
    degree = []
    for i in range(0, k):
        degree.append(1/(2 ** i))
    sum_deg = sum(degree)
    return degree, sum_deg

def chi_test_std(p, interv):
    # статистика
    alpha = 0.05
    S=0
    K = interv
    S = 1
    Xi = list(set(p))
    for x in Xi:
        S += x * p.count(x) / K

    print('Статистика S =', np.round(S, 3))
    print('Статистика S крит. =', np.round(chi2.ppf(1-alpha, K - 1), 3))
    if chi2.ppf(1-alpha, K - 1) > S:
        print(np.round(S, 3), ' <', np.round(chi2.ppf(1-alpha, K - 1), 3),
        '=> гипотеза по хи-квадрат не отклоняется') 
        return S
    else:
        print(np.round(S, 3), ' >', np.round(chi2.ppf(1-alpha, K - 1), 3),
        '=> гипотеза по хи-квадрат отклоняется') 
        return S


def discret_rek(p_rav, n):
    degree, sum_deg = degree_law(n)
    etta = np.zeros(n)
    res_seq = []
    iter = 0
    etta_ = np.zeros(n)

    for M in p_rav:
        i = 1
        P = degree[1]
        while M >= 0 and i < n:
            M -= P
            i += 1
            P = degree[i]
        res_seq.append(i)
    # Определение эффективности алгоритма
    weights = np.ones_like(res_seq) / len(res_seq)
    kint = plot.hist(res_seq, weights=weights)
    interv = len(kint[0]) 
    print('Число интервалов=', interv)
    S = chi_test_std(res_seq, interv)
    K = math.ceil(S * len(res_seq))
    print('Число иттераций: {0}\n'.format(K))
    plot.xlabel('k')
    plot.ylabel('Частота')
    title = 'Стандартный алгоритм при k = ' + str(interv) 
    plot.title(title)
    plot.show()
    return res_seq

# закон Пуассона
def Puasson(l, n):
    # k - номер интервала
    puason = []
    P = lambda k: l ** k / math.factorial(k) * math.exp(-l) # находим вероятности до столбца с лямбдой
    i=0
    while i <= l:
        puason.append(P(i))
        i=i+1
    # завершаем хвост
    # пока текущая вероятность не станет меньше определенного значения 
    while P(i) > 0.001:
        puason.append(P(i))
        i=i+1
    # в последний интервал поместим оставшуюся вероятность 
    puason.append(1 - np.sum(puason[:len(puason)]))
    # суммы вероятностей
    sup_puason = [np.sum(puason[:i]) for i in range(1,len(puason)+1)] # i - конечное число k
    return puason, sup_puason, i

def nestd(p_rav, n):
    L = 7
    puason, sup_puason, K = Puasson(L, n) # сумма с i = 0 до лямбда включительно 
    Q = np.sum(puason[:L+1])
    etta_ = np.zeros(K)
    res_seq = []

    iter = 0
    for alpha in p_rav:
        M = alpha - Q 
        m=L
        P = puason[L]
        if M < 0:
            while True: 
                M=M+P
                iter = iter + 1
                if M >= 0 or m == 0: 
                    etta_[m] = etta_[m] + 1 
                    res_seq.append(m)
                    break
                m = m - 1
                P = puason[m]
        elif M >= 0:
            while True: 
                m = m + 1
                P = puason[m]
                iter = iter + 1
                if M <= 0 or m == n:
                    etta_[m-1] = etta_[m-1] + 1 
                    res_seq.append(m-1)
                    break
                M = M - P
    print('Количество итераций:', iter)
    print('Число интервалов К:', K) # нормировка частот
    etta_ = etta_ / n
    # тест хи-квадрат 
    chi_test(etta_, L)
    plot.bar(np.arange(K), etta_) 
    plot.xticks(np.arange(K), np.arange(K))
    plot.xlabel('k')
    plot.ylabel('Частота')
    title = 'Нестандартный алгоритм при n = ' + str(n) + ', λ = ' + str(L) 
    plot.title(title)
    plot.show()
    return res_seq, K

def teor_puasson(n, l, K):
    puason = [l ** k / math.factorial(k) * math.exp(-l) for k in range(K)]
    plot.bar(np.arange(K), puason)
    plot.xticks(np.arange(K), np.arange(K))
    plot.xlabel('k')
    plot.ylabel('Частота')
    title = 'Теоретические частоты Пуассона при k = ' + str(n) + ', λ = ' + str(l)
    plot.title(title) 
    plot.show()

def teor_degree(n):
    degr = [1 / (2 ** k) for k in range(n)]
    plot.bar(np.arange(n), degr)
    plot.xticks(np.arange(n), np.arange(n))
    plot.xlabel('k')
    plot.ylabel('Частота')
    title = 'Теоретические частоты степенного закона при k = ' + str(n)
    plot.title(title) 
    plot.show()


def main():
    param = convert_param('CompSim_lab2/input.txt')
    N = 1000
    x = [random.uniform(0, 1) for i in range(N)]
    T, x = calc_period(x)
    if(T < 100):
        print('Change parametrs')
    
    # задаем раномерную случайную величину
    x_40 = []
    for i in range(40):
        x_40.append(x[i])
    
    x_100 = []
    for i in range(100):
        x_100.append(x[i])
    
    # задаем дискретную стандартную рекурентную
    # disc_40 = discret_rek(x_40, 40)
    # print(disc_40)
    # teor_degree(6)
    disc_100 = discret_rek(x_100, 100)
    print(disc_100)
    teor_degree(10)

    # задаем дискретную нестандартным
    # nestand_dic_40, K = nestd(x_40, 40)
    # teor_puasson(40, 7, K)
    # print(nestand_dic_40)
    # nestand_dic_100, K = nestd(x_100, 100)
    # teor_puasson(100, 7, K)
    # print(nestand_dic_100)
    

main()