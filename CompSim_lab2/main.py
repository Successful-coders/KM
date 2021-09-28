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
    # columns = list("qqqqqq")
    # create_table(Q_nizh, Q_verh, Q, Q_, test, 1, columns)

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
    interv = n/param['K1']
    v = np.zeros(int(param['K1']))
    start = 0
    w = np.ones_like(x_new) / n
    for i in range(int(param['K1'])):
        v[i] = 0
        end = round(start + interv)
        for j in range(start, end):
            if(x[j] < start):
                v[i] += 1
        v[i] = v[i]/n
    print(1)

def main():
    param = convert_param('CompSim_lab2/input.txt')
    x = create_freq(1000, param)
    T, x = calc_period(x)
    if(T < 100):
        print('Change parametrs')
    else:
        test2(x, 40, param)
    

main()