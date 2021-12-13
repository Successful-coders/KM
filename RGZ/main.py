import numpy as np
import math
from datetime import datetime
from scipy.stats import t as Student
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.special import gamma
import scipy.stats as stats
import random


np.random.seed(1)


# Построение графика плотности распределения
#   @pdf - плотность функции распределения
#   @params - параметры распределения
#   @interval - отрезок, на котором строится график
def plot_pdf(pdf, params, interval=[-10, 10]):
    a, b= interval

    # Точки графика
    x = np.arange(a, b + 0.001, 0.001)

    if len(params) < 2:
        y = [pdf(point, params[0]) for point in x]
    else:
        y = [pdf(point, params[0], params[1]) for point in x]

    # Создание полотна
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    # Ограничение значений осей
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(0, 1)

    # Cетка
    ax.grid()
    ax.plot(x, y, linewidth=3, color='g')

    plt.show()


# Функция отрисовки графиков теоретической функции плотности и эмпирической на одном полотне
#   @pdf - теоретическая функция плотности распределения
#   @params - параметры распределения
#   @sample - выборка
#   @interval - отрезок, на котором строится график
def plot_pdf_theoretical_and_empirical(pdf, params, sample, interval=[-10, 20]):
    n = len(sample)
    a, b = interval

    x = np.arange(a, b + 0.001, 0.001)
    # Теоретические значения
    if len(params) < 2:
        y = [pdf(point, params[0]) for point in x]
    else:
        y = [pdf(point, params[0], params[1]) for point in x]

    # Число интервалов
    K = int(5 * np.log10(n))

    # Эмпирические значения
    weights, bins_edges = np.histogram(sample, bins=K, range=(a, b))

    # Число попаданий переведем в высоты столбца гистограммы
    weights = np.array(weights) / n
    for i in range(len(weights)):
        weights[i] /= (bins_edges[i+1]-bins_edges[i])

    # Создание полотна
    fig = plt.figure(figsize=(7, 5))
    # fig.suptitle('Длина последовательности N = {0}'.format(len(sample)))

    ax = fig.add_subplot(111)

    # Ограничение значений осей
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(0, 1)
    # Cетка
    ax.grid()

    # Вывод эмпирической функции
    ax.hist(bins_edges[:-1], bins_edges, weights=weights)
    # Вывод теоретической функции
    ax.plot(x, y, linewidth=3)
    # Значения оси Ox
    plt.xticks(bins_edges)
    ax.tick_params(axis='x', rotation=90)
    plt.show()


# Проверка гипотеза о согласии смоделированной выборки с теоретическим распреде-лением
#   @sample - выборка
#   @cdf - теоретическая функция распределения
#   @params - параметры теоретического распределения
def chi_square_test(sample, cdf, params, interval=[0, 10], alpha=0.05):
    a, b = interval
    n = len(sample)

    # Число интервалов разбиения
    K = int(5*np.log10(n))

    # Частоты попадания в i-интервал
    weights, bins_edges = np.histogram(sample, bins=K, range=(a, b))
    weights = np.array(weights) / n

    # Значение статистики
    S = 0
    for i in range(K):
        if len (params) < 2:
            P = cdf(bins_edges[i + 1], params[0]) - cdf(bins_edges[i], params[0])
        else:
            P = cdf(bins_edges[i+1], params[0], params[1]) - cdf(bins_edges[i], params[0], params[1])
        S += pow(weights[i] - P, 2) / P
    S *= n

    # Критическое значение статистики
    S_crit = stats.chi2.ppf(1-alpha, K-1)

    print('Критерий \u03c7\u00b2 Пирсона')
    print('Число интервалов K: {0}'.format(K))
    print('Значение статистики хи-квадрат: {0}'.format(np.round(S, 3)))
    print('Критическое значение: {0}'.format(np.round(S_crit, 3)))
    if S < S_crit:
        print('Гипотеза о согласии не отвергается\n')
    else:
        print('Гипотеза о согласии отвергается\n')

    # Решение через достигнутый уровень значимости
    p = integrate.quad(lambda x: pow(x, (K - 1) / 2 - 1) * np.exp(-x / 2), S, math.inf)
    p /= pow(2, (K - 1) / 2) * gamma((K - 1) / 2)

    print('Достигнутый уровень значимости: {0}\nУровень значимости: {1}'.format(np.round(p[0], 4), alpha))
    if p[0] > alpha:
        print('Гипотеза о согласии не отвергается\n\n')
    else:
        print('Гипотеза о согласии отвергается\n\n')


# Вычисление значения I(z)
def I(z, v):
    result = 0
    for k in range(pow(2, 5)):
        result += pow(z/2, v+2*k)/(gamma(k+1)*gamma(k+v+1))
    return result


# Проверка гипотеезы о согласии с теоретическим распределением по критерию Смир-нова
#   @sample - выборка
#   @cdf - теоретическая функция распределения
#   @params - параметры теоретического распределения
def smirnov_test(sample, cdf, params, alpha=0.05):
    var = params
    n = len(sample)
    S_critical = stats.chi2.ppf(1-alpha, 2)

    # Сортировка выборки
    sample = np.sort(sample)

    # Находим Dn+, Dn-, Dn
    Dn_plus = np.empty(n)
    Dn_minus = np.empty(n)
    c = cdf(sample[0], params[0])
    for i in range(n):
        Dn_plus[i] = (i+1)/n - cdf(sample[i], params[0])
        Dn_minus[i] = cdf(sample[i], params[0]) - i/n

    Dn = max(max(Dn_plus), max(Dn_minus))

    # Значение статистики
    S = pow(6*n*Dn + 1, 2) / (9*n)

    print('Критерий Смирнова' )
    print('Значение статистики: {0}'.format(np.round(S, 3)) )
    print('Критическое значение: {0}'.format(np.round(S_critical, 3)) )
    if S < S_critical:
        print('Гипотеза о согласии не отвергается\n' )
    else:
        print('Гипотеза о согласии отвергается\n' )

    # Решение через достигнутый уровень значимости
    p = np.exp(-S/2)

    print('Достигнутый уровень значимости: {0}\nУровень значимости: {1}'.format(np.round(p, 4), alpha) )
    if p > alpha:
        print('Гипотеза о согласии не отвергается\n\n\n\n' )
    else:
        print('Гипотеза о согласии отвергается\n\n\n\n' )

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

# Параметры распределения
var = 8

n = 12

# Отрисовка графиков теоретиических плотностей распределения
# plot_pdf(pdf=Student.pdf, params=[var])
# plot_pdf(pdf=stats.norm.pdf, params=[0, 1], interval=[-4, 4])
# plot_pdf(pdf=stats.chi2.pdf, params=[var])

output = open('result.txt', 'w', encoding="utf-8")

for N in [1000]:
    print('Длина основной генерируемой последовательности N = {0}\n'.format(N))

    chi2_arr = []

    # Начальная точка замера времени моделирования
    start_time = datetime.now()

    # Моделирование стандартной нормальной величины методом суммирования с параметром n
    snd_array = np.empty(N*var)
    for i in range(len(snd_array)):

        # Генерирование n равномерное распределенных на (0; 1) чисел
        alpha = np.random.uniform(0, 1, n)

        snd_array[i] = sum(alpha)
        snd_array[i] -= n / 2
        snd_array[i] /= np.sqrt(n / 12)

        # Первая поправка
        snd_array[i] += (pow(snd_array[i], 3) - 3 * snd_array[i]) / (20 * n)

    # Вывод информации о вспомогательных распределениях
    print('Описание вспомогательных распределений')
    print('Параметры нормального распределения: \u03BC = {0}; \u03C3 = {1}'.format(0, 1))

    print('Первые 50 элементов выборки:\nX={', end='')
    for i in range(50):
        print(np.round(snd_array[i], 3), end=' ')
    print('}')

    plot_pdf_theoretical_and_empirical(pdf=stats.norm.pdf, params=[0,1], sample=snd_array, interval=[-4, 4])

    print('Равномерно распределенных величин потребовалось: {0}\n'.format(len(snd_array)*n))

    chi_square_test(sample=snd_array, cdf=stats.norm.cdf, interval=[-4, 4], params=[0, 1])
    smirnov_test(sample=snd_array, cdf=stats.norm.cdf, params=[0, 1])

    help_snd = snd_array
    # Моделирование распределения Хи-квадрат со степенями свободы var
    for i in range(N):
        help1 = 0
        for j in range(var):
            help1 += pow(help_snd[0], 2)
            help_snd = np.delete(help_snd, 0)
        chi2_arr.append(help1)


    # Вывод информации о вспомогательных распределениях
    print('Параметры распределения \u03c7\u00b2: {0} cтепеней свободы'.format(var))

    print('Первые 50 элементов выборки:\nX={', end='')
    for i in range(50):
        print(np.round(chi2_arr[i], 3), end=' ')
    print('}')

    plot_pdf_theoretical_and_empirical(pdf=stats.chi2.pdf, params=[var], sample=chi2_arr)

    chi_square_test(sample=chi2_arr, cdf=stats.chi2.cdf, params=[var])
    smirnov_test(sample=chi2_arr, cdf=stats.chi2.cdf, params=[var])

    # Моделирование основного распределения
    x_array = []
    for i in range(N):
        zn = snd_array[i]/math.sqrt(chi2_arr[i])
        x_array.append(math.sqrt(n) * zn)

    print('X = {', end='' )
    for i in range(N):
        print(np.round(x_array[i],3), end=' ' )
    print('}' )

    print('Равномерно распределенных величин потребовалось: {0}\n'.format(n*N) )

    plot_pdf_theoretical_and_empirical(pdf=Student.pdf, params=[var], sample=x_array)

    chi_square_test(sample=x_array, cdf=Student.cdf, params=[var])
    smirnov_test(sample=x_array, cdf=Student.cdf, params=[var])

output.close()