import numpy as np
import math
from datetime import datetime
from scipy.stats import t as Student
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.special import gamma
import scipy.stats as stats


np.random.seed(1)


# Построение графика плотности распределения
#   @pdf - плотность функции распределения
#   @params - параметры распределения
#   @interval - отрезок, на котором строится график
def plot_pdf(pdf, params, interval=[0, 10]):
    a, b= interval

    # Точки графика
    x = np.arange(a, b + 0.001, 0.001)

    if len(params) < 2:
        y = [pdf(point, params[0]) for point in x]
    else:
        y = [pdf(point, params[0], params[1]) for point in x]

    # Создание полотна
    fig = plt.figure(figsize=(10, 10))
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
def plot_pdf_theoretical_and_empirical(pdf, params, sample, interval=[0, 10]):
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
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('Длина последовательности N = {0}'.format(len(sample)))

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
    mu, var = params
    n = len(sample)
    S_critical = stats.chi2.ppf(1-alpha, 2)

    # Сортировка выборки
    sample = np.sort(sample)

    # Находим Dn+, Dn-, Dn
    Dn_plus = np.empty(n)
    Dn_minus = np.empty(n)

    for i in range(n):
        Dn_plus[i] = (i+1)/n - cdf(sample[i], mu, var)
        Dn_minus[i] = cdf(sample[i], mu, var) - i/n

    Dn = max(max(Dn_plus), max(Dn_minus))

    # Значение статистики
    S = pow(6*n*Dn + 1, 2) / (9*n)

    print('Критерий Смирнова', file=output)
    print('Значение статистики: {0}'.format(np.round(S, 3)), file=output)
    print('Критическое значение: {0}'.format(np.round(S_critical, 3)), file=output)
    if S < S_critical:
        print('Гипотеза о согласии не отвергается\n', file=output)
    else:
        print('Гипотеза о согласии отвергается\n', file=output)

    # Решение через достигнутый уровень значимости
    p = np.exp(-S/2)

    print('Достигнутый уровень значимости: {0}\nУровень значимости: {1}'.format(np.round(p, 4), alpha), file=output)
    if p > alpha:
        print('Гипотеза о согласии не отвергается\n\n\n\n', file=output)
    else:
        print('Гипотеза о согласии отвергается\n\n\n\n', file=output)



# Параметры распределения
mu = 3
var = 2

n = 8

# Отрисовка графиков теоретиических плотностей распределения
plot_pdf(pdf=Student.pdf, params=[mu, var])
plot_pdf(pdf=stats.norm.pdf, params=[0, 1], interval=[-4, 4])
plot_pdf(pdf=stats.chi2.pdf, params=[mu])
plot_pdf(pdf=stats.chi2.pdf, params=[var])

output = open('result.txt', 'w', encoding="utf-8")

for N in [50]:
    print('Длина основной генерируемой последовательности N = {0}\n'.format(N))

    chi2_array1 = []
    chi2_array2 = []

    # Начальная точка замера времени моделирования
    start_time = datetime.now()

    # Моделирование стандартной нормальной величины методом суммирования с пара-метром n
    snd_array = np.empty(N*(mu+var))
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

    # Моделирование распределения Хи-квадрат со степенями свободы mu и var
    for i in range(N):
        help1 = 0
        help2 = 0

        for j in range(mu):
            help1 += pow(snd_array[0], 2)
            snd_array = np.delete(snd_array, 0)
        chi2_array1.append(help1)

        for j in range(var):
            help2 += pow(snd_array[0], 2)
            snd_array = np.delete(snd_array, 0)
        chi2_array2.append(help2)

    # Вывод информации о вспомогательных распределениях
    print('Параметры распределения \u03c7\u00b2: {0} cтепеней свобо-ды'.format(mu))

    print('Первые 50 элементов выборки:\nX={', end='')
    for i in range(50):
        print(np.round(chi2_array1[i], 3), end=' ')
    print('}')

    plot_pdf_theoretical_and_empirical(pdf=stats.chi2.pdf, params=[mu], sample=chi2_array1)

    chi_square_test(sample=chi2_array1, cdf=stats.chi2.cdf, params=[mu])
    smirnov_test(sample=chi2_array1, cdf=stats.chi2.cdf, params=[mu])

    # Для еще одной выборки
    print('Параметры распределения \u03c7\u00b2: {0} cтепеней свобо-ды'.format(var))

    print('Первые 50 элементов выборки:\nX={', end='')
    for i in range(50):
        print(np.round(chi2_array2[i], 3), end=' ')
    print('}')

    plot_pdf_theoretical_and_empirical(pdf=stats.chi2.pdf, params=[var], sample=chi2_array2)

    chi_square_test(sample=chi2_array2, cdf=stats.chi2.cdf, params=[var])
    smirnov_test(sample=chi2_array2, cdf=stats.chi2.cdf, params=[var])


    # Моделирование основного распределения
    print('Моделирование распределения Фишера с параметрами: \u03BC = {0}; \u03BD = {1}\n'.format(mu, var))

    x_array = []
    for i in range(N):
        x_array.append((chi2_array1[i] * var)/(mu * chi2_array2[i]))

    print('X = {', end='', file=output)
    for i in range(N):
        print(np.round(x_array[i],3), end=' ', file=output)
    print('}', file=output)

    print('Время моделирования выборки из {0} элементов: {1}'.format(N, datetime.now() - start_time))
    print('Равномерно распределенных величин потребовалось: {0}\n'.format(n*N), file=output)

    plot_pdf_theoretical_and_empirical(pdf=Student.pdf, params=[mu, var], sample=x_array)

    chi_square_test(sample=x_array, cdf=Student.cdf, params=[mu, var])
    smirnov_test(sample=x_array, cdf=Student.cdf, params=[mu, var])

output.close()
