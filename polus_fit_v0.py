'''Rational Function program

   Do a Least Squares Fit for a rational function
   = ratio of polynomials of order Nn for de nominator
   , Nd for the denominator.

   The LSF is done using curve_fit from scipy.optimize.minpack
'''
import numpy as np
from scipy.optimize.minpack import curve_fit
import json
import os

try:
    import matplotlib.pyplot as plt
except:
    iPlot = False
else:
    iPlot = True
    from matplotlib import rc

    font = {'family': 'Verdana',
            'weight': 'normal'}
    rc('font', **font)


def get_ac(args):
    a = np.array(args)
    n = int(len(a) // 2)
    n1 = n
    n2 = n
    Cn = a[:n1]
    Cd = a[n1:]
    A = Cn, Cd
    return A


def ratfunctval(x, *args):
    ''' value of rational function in x for use with curve_fit

    '''
    a = np.array(args)
    n = int(len(a) // 2)
    n1 = n
    n2 = n
    Cn = a[:n1]
    Cd = a[n1:]
    ##   print('Cn ', Cn)
    ##   print('Cd ', Cd)
    ##   print('---------------------------------------------------------------')

    A = Cn, Cd
    R = poles_re(x, Cn, Cd)
    return R


def ratfit(x, y, Nom_deg, ini_dat, full=False):
    """

    .. math ::
        E = \\sum_{j=0}^k |R(x_j) - y_j|^2

    """

    x = np.asarray(x) + 0.0
    y = np.asarray(y) + 0.0

    Cn = np.ones(Nom_deg) * ini_dat[0]
    Cd = np.ones(Nom_deg) * ini_dat[1]

    ##    Cn[0] =  10**9
    ##    Cd[0] =  10**9
    args = []
    for i in Cn: args.append(i)
    for i in Cd: args.append(i)
    db = {'maxfev': 3600, 'method': 'trf'}
    opt_param, covar_param = curve_fit(ratfunctval, x, y, args, **db)
    # opt_param, covar_param = curve_fit(ratfunctval, x, y, **db)

    if full:
        return opt_param, covar_param
    else:
        return opt_param


def poles_re(x, a, c):
    sm = 0.
    n = len(a)
    for i in range(n):
        ##        sm += (-np.abs(a[i])*c[i])/(x**2 + (a[i]**2))
        sm += (-a[i] * c[i]) / (x ** 2 + (a[i] ** 2))
    return sm


def poles_im(x, a, c):
    sm = 0.
    n = len(a)
    for i in range(n):
        ##        sm += (x*np.abs(c[i]))/(x**2 + (a[i]**2))
        sm += (x * c[i]) / (x ** 2 + (a[i] ** 2))
    return sm


def calc_re(f, a1, c1, a2=0.0, c2=0.0):
    """


    Parameters
    ----------
    f : TYPE
        DESCRIPTION.
    a1 : TYPE
        DESCRIPTION.
    c1 : TYPE
        DESCRIPTION.
    a2 : TYPE, optional
        DESCRIPTION. The default is 0.0.
    c2 : TYPE, optional
        DESCRIPTION. The default is 0.0.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    tt = (f - a2) ** 2 + a1 ** 2
    pp = (f - a2) * c2 - a1 * c1
    return pp / tt


def calc_im(f, a1, c1, a2=0.0, c2=0.0):
    """


    Parameters
    ----------
    f : TYPE
        DESCRIPTION.
    a1 : TYPE
        DESCRIPTION.
    c1 : TYPE
        DESCRIPTION.
    a2 : TYPE, optional
        DESCRIPTION. The default is 0.0.
    c2 : TYPE, optional
        DESCRIPTION. The default is 0.0.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    tt = (f - a2) ** 2 + a1 ** 2
    pp = -(f - a2) * c1 - a1 * c2
    return pp / tt


def getinitial(fl):
    dd = np.loadtxt(fl, skiprows=0)
    dd = dd[:, :]
    ##    x = 2.*np.pi* dd[:,0]
    x = dd[:, 0]
    yy = dd[:, 1]
    ##    yy = poles(x, aa, cc)
    ##    y = yy - 1.0*yy[-1]

    return x, yy, dd[:, 2]


def podgon(a, c):
    """
    """
    for i in range(len(a)):
        if a[i] > 0.0:
            a[i] *= -1.0
            c[i] *= -1.0

    return a, c


def mist(yy, yyn):
    """


    Parameters
    ----------
    yy : TYPE
        DESCRIPTION.
    yyn : TYPE
        DESCRIPTION.

    Returns
    -------
    rr : TYPE
        DESCRIPTION.

    """
    tt = 0.0
    tp = 0.0
    for y, yn in list(zip(yy, yyn)):
        tp += y ** 2
        tt += (yn - y) ** 2
    rr = np.sqrt(tt) / np.sqrt(tp)
    return rr


def real_fit(dr, db):
    """
    """

    flnm = db['file']  # 'real+.txt'
    N_pol = db['number_poles']
    in_dt = db['init_data']

    ##    flnm = 'ab_eps(f).txt'
    ##    flnm = 'init.txt'
    fl = flnm
    ##    x, y = getracion(dr)
    x, y, yi = getinitial(fl)
    yst = 1.0 * y[-1]
    y = y - 1.0 * yst
    w = 2 * np.pi * x


    try:
        param, covar = ratfit(w, y, N_pol, in_dt, full=True)
    except RuntimeError:
        print('Решение не сходится')
        return None, None
    except np.linalg.LinAlgError as e:
        print('Не удалось решить уравнение. Попробуйте ещё раз.')
        print(e)
        return None, None

    ##    print(covar)
    ##    yn = ratfunctval(x, *param)
    ##    np.savetxt(os.path.join(os.path.dirname(__file__),'yn'),yn)

    ##    print(param)
    aa, cc = get_ac(param)
    ##    print('---------------------------------------------------------------')
    ##    print('A ', aa)
    ##    print('C ', cc)
    ##    print('---------------------------------------------------------------')
    aa, cc = podgon(aa, cc)
    print('----EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE----------')
    print('A ', aa)
    print('C ', cc)
    print('---------------------------------------------------------------')

    yn = poles_re(w, aa, cc)
    yin = poles_im(w, aa, cc)
    print('Calculation error {0:12.4e}'.format(mist(y, yn)))

    return aa, cc


def plot_graphic(*args):
    x, y, yn, yst, yi, yin = args

    extend = 8.85 * 10e-12 * x / 9e9

    fig = plt.figure('solv_real', figsize=(9, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogx(x, y, 'r-.', label='Экспериментальные данные')
    ax.semilogx(x, yn + yst, 'b-', label='Расчетные данные')
    ##    ax.loglog(x, y+0*yst, 'r-.', label='original')
    ##    ax.loglog(x, yn+0*yst, 'b-', label='calculate')
    ax.set_xlabel("f")
    ax.legend(loc='best', fancybox=True, shadow=True)
    ax.grid(True)
    ax.set_title('Диэлектрическая проницаемость')
    # ax.set_title('real part')
    fig = plt.figure('solv_imag', figsize=(9, 7))
    plt.tight_layout()
    ax = fig.add_subplot(1, 1, 1)
    # ax.semilogx(x, yi, 'r-.', label='original')
    # ax.semilogx(x, yin, 'b-', label='calculate')
    ax.loglog(x, yi * y * extend, 'r-.', label='Экспериментальные данные')
    ax.loglog(x, yin * extend, 'b-', label='Расчетные данные')
    ax.set_xlabel("Гц")
    ax.set_ylabel(r'с$^{-1}$', size=18)
    ax.legend(loc='best', fancybox=True, shadow=True)
    ax.grid(True)
    ax.set_title('Электропроводность')
    # ax.set_title('imaginary part')
    plt.tight_layout()

    if False:
        fig = plt.figure('error', figsize=(9, 7))
        ax = fig.add_subplot(2, 1, 1)
        ax.semilogx(x, y - yn - yst, 'r-.', label='error')
        ax.set_xlabel("f")
        ax.legend(loc='best', fancybox=True, shadow=True)
        ax.grid(True)
        ax.set_title('real part')
        ax = fig.add_subplot(2, 1, 2)
        ax.semilogx(x, yi - yin, 'r-.', label='error')
        ax.set_xlabel("f")
        ax.legend(loc='best', fancybox=True, shadow=True)
        ax.grid(True)
        ax.set_title('imaginary part')
        plt.tight_layout()

    plt.show()


def main(db):
    # tt = os.path.splitext(db['file'])

    # filout = tt[0] + '_' + str(db['number_poles']) + tt[1]
    dr = os.path.dirname(__file__)

    aa, cc = real_fit(dr, db)
    if aa is None and cc is None:
        return [None for _ in range(8)]
    flnm = db['file']  # 'real+.txt'
    x, y, yi = getinitial(os.path.join(dr, flnm))
    yst = 1.0 * y[-1]

    w = 2 * np.pi * x
    yn = poles_re(w, aa, cc)
    yin = poles_im(w, aa, cc)
    nl = np.zeros_like(aa)
    dat = list(zip(aa, nl, cc, nl))
    # np.savetxt(os.path.join(dr, filout), dat, header='poles_real poles_imag residuals_real residuals_imag')
    # np.savetxt(os.path.join(dr, 'yn'), list(zip(x, yn)))

    return x, y, yn, yst, yi, yin, aa, cc


if __name__ == '__main__':
    """

    """

    import os

    db = {'file': 'ab_eps(f).txt', 'number_poles': 10, 'init_data': (1.e7, 1.e7)}

    # with open(os.path.join(dr, "init_fit.json"), "w") as write_file:
    #     json.dump(db, write_file)

    # with open(os.path.join(dr, "init_fit.json"), "r") as read_file:
    #     db = json.load(read_file)

    main(db)
