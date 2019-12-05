#import time
import numpy as np
#from scipy.special import eval_legendre as lp
import matplotlib.pyplot as plt
from rt_002_polleg import polleg
#==============================================================================
#
if __name__ == "__main__":
#   scattering angle & cosine
    sca = np.linspace(0.0, 180.0, 181)
    x = np.cos(np.radians(sca))
#   Rayleigh phase function
    r = 0.75*(1.0 + x*x)
#   expansion moments for the Rayleigh phase function: 2k+1 is INCLUDED
    xk_r = np.array([1.0, 0.0, 0.5])
    nk = len(xk_r)
#   restore Rayleigh phase function from moments: 3 different options
    pk = polleg(x, nk-1)   # recall, k = 0,1,2 hence nk-1;   not sure if the built-in eval_legendre is better
    rx_opt1 = np.dot(xk_r, pk)  # this looks nice, but NOT efficient because ...
#
    rx_opt2 = np.zeros_like(rx_opt1)
    for ik in range(nk):          # we ALREADY HAVE this loop in polleg!
        for ix in range(len(x)):  # we ALREADY HAVE this loop in polleg AS WELL!!
            rx_opt2[ix] += xk_r[ik]*pk[ik,ix]
#
#   Less nice, a bit harder to understand, but efficient way is:
#      compute Legendre polynomials "on the go"
    rx_opt3 = np.zeros_like(rx_opt1)
    for ix in range(len(x)):
        xi = x[ix]              # avoids addressing to an array
        rxi = 1.0               # xk[k=0]*Pk[k=0] + xk[1]*Pk[k=1] = 1*1 + 0*x
        p0 = 1.0                # for Pk-recurrence
        p1 = xi                 # for Pk-recurrence
        for ik in range(2, nk): #
            p2 = (2.0 - 1.0/ik)*xi*p1 - (1.0 - 1.0/ik)*p0 # recurrence
            rxi += xk_r[ik]*p2                            # accumulation - can not be fully vectorized :( 
            p0 = p1             # for Pk-recurrence
            p1 = p2             # for Pk-recurrence
        rx_opt3[ix] = rxi       # avoids addressing to an array inside the ik-loop
#
    plt.figure()
    plt.plot(sca, r, 'k', sca[0::10], rx_opt1[0::10], 'ro',\
       sca[2::10], rx_opt2[2::10], 'go', sca[4::10], rx_opt3[4::10], 'bo')
    plt.title('Rayleigh phase function')
    plt.xlabel('scattering angle, deg.')
    plt.legend(['explicit formula', 'from moments, xk: opt.1', 'from moments, xk: opt.2', 'from moments, xk: opt.3'])
    plt.grid(True)