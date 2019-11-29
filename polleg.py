import time
import numpy as np
from scipy.special import eval_legendre as lp
import matplotlib.pyplot as plt
#
def polleg(x, kmax):
    '''
    Purpose:
	    To compute the Legendre polynomials, Pk(x), for all orders k=0:kmax
    In:
        x     d[nx]          abscissas, x = [-1:+1], nx=len(x)
	    kmax  i[1]           maximum order, k = 0,1,2...kmax    
    Out:
	    pk    [nx, kmax+1]   Legendre polynomials
    Tree:
	    -
    Comments:
        Q&A:
        
        Q1 shall we add theory/equations like this (will not be possible for the RT solver) >>>
	    Pk(x) = Qkm(x) for m = 0. The Bonnet recursion formula [1, 2]:
	    (k+1)P{k+1}(x) = (2k+1)*P{k}(x) - k*P{k-1}(x),                           (1)
	    where k = 0:K, P{0}(x) = 1.0, P{1}(x) = x.
	    For fast summation over k, this index changes first.
        
        Q2: name of the subroutine - why not 'legendre', 'legpol', 'lp', etc...
        A2: Later we will have other polynomials, so i belive it is a good idea to
        start with 'pol' and then specify the type of the polynomial: 'polleg', 'polqkm', 'polrtm'...
    References:
	    1. https://en.wikipedia.org/wiki/Legendre_polynomials
	    2. http://mathworld.wolfram.com/LegendrePolynomial.html
        3. https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.legendre.html
        4. https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_legendre.html
    '''
    nx = len(x)
    nk = kmax+1
    pk = np.zeros((nk, nx))
    if kmax == 0:
        pk[0,:] = 1.0
    else:
        pk[0,:] = 1.0
        pk[1,:] = x
        for ik in range(2,nk):
            pk[ik,:] = (2.0 - 1.0/ik)*x*pk[ik-1,:] - (1.0 - 1.0/ik)*pk[ik-2,:]
    return pk
#==============================================================================
#
if __name__ == "__main__":
    x = np.linspace(-1.0, 1.0, 201)
    nx = len(x)
    fig, axs = plt.subplots(nrows=1, ncols=2)
#
#   Plot a few first orders
    axs[0].plot(x, lp(0, x), 'r', x, lp(1, x), 'g', x, lp(2, x), 'b', x, lp(10, x), 'k')
    axs[0].set_title('python')
    axs[0].grid(True)
#
    pk = polleg(x, 10)
    axs[1].plot(x, pk[0,:], 'r', x, pk[1,:], 'g', x, pk[2,:], 'b', x, pk[10,:], 'k')
    axs[1].set_title('home-cooked')
    axs[1].grid(True)
#
#   Compare for all k=0:kmax & all x
    kmax = 999
    time_start = time.time()
    pk = polleg(x, kmax)
    time_end = time.time()
    print('polleg runtime = %.1f sec.'%(time_end-time_start))
    pk_py = np.zeros_like(pk)
    time_start = time.time()
    for ik in range(kmax+1):
        pk_py[ik,:] = lp(ik, x)
    time_end = time.time()
    print("python's runtime = %.1f sec."%(time_end-time_start))
    print("max absolute difference = %.1e."%np.amax(np.abs(pk - pk_py)))