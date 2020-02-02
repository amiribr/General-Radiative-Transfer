import time
import numpy as np
#
def sglscat(mu0, tau0, xk, mu, az):
    '''
    Purpose:
	    To compute single scattering approximation in homogeneous atmosphere
    In:
        mu0   d        cos(sza) > 0, sza=[0, 90)
	    tau0  d        total optical thickness of atmosphere
        xk    d[nk]    expansion moments, (2k+1) included, nk=len(xk)
        mu    d[nmu]   -cos(vza); mu = [mu_toa < 0, mu_boa > 0], mu_toa, if any, must come first
        az    d[naz]   relative azimuth in radians; az=0 - forward scattering   
    Out:
	    I1    d[naz, nmu]   single scattering intensity
    Tree:
	    -
    Comments:
        The goal of this subroutine is to:
            a) discuss & create a template for other subroutines (style, comments, etc.);
            b) learn creating a "pythonish" numerically intense code: replace for-loops where possible
            c) discuss input format
        Theoretical basis is described e.g. in [1].
        
        My questions for discussion:            
            a) shall we provide size of arrays on input (in real RT code, this increases the number of
            input parameters and makes the input less readable, but explicit), or let the subroutine
            determine the sizes.
            b) shall we add revision history for each subroutine in the file or let the GitHub do the job? 
    References:
	    1. file rt_008_single _scattering, Eq.(14)
        2. https://stackoverflow.com/questions/29347987/why-cant-i-suppress-numpy-warnings
        4. https://en.wikipedia.org/wiki/Legendre_polynomials
    Revision History:
        2020-02-02 - 1) vectorized using python functions, original code & comments are totally revised
                     2) fixed typo: wrong 'mu2' in '-mu0*mu2 + ...'
                     3) for Rayleigh, sum up first 3 moments explicetely
        2020-01-26 - first created, *not yet tested*
    '''
#   auxiliary parameter:
    const_tiny = 1.0e-8    # to check fo  mu -> mu0
#   size:
    nk = len(xk)
    nmu = len(mu)
    nup = np.count_nonzero(mu<0.0)
#   tau-dependedn factor: see [2] to suppress warnings
    f = np.zeros(nmu)
    f[0:nup] = (mu0/(mu0 - mu[0:nup]))*(1.0 - np.exp(tau0/mu[0:nup] - tau0/mu0))
    f[nup:nmu] = np.where( np.abs(mu[nup:nmu] - mu0) < const_tiny,  tau0/mu[nup:nmu]*np.exp(-tau0/mu0), \
                           (mu0/(mu0 - mu[nup:nmu]))*(np.exp(-tau0/mu0) - np.exp(-tau0/mu[nup:nmu])) )
#   compute cos(scattering_angle) = mus:
    s1 = -mu0*mu
    s2 = np.outer(np.cos(az), np.sqrt(1.0 - mu*mu)*np.sqrt(1.0 - mu0*mu0))
    mus = np.add(s1, s2)    # mus.shape = (naz, nmu)
#   phaZe function @ mus:
    pk1 = mus               # P1(x) = x [3]
    pk2 = 1.5*mus*mus - 0.5 # P2(x) = (3x2-1)/2 [3]
    z = 1.0 + xk[1]*pk1 + xk[2]*pk2 # sum up 3 first moments explicetely (Rayleigh), run recursion afterwards
    for ik in range(3, nk):
        pk = (2.0 - 1.0/ik)*mus*pk2 - (1.0 - 1.0/ik)*pk1 # [3]
        z += xk[ik]*pk
        pk1 = np.copy(pk2)
        pk2 = np.copy(pk)
#   final result:
    return np.multiply(z, f)
#==============================================================================
#
if __name__ == "__main__":
#
    time_start = time.time()
#
    tau0 = 0.1
    mu0 = 0.5
    mu = np.array([-0.9, -0.5, 0.1, 0.5, 0.9])
    az = np.array([0.0, 45.0, 90.0, 180.0])
    xk = np.array([1.0, 0.0, 0.5])
    I1 = sglscat(mu0, tau0, xk, mu, np.radians(az))
    for iaz in range(len(az)):
        print('az=', az[iaz], 'I1 = ', I1[iaz,:])
#
    time_end = time.time()
#
    print("python's runtime = %.1f sec."%(time_end-time_start))