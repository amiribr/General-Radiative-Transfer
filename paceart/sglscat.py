import numpy as np
#
def sglscat(mu0, mu, az, tau0, ssa, xk):
    '''
    Purpose:
	    To compute single scattering approximation in homogeneous atmosphere
    In:
        mu0   d         cos(sza) > 0, sza=[0, 90)
        mu    d[nmu]   -cos(vza); mu = [mu_toa < 0, mu_boa > 0], mu_toa, if any, must come first
        az    d[naz]    relative azimuth in radians; az=0 - forward scattering 
	    tau0  d         total optical thickness of atmosphere
        xk    d[nk]     expansion moments, (2k+1) included, nk=len(xk)     
    Out:
	    I1    d[naz, nmu]   single scattering intensity
    Tree:
	    -
    Comments:
        Corrections after 2020-02-22 tests:
            a) -mu*mu0 -> +mu*mu0
            b)  scaling factor ssa/4pi (from RTE)
            c)  sequince of input parameters chenged: geometru first, optical paramters next
            d)  comments updatd
    References:
	    1. Wauben WMF, de Haan JF, Hovenier JW, 1993: Astron.Astrophys., v.276, pp.589-602.
    Revision History:
        2020-02-22 - Tested vs published results [1]
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
    s1 = mu0*mu
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
    return np.multiply(z, f)*ssa/(4.0*np.pi)
#==============================================================================