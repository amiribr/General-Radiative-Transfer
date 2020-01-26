import time
import numpy as np
from scipy.special import eval_legendre as lp
#
def sglscat(mu0, tau0, xk, mu, az):
    '''
    Purpose:
	    To compute single scattering approximation in homogeneous atmosphere
    In:
        mu0   d        cos(sza) > 0, sza=[0, 90)
	    tau0  d        total optical thickness of atmosphere
        xk    d[nk]    expansion moments, (2k+1) included, nk=len(xk)
        mu    d[nmu]   -cos(vza); mu = [mu_toa < 0, mu_boa > 0], mu_toa, if exist, must come first
        az    d[naz]   relative azimuth in radians; az=0 - forward scattering   
    Out:
	    inten1    [nmu,naz]   single scattering intensity
    Tree:
	    -
    Comments:
        The goal of this subroutine is to:
            a) discuss & create a template for other subroutines (style, comments, etc.);
            b) learn creating a "pythonish" numerically intense code: replace for-loops where possible
            c) discuss input format
        Theoretical basis is described e.g. in [1].
        
        My questions for discussion:
            a) how to efficiently remove the for-loops? See my "thinkme" part, where i am trying to compute
            the scattering angle without explicit for-loops. It works, but looks ugly....
            
            b) shall we provide size of arrays on input (in real RT code, this increases the number of
            input parameters and makes the input less readable, but explicit), or let the subroutine
            determine the sizes.
            
            c) shall we stay with built-in Legendre function (easier), or use or own explicitly coded
            in the subroutine. Note, that for polarization, python does not provide generalized Legendre
            plynomials, Pkmn(x). At least not now, as far as i know...
            
            d) shall we add revision history for each subroutine in the file or let the GitHub do the job? 
    References:
	    1. file rt_008_single _scattering, Eq.(14)
    Revision History:
        2020-01-26 - first created, *not yet tested*
    '''
#------------------------------------------------------------------------------
#thinkme: i'm trying to create a 2d-array of scattering angles, mus[len(mu), len(mu0)]
#         in a nice python way. The result looks ugly and probably inefficient:
#    
#         smu = np.sqrt(1.0 - mu*mu)   # check for mu*mu > 1.0 due to round-off error
#         smu0 = np.sqrt(1.0 - mu0*mu0)
#         caz = np.cos(azi)
#         mumu0 = np.transpose(np.tile(mu*mu0, (len(azi), 1)))
#         mus = -mumu0 + np.outer(smu*smu0, caz)
#------------------------------------------------------------------------------
    const_tiny = 1.0e-8          # to check fo  mu -> mu0
    mu02 = mu0*mu0
    mu2 = mu*mu
    naz = len(az)
    nk = len(xk)
    nmu = len(mu)
    nup = np.count_nonzero(mu<0.0)
    ndn = nmu-nup
    print('nmu=%i, nup=%i, ndn=%i'%(nmu,nup,ndn))
    inten1 = np.zeros((nmu, naz))
    for imu in range(nup):
        for iaz in range(naz):
            mus = -mu0*mu2[imu] + np.sqrt((1.0 - mu2[imu])*(1.0 - mu02))*np.cos(az[iaz])
            for ik in range(nk):
                pk = lp(ik, mus)
                inten1[imu, iaz] += xk[ik]*pk*(mu0/(mu0 - mu[imu]))*(1.0 - np.exp(tau0/mu[imu] - tau0/mu0))
    for imu in range(nup, nmu):
        for iaz in range(naz):
            mus = -mu0*mu2[imu] + np.sqrt((1.0 - mu2[imu])*(1.0 - mu02))*np.cos(az[iaz])
            for ik in range(nk):
                pk = lp(ik, mus)
                if (np.abs(mu[imu] - mu0) < const_tiny):
                    inten1[imu, iaz] += xk[ik]*pk*tau0/mu0*np.exp(-tau0/mu0)
                else:
                    inten1[imu, iaz] += xk[ik]*pk*(mu0/(mu0 - mu[imu]))*(np.exp(-tau0/mu0) - np.exp(-tau0/mu[imu]))      
    return inten1
#==============================================================================
#
if __name__ == "__main__":
#
    time_start = time.time()
#
    tau0 = 0.1
    mu0 = 0.5
    mu = np.array([-0.9, -0.5, 0.1, 0.5, 0.9])
    az = np.array([0.0, np.pi/4, np.pi/2, np.pi])
    xk = np.array([1.0, 0.0, 0.5])
    inten1 = sglscat(mu0, tau0, xk, mu, az)
    for imu in range(len(mu)):
        print('mu=', mu[imu], 'inten1 = ', inten1[imu,:])
#
    time_end = time.time()
#
    print("python's runtime = %.1f sec."%(time_end-time_start))