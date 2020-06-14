import time
import numpy as np
#------------------------------------------------------------------------------
#
def gauszw(x1, x2, n):
    '''
    Task:
        To compute 'n' Gauss nodes and weights within (x1, x2).
    In:
	    x1, x2   d       interval
        n        i       number of nodes
    Out:
	    x, w     d[n]    zeros and weights
    Tree:
        -
    Comments:
	    Tested only for x1 < x2. To test run, e.g.,
            ng = 64
            x, w = gauszw(-1.0, 1.0, ng)
        and comapare vs [1].
    Refs:
        1. https://pomax.github.io/bezierinfo/legendre-gauss.html
    Revision History:
        2020-06-12:
            update comments;
        20yy-mm-dd:
            first created and tested vs f90, cpp, [1].
    '''
#------------------------------------------------------------------------------
    _yeps = 3.0e-14
    x = np.zeros(n)
    w = np.zeros(n)
    m = int((n+1)/2)
    yxm = 0.5*(x2 + x1)
    yxl = 0.5*(x2 - x1)
    for i in range(m):
        yz = np.cos(np.pi*(i + 0.75)/(n + 0.5))
        while True:
            yp1 = 1.0
            yp2 = 0.0
            for j in range(n):
                yp3 = yp2
                yp2 = yp1
                yp1 = ((2.0*j + 1.0)*yz*yp2 - j*yp3 )/(j+1)
#-----------end for j-------------------------------------------------------------
            ypp = n*(yz*yp1 - yp2)/(yz*yz - 1.0)
            yz1 = yz
            yz = yz1 - yp1/ypp
            if (np.abs(yz - yz1) < _yeps):
                break # exit while loop
#-----end while-------------------------------------------------------------------
        x[i] = yxm - yz*yxl
        x[n-1-i] = yxm + yxl*yz
        w[i] = 2.0*yxl/((1.0 - yz*yz)*ypp*ypp)
        w[n-1-i] = w[i]
#---end for i------------------------------------------------------------------
    return x, w
#==============================================================================
#
def gsit(nit, ng1, nlr, dtau, ssa, xk):
    '''
    Task:
	    Solve RTE in a basic scenario using Gauss-Seidel (GS) iterations.
    In:
        nit    i       number of iterations, nit > 0
        ng1    i       number of gauss nodes per hemisphere
        nlr    i       number of layer elements dtau, tau0 = dtau*nlr
        dtau   d       thickness of element layer (integration step over tau)
        ssa    d       single scattering albedo
        xk     d[nk]   expansion moments, (2k+1) included, nk=len(xk)   
    Out:
	    mug, wg    d[ng1*2]        Gauss nodes & weights
        Iup, Idn   d[nlr+1, ng1]   intensity, I = f(tau), @ Gauss nodes
    Tree:
	    gsit()
            > gauszw() - computes Gauss zeros and weights.
    Note:
        TOA scaling factor = 1.0;
    Refs:
	    1. -
    Revision History:
        2020-06-14:
            Output modified: Iup,Idn = f(tau), Gauss nodes & weights
            Typo: Iup[nb-2, :] = I11up*np.exp(-tau[nb-2]/mu0) + (1.0 - np.exp(-dtau/mup))*J
                                                         --- mu0 was missing (not error because mu0=1)
        2020-06-12:
            major changes:
                identical form for SS & MS;
                symmetry of scattering at Gauss nodes is now used;
            minor changes:
                code cleaning;
                comments added;
            tested:
                for R&A vs IPOL as before.
        2020-06-07:
            tested vs IPOL for fine aerosol, erros fixed.
        2020-06-05:
            first created and tested vs IPOL for Rayleigh.
    '''
#
#   constants:
    mu0 = 1.0
#
#   parameters:
    nb = nlr+1
    nk = len(xk)
    ng2 = ng1*2
    tau0 = nlr*dtau
    tau = np.linspace(0.0, tau0, nb)
#
#   Gauss nodes and weights: mup - positive Gauss nodes; mug - all Gauss nodes:
    mup, w = gauszw(0.0, 1.0, ng1)
    mug = np.concatenate((-mup, mup))
    wg = np.concatenate((w, w))
#
#   Polynomilas Pk(x) & phaze function p @ all Gauss nodes, mug:
    pk = np.zeros((nk, ng2))
    pk[0, :] = 1.0
    pk[1, :] = mug
    pk[2, :] = 1.5*mug*mug - 0.5
    p = 1.0 + xk[1]*pk[1, :] + xk[2]*pk[2, :] # sum up 3 first moments explicetely (Rayleigh); recursion afterwards
    for ik in range(3, nk):
        pk[ik, :] = (2.0 - 1.0/ik)*mug*pk[ik-1, :] - (1.0 - 1.0/ik)*pk[ik-2, :]
        p += xk[ik]*pk[ik, :]
#   include scaling factor for conveneince:
    p *= 0.5*ssa
#
#   Single scattering up & down using layer dtau (alone):
#   -this form of SS allows for different optical layers;
#   -this form of SS is similar to MS below.    
    I11up = p[0:ng1]*mu0/(mu0 + mup)*(1.0 - np.exp(-dtau/mup - dtau/mu0))
    I11dn = p[ng1:ng2]*mu0/(mu0 - mup)*(np.exp(-dtau/mu0) - np.exp(-dtau/mup))
#   down:
    I1dn = np.zeros((nb, ng1))
    I1dn[1, :] = I11dn
    for ib in range(2, nb):
        I1dn[ib, :] = I1dn[ib-1, :]*np.exp(-dtau/mup) + I11dn*np.exp(-tau[ib-1]/mu0)
#   up:
    I1up = np.zeros_like(I1dn)
    I1up[nb-2, :] = I11up*np.exp(-tau[nb-2]/mu0)
    for ib in range(nb-3, -1, -1):
        I1up[ib, :] = I1up[ib+1, :]*np.exp(-dtau/mup) + I11up*np.exp(-tau[ib]/mu0) # note: mup > 0
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Check SS using analytical expression for SS=f(tau); homogeneous layer only:
#    for ib in range(nb):
#        taui = tau[ib]
#        e0 = np.exp(-taui/mu0)
#        I1up[ib, :] = p[0:ng1]*mu0/(mu0 + mup)*( e0 - np.exp(-(tau0 - taui)/mup - tau0/mu0) )
#        I1dn[ib, :] = p[ng1:ng2]*mu0/(mu0 - mup)*( e0 - np.exp(-taui/mup) )
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#   For multiple scattering (odd/even k are needed for up/down - not yet applied):
    wpij = np.zeros((ng2, ng2)) # sum{xk*pk(mui)*pk(muj)*wj, k=0:nk}
    for ig in range(ng2):
        for jg in range(ng2):
            wpij[ig, jg] = 1.0 + xk[1]*pk[1, ig]*pk[1, jg] + xk[2]*pk[2, ig]*pk[2, jg]
            for ik in range(3, nk):
                wpij[ig, jg] += xk[ik]*pk[ik, ig]*pk[ik, jg]
            wpij[ig, jg] *= 0.5*ssa*wg[jg]
#      
#   all Gauss directions for upward, wpup, and downward, wpdn, directions:
#   [Jup; Jdn] = [[Tup Rup]; [Rdn Tdn]]; Tup = Tdn = T; Rup = Rdn  = R   
    T = wpij[0:ng1, 0:ng1]
    R = wpij[0:ng1, ng1:ng2]
#
    Iup = np.copy(I1up)
    Idn = np.copy(I1dn)
    for itr in range(nit):
#       down:
        Iup05 = 0.5*(Iup[0, :] + Iup[1, :]) 
        Idn05 = 0.5*(Idn[0, :] + Idn[1, :]) # Idn[0, :] = 0.0        
        J = np.matmul(R, Iup05) + np.matmul(T, Idn05)
        Idn[1, :] = I11dn + (1.0 - np.exp(-dtau/mup))*J
        for ib in range(2, nb):
            Iup05 = 0.5*(Iup[ib-1, :] + Iup[ib, :]) 
            Idn05 = 0.5*(Idn[ib-1, :] + Idn[ib, :])
            J = np.matmul(R, Iup05) + np.matmul(T, Idn05)
            Idn[ib, :] = Idn[ib-1, :]*np.exp(-dtau/mup) + \
                             I11dn*np.exp(-tau[ib-1]/mu0) + \
                                 (1.0 - np.exp(-dtau/mup))*J
#       up:
        Iup05 = 0.5*(Iup[nb-2, :] + Iup[nb-1, :]) # Iup[nb-1, :] = 0.0 
        Idn05 = 0.5*(Idn[nb-2, :] + Idn[nb-1, :]) # Idn[0, :] = 0.0        
        J = np.matmul(T, Iup05) + np.matmul(R, Idn05)
        Iup[nb-2, :] = I11up*np.exp(-tau[nb-2]/mu0) + (1.0 - np.exp(-dtau/mup))*J
        for ib in range(nb-3, -1, -1): # going up, ib = 0 (TOA) must be included
            Iup05 = 0.5*(Iup[ib, :] + Iup[ib+1, :]) 
            Idn05 = 0.5*(Idn[ib, :] + Idn[ib+1, :])
            J = np.matmul(T, Iup05) + np.matmul(R, Idn05)
            Iup[ib, :] = Iup[ib+1, :]*np.exp(-dtau/mup) + \
                             I11up*np.exp(-tau[ib]/mu0) + \
                                 (1.0 - np.exp(-dtau/mup))*J
#       print("iter=%i"%itr)
#   TOA & BOA:
    return mug, wg, Iup[:, :], Idn[:, :]
#==============================================================================
#
def sfiup(mu, nlr, dtau, ssa, xk, mug, wg, Igup, Igdn):
    '''
    Task:
	    Source function integration up
    In:
        mu     d            upward LOS cos(VZA) < 0
        nlr    i            number of layer elements dtau, tau0 = dtau*nlr
        dtau   d            thickness of element layer (integration step over tau)
        ssa    d            single scattering albedo
        xk     d[nk]        expansion moments, (2k+1) included, nk=len(xk)
        mug    d[ng2]       Gauss nodes
        wg     d[ng2]       Gauss weights
        Igup   d[ng1, nb]   RTE solution at Gauss nodes & all boundaries upward   
        Igdn   d[ng1, nb]   Same as Igup, except for downward
    Out:
	    Itoa   d            Itoa=f(mu)
    Tree:
        -
    Note:
        TOA scaling factor = 1.0;
    Refs:
	    1. -
    Revision History:
        2020-06-14:
            First created and tested vs IPOL for R&A
    '''
#
#   constants:
    mu0 = 1.0
#
#   parameters:
    ng2 = ng1*2
    nk = len(xk)
    mup = -mu
    nb = nlr+1
    tau0 = nlr*dtau
    tau = np.linspace(0.0, tau0, nb)
#
#   Polynomilas Pk(x) & phaze function p @ all Gauss nodes, mug:
    pku = np.zeros(nk) # Pk @ user node
    pk = np.zeros((nk, ng2))
    pku[0] = 1.0
    pku[1] = mu
    pku[2] = 1.5*mu*mu - 0.5
    p = 1.0 + xk[1]*pku[1] + xk[2]*pku[2] # sum up 3 first moments explicetely (Rayleigh); recursion afterwards
    for ik in range(3, nk):
        pku[ik] = (2.0 - 1.0/ik)*mu*pku[ik-1] - (1.0 - 1.0/ik)*pku[ik-2]
        p += xk[ik]*pku[ik]
#   include scaling factor for conveneince:
    p *= 0.5*ssa
#   
    I11up = p*mu0/(mu0 + mup)*(1.0 - np.exp(-dtau/mup - dtau/mu0))
#
    I1up = np.zeros(nb)
    I1up[nb-2] = I11up*np.exp(-tau[nb-2]/mu0)
    for ib in range(nb-3, -1, -1):
        I1up[ib] = I1up[ib+1]*np.exp(-dtau/mup) + I11up*np.exp(-tau[ib]/mu0)
#
    wpij = np.zeros(ng2) # sum{xk*pk(mu)*pk(muj)*wj, k=0:nk}   
    for jg in range(ng2):
        muj = mug[jg]
        pk_2 = muj
        pk_1 = 1.5*muj*muj - 0.5
        wpij[jg] = 1.0 + xk[1]*pku[1]*pk_2 + xk[2]*pku[2]*pk_1
        for ik in range(3, nk):
            pk = (2.0 - 1.0/ik)*muj*pk_1 - (1.0 - 1.0/ik)*pk_2
            wpij[jg] += xk[ik]*pku[ik]*pk
            pk_2 = pk_1
            pk_1 = pk
    wpij *= 0.5*ssa*wg
#
    Iup = np.copy(I1up) # boundary condition: Iup[nb-1, :] = 0.0
#   up:
    Iup05 = 0.5*(Igup[nb-2, :] + Igup[nb-1, :]) # Iup[nb-1, :] = 0.0 
    Idn05 = 0.5*(Igdn[nb-2, :] + Igdn[nb-1, :])
    I05 = np.concatenate((Iup05, Idn05))    
    J = np.dot(wpij, I05)
    Iup[nb-2] = I11up*np.exp(-tau[nb-2]) + (1.0 - np.exp(-dtau/mup))*J
    for ib in range(nb-3, -1, -1):
        Iup05 = 0.5*(Igup[ib, :] + Igup[ib+1, :]) 
        Idn05 = 0.5*(Igdn[ib, :] + Igdn[ib+1, :])
        I05 = np.concatenate((Iup05, Idn05))    
        J = np.dot(wpij, I05)
        Iup[ib] = Iup[ib+1]*np.exp(-dtau/mup) + \
                         I11up*np.exp(-tau[ib]/mu0) + \
                             (1.0 - np.exp(-dtau/mup))*J
    return Iup[0]
#==============================================================================
#
if __name__ == "__main__":
#
    prnt_scrn = False
    scalef = 0.5/np.pi               # TOA flux scaling factor, 1/2pi
    fname_xk_aer = 'xk0036_0695.txt' # phase *matrix* moments
#
    phasefun = 'a'                 # 'a': aerosol (case sensetive); rayleigh otherwise
    nit = 10                       # number of iterations: nit = 0 returns SS
    ng1 = 16                       # number of Gauss nodes per heimsphere
    nlr = 100                      # number of layer elements
    dtau = 0.01                    # integration step over tau
    ssa = 0.99999999               # single scattering albedo
#
    muup = np.array([-0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9])
#    mudn = np.array([ 0.2,  0.4,  0.6,  0.8,  0.9])
#
    if phasefun == 'a':
        print("Aerosol:")
        #icolumn = 4 # column number with benchmark results (0-offset)
        #if nit == 0: icolumn = 3 # SS
        xk_all = np.loadtxt(fname_xk_aer, skiprows=8)
        xk = xk_all[:, 0]
        I_MS_test = np.array([0.516133E-01, 0.487570E-01, 0.437619E-01, 0.383643E-01,
                              0.333712E-01, 0.290310E-01, 0.252454E-01, 0.216996E-01])
    else:
        print("Rayleigh:")
        #icolumn = 2 # column number with benchmark results (0-offset)
        #if nit == 0: icolumn = 1 # SS
        xk = np.array([1.0, 0.0, 0.5]) # pure Rayleigh
        I_MS_test = np.array([0.128224E+00, 0.124847E+00, 0.119759E+00, 0.114495E+00,
                              0.109732E+00, 0.105696E+00, 0.102414E+00, 0.998377E-01])
#
    time_start = time.time()
#
    mug, wg, Igup, Igdn = gsit(nit, ng1, nlr, dtau, ssa, xk)
    for imu, mu in enumerate(muup):
       Itoa = sfiup(mu, nlr, dtau, ssa, xk, mug, wg, Igup, Igdn)
       Itoa *= scalef
       Itest = I_MS_test[imu]
       reldev = 100.0*np.abs(1.0 - Itoa/Itest)
       print("mu = %.1f,  I = %.4e,  Itest = %.5e,  |err, o/o|=%.2f"
             %(muup[imu], Itoa, Itest, reldev))
    #Iboa *= scalef
#
    time_end = time.time()
#
#    Ibmark = np.loadtxt('test_gsit_R&A.txt', comments='#', skiprows=1)
#    Ibm_toa = Ibmark[0:ng1, icolumn]
#    Ibm_boa = Ibmark[ng1:2*ng1, icolumn]
#    rerr_toa = 100.0*np.abs(1.0 - Itoa/Ibm_toa)
#    rerr_boa = 100.0*np.abs(1.0 - Iboa/Ibm_boa)
#    if nit == 0: print("Warning: nit=0 -> Single Scattering only")
#    print("number of iterations, nitr =", nit)
#    print("TOA errors, percent: max=%.2f, avr=%.2f"%(np.max(rerr_toa),np.average(rerr_toa)))
#    print("BOA errors, percent: max=%.2f, avr=%.2f"%(np.max(rerr_boa),np.average(rerr_boa)))
#    print("Excluding horizon, mu -> -0.0 & mu -> +0.0:")
#    print("TOA errors, percent: max=%.2f, avr=%.2f"%(np.max(rerr_toa[1:]),np.average(rerr_toa[1:])))
#    print("BOA errors, percent: max=%.2f, avr=%.2f"%(np.max(rerr_boa[1:]),np.average(rerr_boa[1:])))
#    if prnt_scrn:
#        for ig in range(ng1):
#            print("mu=%.4f, I=%.4e, err=%.2f,    mu=%.4f, I=%.4e, err=%.2f"
#                  %(mutoa[ig], Itoa[ig], rerr_toa[ig], muboa[ig], Iboa[ig], rerr_boa[ig]))
#
    print("gsit runtime = %.1f sec."%(time_end-time_start))