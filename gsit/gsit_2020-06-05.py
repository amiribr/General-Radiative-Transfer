import time
import numpy as np
#------------------------------------------------------------------------------
#
def gauszw(x1, x2, n):
#Goal:
#	-
#In:
#	-
#Out:
#	-
#Comments:
#	https://pomax.github.io/bezierinfo/legendre-gauss.html
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
# Test: compare vs known values
#ng = 64
#x, w = gauszw(-1.0, 1.0, ng)
#for i in range(ng):
#    print(i, x[i], w[i])
#print 'done!'
#------------------------------------------------------------------------------
#
def gsit(nit, ng1, nlr, dtau, ssa, xk):
    '''
    Task:
	    Solve RTE in a basic scenario using Gauss-Seidel (GS) iterations 
    In:
        nit    i       number of iterations, nit > 0
	    ng1    i       number of gauss nodes per hemisphere
        nlr    i       number of layer elements dtau, tau9 = dtau*nlr
        dtau   d       thickness of element layer (integration step over tau)
        ssa    d       single scattering albedo
        xk     d[nk]   expansion moments, (2k+1) included, nk=len(xk)   
    Out:
	    Itoa, Iboa   d[2*ng1]   intensity @ TOA & BOA
    Tree:
	    gsit()
            > gauszw() - computes Gauss zeros and weights 
    Comments:
        TOA scaling factor = 1.0;
    References:
	    1. -
    Revision History:
        2020-06-06:
            first created and tested vs RT3 [2] & IPOL [3]
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
    wj = np.concatenate((w, w))
#
#   Polynomilas Pk(x) & phaze function p @ all Gauss nodes, mug:
    pk = np.zeros((nk, ng2))
    pk[0, :] = 1.0
    pk[1, :] = mug
    pk[2, :] = 1.5*mug*mug - 0.5
    p = 1.0 + xk[1]*pk[1, :] + xk[2]*pk[2, :] # sum up 3 first moments explicetely (Rayleigh), run recursion afterwards
    for ik in range(3, nk):
        pk[ik, :] = (2.0 - 1.0/ik)*mug*pk[ik-1, :] - (1.0 - 1.0/ik)*pk[ik-1, :]
        p += xk[ik]*pk[ik, :]
#   include scaling factor for conveneince:
    p *= 0.5*ssa
#    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~????????????????????????????
#   Single scattering up & down @ all boundaries, ib:
    I1up = np.zeros((nb, ng1))
    I1dn = np.zeros_like(I1up)
    for ib in range(nb):
        taui = tau[ib]
        e0 = np.exp(-taui/mu0)
        I1up[ib, :] = p[0:ng1]*mu0/(mu0 + mup)*( e0 - np.exp(-(tau0 - taui)/mup - tau0/mu0) )
        I1dn[ib, :] = p[ng1:ng2]*mu0/(mu0 - mup)*( e0 - np.exp(-taui/mup) )
#
#   Single scattering up & down using layer dtau (alone)
    I11up = p[0:ng1]*mu0/(mu0 + mup)*(1.0 - np.exp(-dtau/mup - dtau/mu0))
    I11dn = p[ng1:ng2]*mu0/(mu0 - mup)*(np.exp(-dtau/mu0) - np.exp(-dtau/mup))
    Issdn = np.zeros_like(I1up)
    for ib in range(1, nb):
        taui = tau[ib]
        Issdn[ib, :] = Issdn[ib-1, :]*np.exp(-dtau/mup) + I11dn*np.exp(-tau[ib-1]/mu0)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~???????????????????????????
#
#   For multiple scattering:
    wpij = np.zeros((ng2, ng2)) # sum{xk*pk(mui)*pk(muj)*wj, k=0:nk}
    for ig in range(ng2):
        for jg in range(ng2):
            wpij[ig, jg] = 1.0 + xk[1]*pk[1, ig]*pk[1, jg] + xk[2]*pk[2, ig]*pk[2, jg]
            for ik in range(3, nk):
                wpij[ig, jg] += xk[ik]*pk[ik, ig]*pk[ik, jg]
            wpij[ig, jg] *= 0.5*ssa*wj[jg]
#      
#   all Gauss directions for upward, wpup, and downward, wpdn, directions:
    wpup = wpij[0:ng1, :]
    wpdn = wpij[ng1:ng2, :]
#
    Iup = np.copy(I1up)
    Idn = np.copy(I1dn)
    for itr in range(nit):
#       down
        Iup05 = 0.5*(Iup[0, :] + Iup[1, :]) 
        Idn05 = 0.5*(Idn[0, :] + Idn[1, :]) # Idn[0, :] = 0.0
        I05 = np.concatenate((Iup05, Idn05))
        J = np.matmul(wpdn, I05)
        Idn[1, :] = I11dn + (1.0 - np.exp(-dtau/mup))*J
        for ib in range(2, nb):
            Iup05 = 0.5*(Iup[ib-1, :] + Iup[ib, :]) 
            Idn05 = 0.5*(Idn[ib-1, :] + Idn[ib, :])
            I05 = np.concatenate((Iup05, Idn05))
            J = np.matmul(wpdn, I05)
            Idn[ib, :] = Idn[ib-1, :]*np.exp(-dtau/mup) + \
                             I11dn*np.exp(-tau[ib-1]/mu0) + \
                                 (1.0 - np.exp(-dtau/mup))*J
#       up
        Iup05 = 0.5*(Iup[nb-2, :] + Iup[nb-1, :]) # Iup[nb-1, :] = 0.0 
        Idn05 = 0.5*(Idn[nb-2, :] + Idn[nb-1, :]) # Idn[0, :] = 0.0
        I05 = np.concatenate((Iup05, Idn05))
        J = np.matmul(wpup, I05)
        Iup[nb-2, :] = I11up*np.exp(-tau[nb-2]) + (1.0 - np.exp(-dtau/mup))*J
        for ib in range(nb-3, -1, -1): # going up, ib = 0 (TOA) must be included
            Iup05 = 0.5*(Iup[ib, :] + Iup[ib+1, :]) 
            Idn05 = 0.5*(Idn[ib, :] + Idn[ib+1, :])
            I05 = np.concatenate((Iup05, Idn05))
            J = np.matmul(wpdn, I05)
            Iup[ib, :] = Iup[ib+1, :]*np.exp(-dtau/mup) + \
                             I11dn*np.exp(-tau[ib]/mu0) + \
                                 (1.0 - np.exp(-dtau/mup))*J
        print("iter=%i"%itr)
#   TOA & BOA:
    return mug[0:ng1], Iup[0,:], mug[ng1:ng2], Idn[nb-1, :] #Issdn[nb-1, :]-I1dn[nb-1, :]
#==============================================================================
#
if __name__ == "__main__":
#
    scalef = 0.5/np.pi             # TOA flux scaling factor, 1/2pi
#
    nit = 10                        # number of iterations: nit = 0 returns SS
    ng1 = 16                       # number of Gauss nodes per heimsphere
    nlr = 100                      # number of layer elements
    dtau = 0.01                    # integration step over tau
    ssa = 0.99999999               # single scattering albedo
    xk = np.array([1.0, 0.0, 0.5]) # expansion moments of phase function
#
    time_start = time.time()
#
    mutoa,Itoa,muboa,Iboa = gsit(nit, ng1, nlr, dtau, ssa, xk)
#
    time_end = time.time()
#
    for ig in range(ng1):
        print("mu=%.4f, I=%.4e,   mu=%.4f, I=%.4e,"
              %(mutoa[ig], scalef*Itoa[ig], muboa[ig], scalef*Iboa[ig]))
#
    print("gsit runtime = %.1f sec."%(time_end-time_start))