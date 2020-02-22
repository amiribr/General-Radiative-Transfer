import time
import numpy as np
from sglscat import sglscat
#
if __name__ == "__main__":
#
    nskip_txt = 5
#
#   TEST 001: Single scattering
#
#   -Input:
    tau0 = 0.2
    ssa = 1.0
    mu0 = 0.6
    mu = np.array([-0.6, 0.6])
    az = np.array([0.0, 90.0, 180.0])
    xkfile = 'xk0117_0804.txt'
    solar_flux_toa = np.pi
    bmfile = 'bmark001.txt'
#   -Auxiliary steps: convert to radians, extract phase function moments from matrix, etc
    phi = np.radians(az)
    xk6 = np.loadtxt(xkfile, comments='#', skiprows=nskip_txt) # k, a1k:a4k, b1k, b2k
    xk = xk6[:, 1]
#   -Main code executed & timed here
    time_start = time.time()
    I1 = solar_flux_toa*sglscat(mu0, mu, phi, tau0, ssa, xk)
    time_end = time.time()
#   -Read benchmark (see txt file for reference)
    bmark = np.loadtxt(bmfile, comments='#', skiprows=nskip_txt) # 0:sza, 1:mu0=cos(sza), 2:az, 3:vza, 4:mu=-cos(vza), 5:I1, 6:Q1, 7:U1
    Ibm = bmark[:, 5]
#   -Test vs bmark & print results
    print()
    ibmark = 0
    for iaz,azi in enumerate(az):
        for imu, mui in enumerate(mu):
            Ib = Ibm[ibmark]
            err = 100.0*np.abs(1.0 - I1[iaz,imu]/Ib)
            print('az = %7.2f   mu = %6.2f   I = %10.4e   err = %4.2f o/o' %(azi, mui, I1[iaz,imu], err))
            ibmark += 1
    print("test 001: runtime = %.1f sec."%(time_end-time_start))
    print()
#------------------------------------------------------------------------------
#
#   TEST 002: Single scattering
#

#==============================================================================