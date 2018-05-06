#Model fitting

import astropy.units as u
import naima as nai
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, special
from math import *
from lmfit import Minimizer, Parameters, report_fit


N1=50
N2=50
N3=50

eV2erg=1.6022e-12
pc2cm=3.086e18  
u_mag=(3e-6)**2.0/8.0/pi     # erg/cm3 (for 3microGauss)
u_rad=0.26*eV2erg            # erg/cm3 (for CMB at 2.7K)  
eps_rad=4.55e-10             # in m_e*c^2 (for CMB at 2.7K)
c=3e10                       # en cm/s
sigma_T=6.65e-25             # en cm2
mec2=511e3*eV2erg            # en erg  



D100=4.5e27/pc2cm**2 #pc2/s
delta=1./3.

dsrc=250 #parsecs
eta=0.3   #efficacite
L=3.26E34/eV2erg*1e-12    #energy loss rate
gamma=2.24
Q0=(eta*L*(-gamma+2))/(5e2**(-gamma+2)-1e-3**(-gamma+2))  # TeV^(gamma-1)/s
dmax=100

age_plsr=342000*365.25*24*3600 #en sec

data_geminga = np.loadtxt('Data.txt')

data=data_geminga[:,1]*10**(-12)
eps_data=data_geminga[:,2]*10**(-13)


Ee=np.logspace(log(0.1,10),log(500,10),N1) #energie des electrons
Eg=np.logspace(log(0.1,10),log(100,10),N2)  #energie des photons
deltad=dmax/N3
d=data_geminga[:,0]





def log_integral(x,y):
	res=0.0
	if x.size != y.size:
		print ('Incompatible arrays for log-integral calculation.')
	else:
		for i in range(1,x.size):
			res += (x[i]-x[i-1])/(x[i]+x[i-1])*(x[i]*y[i]+x[i-1]*y[i-1])
	return res


def M(l,d,D,rd,q0):
    return (q0*special.erfc(sqrt(l**2+d**2)/rd))/(sqrt(l**2+d**2)*4*pi*D)  # TeV^(gamma-1)/pc3

 
def fcn2min(params, d, data):    
    q0 = params['normalisation']
    d100 = params['coeffdediffusion']   
    gam=params['indicespect']
    
    ILV=[]    #integrale sur la ligne de visee
    SB2=[]     #surface brightness 

  
    for i in d:
        s=[]
        rayon=[]
        for j in Ee:
            D=d100*pow((j/1e2),delta)   # pc2/s
            #boucle pour choisir le temps le plus restrictif
            lorentz=j/0.511e-6 
            tcool=mec2/((4./3)*c*sigma_T*lorentz*(u_mag+u_rad/(1+4*lorentz*eps_rad)**1.5))   #s
            if tcool>=age_plsr:
                tcool = age_plsr
            rd=2*sqrt(D*tcool)
            #pc
            rayon.append(rd)
            m=integrate.quad(M,-dsrc,dsrc,args=(i,D,rd,q0))  #TeV^(gamma-1)/pc2
            s.append(m[0])
            
        s=np.asarray(s)*np.power(Ee,-gam)*(dsrc**2)*(pi/180.)**2.0  # /TeV/deg2
        ILV.append(s)
        TB=nai.models.TableModel(Ee*u.TeV, s/u.TeV, amplitude=1)  
        IC = nai.models.InverseCompton(TB, seed_photon_fields=['CMB'])
        flux_IC = IC.flux(Eg*u.TeV, distance=dsrc*u.pc)  #mettre IC.flux ou IC.sed en fct de ce que l'on veut
        flux_IC = flux_IC.to(u.TeV**-1*u.cm**-2*u.s**-1)
        idx=(Eg>=5) & (Eg<=50)
        sb2=log_integral(Eg[idx],Eg[idx]*flux_IC.value[idx])
        SB2.append(sb2)
        
    
    return (SB2 - data)/eps_data








# define objective function: returns the array to be minimized

params = Parameters()
params.add('normalisation',value=Q0,min=1E28,max=1E36)
params.add('coeffdediffusion', value=D100,min=0,max=1E-5)
params.add('indicespect',value=gamma,min=0.1,max=5,vary=False)


# do fit, here with leastsq model
minner = Minimizer(fcn2min, params, fcn_args=(d, data))
result = minner.minimize()
# calculate final result
final = data + result.residual*eps_data # write error report
report_fit(result)


print(result.success)
print(result.message)
#try to plot results
try:
    plt.errorbar(d, data*1E12, xerr=0, yerr=eps_data*1E12, fmt='k.') 
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)

    plt.plot(d, final*1E12, 'r')     
    ax1.set_xlim(0,12)  
    ax1.set_xlabel('Distance from pulsar [degree]')
    ax1.set_ylabel('Surface brightness [10^-12 TeV/cm2/s/deg2]')
    ax2 = ax1.twiny()  
    ax2.set_xlim(0,50) 
    ax2.set_xlabel('Distance from pulsar [pc]') 
    
    plt.show()
except ImportError: 
    pass


