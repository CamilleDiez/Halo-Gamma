#Intégration de éq 7 pour vérifier surface brightness de l'article


from math import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

"""
Valeurs numériques des constantes :
"""


D100=4.5*10**(27)/(3.086*10**(18))**2 #pc²/s
delta=1/3
te=10000*365.25*24*3600 #s
dsrc=250 #parsecs
N0=13.6*10**(-15) #photons/TeV/cm²/s
alpha=2.34
thetad=(180/(pi*dsrc))*2*np.sqrt(D100*te) #theta en fct de Ee en degrés
theta=np.linspace(0,12,50) 
rd = theta*pi*(dsrc/180) #relation entre les deux abscisses

data_geminga = np.loadtxt('Data.txt')
data=data_geminga[:,1]#*10**(-12)
eps_data=data_geminga[:,2]/10
d=data_geminga[:,0]*(pi*(dsrc/180))

Y=[] #intégrale du flux en energie


def N(E,theta): 
    return E*N0*(10**12)*pow((E/20),(-alpha))*1.22*exp(-(theta**2)/(thetad**2))/(pi**(3/2)*thetad*(theta+0.06*thetad))



for i in theta:    
    I=integrate.quad(N,5,50,args=(i)) #intégrale du flux en energie     
    Y.append(I[0])
    



fig = plt.figure()  
ax1 = fig.add_subplot(111)

ax1.plot(theta, Y,'r-')

ax1.set_xlim(0,12)
ax1.set_xlabel('Distance from pulsar [degree]')
ax1.set_ylabel('Surface brightness [10^-12 TeV/cm2/s/deg2]')

ax2 = ax1.twiny()
ax2.set_xlim(0, 50)
ax2.set_xlabel("Distance from pulsar [pc]")


plt.errorbar(d, data, xerr=0, yerr=eps_data, fmt='k.') 


