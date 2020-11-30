#!/usr/bin/env python3
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import math
import matplotlib.font_manager
plt.rcParams['figure.figsize'] = [12,7]
plt.rcParams["font.family"] = "Times New Roman"

------------------------------------------------------------------------------
# 51 Pegasi b

Ag=0.5
Rp = 1.3583*(10**(8))
Rs = 8.6058*(10**(8))
a = 0.052*149.60*(10**(9))
theta = np.linspace(0,2*np.pi,1)
e = 0.0069
r = (a*(1-(e**2)))/(1+(e*(np.cos(theta))))
Ls = 1.3*3.827*(10**26)
Omega = Ag*(3/2)
------------------------------------------------------------------------------

# Fractional Flux with Plane Parallel Ray Approximation


N_Orbit = 1


PSI = np.zeros([1000,np.ceil(N_Orbit).astype('int')])
FF = np.zeros([1000,np.ceil(N_Orbit).astype('int')])

for j in range(1, np.ceil(N_Orbit).astype('int')+1):
    if j != np.ceil(N_Orbit).astype('int'):
        psi = np.linspace(2*(j-1)*(np.pi),2*j*np.pi,1000)
        ff = (np.sign((((2*j)-1)*(np.pi))-psi))*Ag*((Rp/r)**2)*(((((((2*j)-1)*(np.pi))-psi)*(np.cos(psi)))+(np.sin(psi)))/(4*np.pi))
        PSI[:,j-1] = psi
        FF[:,j-1] = ff
    elif(j == np.ceil(N_Orbit).astype('int') and np.modf(N_Orbit)[0] ==0):
        psi = np.linspace(2*(j-1)*(np.pi),2*(j)*np.pi,1000)
        ff = (np.sign((((2*j)-1)*(np.pi))-psi))*Ag*((Rp/r)**2)*(((((((2*j)-1)*(np.pi))-psi)*(np.cos(psi)))+(np.sin(psi)))/(4*np.pi))
        PSI[:,j-1] = psi
        FF[:,j-1] = ff    
    elif(j == np.ceil(N_Orbit).astype('int') and np.modf(N_Orbit)[0] !=0):
        psi = np.linspace(2*(j-1)*(np.pi),2*((j-1)+(np.modf(N_Orbit)[0]))*np.pi,1000)
        ff = (np.sign((((2*j)-1)*(np.pi))-psi))*Ag*((Rp/r)**2)*(((((((2*j)-1)*(np.pi))-psi)*(np.cos(psi)))+(np.sin(psi)))/(4*np.pi))
        PSI[:,j-1] = psi
        FF[:,j-1] = ff    
    plt.plot(psi,ff, 'k--')


plt.xlabel('Time / Phase Angle', fontsize=16)
plt.ylabel('Reflected Fractional Flux ($\mathfrak{F}$)', fontsize=16)
#plt.legend(fontsize=14)
plt.legend(['Reflected Light of a Planet over %s Orbits' %N_Orbit],bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.5, fontsize=14)
plt.show()



-----------------------------------------------------------------------------

# Luminosity with Plane Parallel Ray Approximation


N_Orbit = 1


PSI = np.zeros([1000,np.ceil(N_Orbit).astype('int')])
LUM = np.zeros([1000,np.ceil(N_Orbit).astype('int')])

for j in range(1, np.ceil(N_Orbit).astype('int')+1):
    if j != np.ceil(N_Orbit).astype('int'):
        psi = np.linspace(2*(j-1)*(np.pi),2*j*np.pi,1000)
        lum = (2/3)*(Ls)*(np.sign((((2*j)-1)*(np.pi))-psi))*Omega*((Rp/r)**2)*(((((((2*j)-1)*(np.pi))-psi)*(np.cos(psi)))+(np.sin(psi)))/(4*np.pi))
        PSI[:,j-1] = psi
        LUM[:,j-1] = lum
    elif(j == np.ceil(N_Orbit).astype('int') and np.modf(N_Orbit)[0] ==0):
        psi = np.linspace(2*(j-1)*(np.pi),2*(j)*np.pi,1000)
        lum = (2/3)*(Ls)*(np.sign((((2*j)-1)*(np.pi))-psi))*Omega*((Rp/r)**2)*(((((((2*j)-1)*(np.pi))-psi)*(np.cos(psi)))+(np.sin(psi)))/(4*np.pi))
        PSI[:,j-1] = psi
        LUM[:,j-1] = lum    
    elif(j == np.ceil(N_Orbit).astype('int') and np.modf(N_Orbit)[0] !=0):
        psi = np.linspace(2*(j-1)*(np.pi),2*((j-1)+(np.modf(N_Orbit)[0]))*np.pi,1000)
        lum = (2/3)*(Ls)*(np.sign((((2*j)-1)*(np.pi))-psi))*Omega*((Rp/r)**2)*(((((((2*j)-1)*(np.pi))-psi)*(np.cos(psi)))+(np.sin(psi)))/(4*np.pi))
        PSI[:,j-1] = psi
        LUM[:,j-1] = lum    
    plt.plot(psi,lum, 'g--')


plt.xlabel('Time / Phase Angle', fontsize=16)
plt.ylabel('Reflected Luminosity ($\mathcal{L}$)', fontsize=16)
#plt.legend(fontsize=14)
plt.legend(['Reflected Light of a Planet over %s Orbits' %N_Orbit],bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.5, fontsize=14)
plt.show()



-----------------------------------------------------------------------------

# Luminosity for Close-in Planets (Kopal, 1954)


N_Orbit = 1


PSI = np.zeros([1000,np.ceil(N_Orbit).astype('int')])
LUM_K = np.zeros([1000,np.ceil(N_Orbit).astype('int')])

for j in range(1, np.ceil(N_Orbit).astype('int')+1):
    if j != np.ceil(N_Orbit).astype('int'):
        psi = np.linspace(2*(j-1)*(np.pi),2*j*np.pi,1000)
        lum_k = Ls*(((2/3)*(np.sign((((2*j)-1)*(np.pi))-psi))*Omega*((Rp/r)**2)*(((((((2*j)-1)*(np.pi))-psi)*(np.cos(psi)))+(np.sin(psi)))/(4*np.pi))) + ((1/32)*((Rp/r)**3)*((3*(np.cos(psi))**2)+(2*np.cos(psi))-1)) + (((Rp/r)**4)*(((np.cos(psi))**2)*(np.sin(psi)))/(4*np.pi)) - (((((Rp*Rs)**0.5)/r)**4)*(np.sin(psi))/(4*np.pi)))
        PSI[:,j-1] = psi
        LUM_K[:,j-1] = lum_k
    elif(j == np.ceil(N_Orbit).astype('int') and np.modf(N_Orbit)[0] ==0):
        psi = np.linspace(2*(j-1)*(np.pi),2*(j)*np.pi,1000)
        lum_k = Ls*(((2/3)*(np.sign((((2*j)-1)*(np.pi))-psi))*Omega*((Rp/r)**2)*(((((((2*j)-1)*(np.pi))-psi)*(np.cos(psi)))+(np.sin(psi)))/(4*np.pi))) + ((1/32)*((Rp/r)**3)*((3*(np.cos(psi))**2)+(2*np.cos(psi))-1)) + (((Rp/r)**4)*(((np.cos(psi))**2)*(np.sin(psi)))/(4*np.pi)) - (((((Rp*Rs)**0.5)/r)**4)*(np.sin(psi))/(4*np.pi)))
        PSI[:,j-1] = psi
        LUM_K[:,j-1] = lum_k    
    elif(j == np.ceil(N_Orbit).astype('int') and np.modf(N_Orbit)[0] !=0):
        psi = np.linspace(2*(j-1)*(np.pi),2*((j-1)+(np.modf(N_Orbit)[0]))*np.pi,1000)
        lum_k = Ls*(((2/3)*(np.sign((((2*j)-1)*(np.pi))-psi))*Omega*((Rp/r)**2)*(((((((2*j)-1)*(np.pi))-psi)*(np.cos(psi)))+(np.sin(psi)))/(4*np.pi))) + ((1/32)*((Rp/r)**3)*((3*(np.cos(psi))**2)+(2*np.cos(psi))-1)) + (((Rp/r)**4)*(((np.cos(psi))**2)*(np.sin(psi)))/(4*np.pi)) - (((((Rp*Rs)**0.5)/r)**4)*(np.sin(psi))/(4*np.pi)))
        PSI[:,j-1] = psi
        LUM_K[:,j-1] = lum_k    
    plt.plot(psi,lum_k, 'b--')


plt.xlabel('Time / Phase Angle', fontsize=16)
plt.ylabel('Reflected Luminosity ($\mathcal{L}$)', fontsize=16)
#plt.legend(fontsize=14)
plt.legend(['Reflected Light of a Planet over %s Orbits (Kopal, 1954)' %N_Orbit],bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.5, fontsize=14)
plt.show()





-----------------------------------------------------------------------------

# Luminosity for Close-in Planets (Kopal, 1954) Edit Edit Edit Edit Edit 


N_Orbit = 1.25
Angle = np.arcsin((Rp+Rs)/r)
Angle_Deg = math.degrees(np.arcsin((Rp+Rs)/r))


Psi = []
Lum = []
Range = np.linspace(0,2*np.pi,1000)
for psi in Range:
    if (-Angle <= psi < Angle):
        lum = (Ls/4)*((Rp/r)**2)*(np.cos(psi))*((2/3)+((Rp+Rs)/(2*r)))
        Psi.append(psi)
        Lum.append(lum) 
    elif (Angle <= psi <= np.pi-Angle):
        lum = Ls*(((2/3)*((Rp/r)**2)*(((np.pi-psi)*(np.cos(psi)))+(np.sin(psi)))/(4*np.pi))+((1/32)*((Rp/r)**3)*((3*((np.cos(psi))**2))+(2*np.cos(psi))-1))+(((Rp/r)**4)*((np.sin(psi))*((np.cos(psi))**2))/(4*np.pi))-(((((Rp*Rs)**0.5)/(r))**4)*(np.sin(psi))/(4*np.pi)))
        Psi.append(psi)
        Lum.append(lum)
    elif (np.pi-Angle < psi <= np.pi+Angle):
        lum = Ls*(((2/3)*((Rp/r)**2)*(((np.pi-psi)*(np.cos(psi)))+(np.sin(psi)))/(4*np.pi))+((1/32)*((Rp/r)**3)*((3*((np.cos(psi))**2))+(2*np.cos(psi))-1))+(((Rp/r)**4)*((np.sin(psi))*((np.cos(psi))**2))/(4*np.pi))-(((((Rp*Rs)**0.5)/(r))**4)*(np.sin(psi))/(4*np.pi)))
        #lum = Ls*((2/3)*((Rp/r)**2)*(((np.pi-psi)*(np.cos(psi)))+(np.sin(psi)))/(4*np.pi))
        Psi.append(psi)
        Lum.append(lum)
        
Psi = np.asarray(Psi)
Lum = np.squeeze(np.asarray(Lum))   
lum2 = Ls*((2/3)*((Rp/r)**2)*(((np.pi-Psi)*(np.cos(Psi)))+(np.sin(Psi)))/(4*np.pi)) 
plt.plot(Psi,lum2, 'r--')
plt.plot(Psi,Lum, 'k--')
plt.show()




FD = (Lum/lum2)-1
plt.plot(Psi,FD, 'c--')
plt.legend(['Fractional Difference'],bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.5, fontsize=14)
plt.xlabel('Time / Phase Angle', fontsize=16)
plt.ylabel('Frac. Difference', fontsize=16)

plt.show()











plt.xlabel('Time / Phase Angle', fontsize=16)
plt.ylabel('Reflected Luminosity ($\mathcal{L}$)', fontsize=16)
#plt.legend(fontsize=14)
plt.legend(['Reflected Light of a Planet over %s Orbits (Kopal, 1954)' %N_Orbit],bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.5, fontsize=14)
plt.show()


-----------------------------------------------------------------------------
# Check
x1 = np.linspace(0,0.12933666,1000)
x2 = np.linspace(0.12933666, 2*np.pi,1000)
a = (((np.pi-x)*(np.cos(x)))+(np.sin(x)))/(np.pi)
n = np.cos(x)
plt.plot(x,m, 'm--')
plt.plot(x,n, 'c--')
plt.show()

-----------------------------------------------------------------------------

# Fractional Difference


FD = (LUM_K/LUM)-1
plt.plot(PSI,FD, 'r--')

plt.legend(['Fractional Difference'],bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.5, fontsize=14)
plt.xlabel('Time / Phase Angle', fontsize=16)
plt.ylabel('Frac. Difference', fontsize=16)

plt.show()







-----------------------------------------------------------------------------




