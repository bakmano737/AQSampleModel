import numpy as np
import rungekutta as rk
from matplotlib import pyplot as plt

# Define Constants
O2 = 209500.0
CO2 = 355.0
M = 780300.0
H2O = 2000.0

# Time dependent reaction rate
def k1(t):
    # t in minutes
    if t <= 600:
        return 0.2053*(t/60.0) - 0.02053*(t/60.0)**2
    else:
        print("NIGHT")
        return 0

# Rates (ppm^-1*min^-1)
k2  = 2.183e-05
k3  = 26.6
k4  = 1.2e+04
k5  = 1.6e+04
k6  = 1.3e+04
k7  = 1.1e+04
k8  = 2.1e+03
k9  = 2.6e-02
k10 = 0.034

# PSSA Species
def pssaO(t, NO2):
    return k1(t)*NO2/(k2*O2*M)

def pssaOH(HO2, NO, CH2OH, NO2, C2H4):
    nom = k4*HO2*NO + k8*CH2OH*O2
    den = k5*NO2 + k6*C2H4
    return nom/den

def pssaHO2(HOCH2CH2O2, NO, C2H4):
    nom = 0.28*k7*HOCH2CH2O2*NO + 0.12*k9*C2H4*O2
    den = k4*NO
    return nom/den

def pssaHOCH2CH2O2(C2H4, OH, NO):
    return k6*C2H4*OH/(k7*NO)

def pssaCH2OH(HOCH2CH2O2, NO):
    return 0.72*k7*HOCH2CH2O2*NO/(k8*O2)

# Active Species DiffEq Functions
def diffC2H4(t, C2H4, OH, O3):
    return -k6*C2H4*OH - k9*C2H4*O3

def diffO(t, O, NO2):
    return k1(t)*NO2 - k2*O*O2*M

def diffO3(t, O3, O, NO, C2H4):
    return k2*O*O2*M - k3*O3*NO - k9*C2H4*O3

def diffNO(t, NO, NO2, O3, HOCH2CH2O2, HO2):
    return k1(t)*NO2 - k3*O3*NO - k4*HO2*NO - k7*HOCH2CH2O2*NO

def diffNO2(t, NO2, NO, O3, HO2, OH, HOCH2CH2O2):
    return -k1(t)*NO2 + k3*NO*O3 + k4*NO*HO2 - k5*OH*NO + k7*HOCH2CH2O2*NO

def diffHNO3(t, HNO3, OH, NO2):
    return k5*OH*NO2

def diffHCHO(t, HCHO, HOCH2CH2O2, NO, CH2OH, NO2, C2H4, O3):
    return 0.72*k7*HOCH2CH2O2*NO + k8*CH2OH*O2 + k9*C2H4*O3

def diffHOCH2CHO(t, HOCH2CHO, HOCH2CH2O2, NO):
    return 0.28*k7*HOCH2CH2O2*NO

def diffH2COO(t, H2COO, C2H4, O3):
    return 0.4*k9*C2H4*O3 - k10*H2COO*H2O

def diffCO(t, CO, C2H4, O3):
    return 0.42*k9*C2H4*O3

def diffH2(t, H2, C2H4, O3):
    return 0.12*k9*C2H4*O3

def diffHCOOH(t, HCOOH, H2COO):
    return k10*H2COO*H2O

# Solving Procedure
def solve():
    # Simulation parameters
    tn = 0.0 #minutes
    dt = 1.0e-4 #minutes
    tm = 12.0*60.0 #minutes
    st = int(tm/dt)
    # All initialize to zero
    C2H4  = np.zeros(st)
    O     = np.zeros(st)
    O3    = np.zeros(st) 
    OH    = np.zeros(st)
    NO    = np.zeros(st)
    NO2   = np.zeros(st)
    HO2   = np.zeros(st)
    HNO3  = np.zeros(st)
    HCHO  = np.zeros(st)
    CH2OH = np.zeros(st)
    H2COO = np.zeros(st)
    CO    = np.zeros(st)
    H2    = np.zeros(st)
    HCOOH = np.zeros(st)
    HOCH2CHO   = np.zeros(st)
    HOCH2CH2O2 = np.zeros(st)
    # Initialize (ppm)
    C2H4[0] = 3.0
    NO[0]   = 0.375
    NO2[0]  = 0.125
    
    # Loop through the simulation period
    # Array index - i
    i = 1
    tn = tn + dt
    #print(C2H4[0])
    #print(HOCH2CH2O2[0])
    #print(H2[10])
    #print(HOCH2CH2O2[i-1])
    #print(HO2.size)
    while(tn<=tm):
        # Solve PSSA
        #print(HOCH2CH2O2[0])
        #print(NO[i-1])
        #print(C2H4[i-1])
        O[i]   = pssaO(tn,NO2[i-1])
        HO2[i] = pssaHO2(HOCH2CH2O2[i-1], NO[i-1], C2H4[i-1])
        OH[i]  = pssaOH(HO2[i-1], NO[i-1], CH2OH[i-1], NO2[i-1], C2H4[i-1])
        HOCH2CH2O2[i] = pssaHOCH2CH2O2(C2H4[i-1], OH[i-1], NO[i-1])
        CH2OH[i] = pssaCH2OH(HOCH2CH2O2[i-1], NO[i-1])
        # Step Diff Eqs
        # These argument lists are simply to make the rk4 calls cleaner
        argNO2      = [NO[i-1], O3[i-1], HO2[i-1], OH[i-1], HOCH2CH2O2[i-1]]
        argNO       = [NO2[i-1], O3[i-1], HOCH2CH2O2[i-1], HO2[i-1]]
        argO3       = [O[i-1], NO[i-1], C2H4[i-1]]
        argHNO3     = [OH[i-1], NO2[i-1]]
        argC2H4     = [OH[i-1], O3[i-1]]
        argHCHO     = [HOCH2CH2O2[i-1], NO[i-1], CH2OH[i-1], NO2[i-1], C2H4[i-1], O3[i-1]]
        argH2COO    = [C2H4[i-1], O3[i-1]]
        argH2       = [C2H4[i-1], O3[i-1]]
        argCO       = [C2H4[i-1], O3[i-1]]
        argHOCH2CHO = [HOCH2CH2O2[i-1], NO[i-1]]

        NO2[i]      = rk.rk4(tn, NO2[i-1],   dt, diffNO2, argNO2)
        NO[i]       = rk.rk4(tn, NO[i-1],    dt, diffNO, argNO)
        #O[i]        = rk.rk4(tn, O[i-1],     dt, diffO, [NO2[i-1]])
        O3[i]       = rk.rk4(tn, O3[i-1],    dt, diffO3, argO3)
        H2[i]       = rk.rk4(tn, H2[i-1],    dt, diffH2, argH2)
        CO[i]       = rk.rk4(tn, CO[i-1],    dt, diffCO, argCO)
        HNO3[i]     = rk.rk4(tn, HNO3[i-1],  dt, diffHNO3, argHNO3)
        C2H4[i]     = rk.rk4(tn, C2H4[i-1],  dt, diffC2H4, argC2H4)
        HCHO[i]     = rk.rk4(tn, HCHO[i-1],  dt, diffHCHO, argHCHO)
        H2COO[i]    = rk.rk4(tn, H2COO[i-1], dt, diffH2COO, argH2COO)
        HCOOH[i]    = rk.rk4(tn, HCOOH[i-1], dt, diffHCOOH, [H2COO[i-1]])
        HOCH2CHO[i] = rk.rk4(tn, HOCH2CHO[i-1], dt, diffHOCH2CHO, argHOCH2CHO)

        print("")
        print("Current State")
        print("i:          {0:11d}".format(i))
        print("tn:         {0:11.4f}".format(tn))
        print("C2H4:       {0:11.4f}".format(C2H4[i]))
        print("O:          {0:11.4f}".format(O[i]))
        print("O2:         {0:11.4f}".format(O2))
        print("O3:         {0:11.4f}".format(O3[i]))
        print("NO:         {0:11.4f}".format(NO[i]))
        print("NO2:        {0:11.4f}".format(NO2[i]))
        print("OH:         {0:11.4f}".format(OH[i]))
        print("HO2:        {0:11.4f}".format(HO2[i]))
        print("HNO3:       {0:11.4f}".format(HNO3[i]))
        print("HCHO:       {0:11.4f}".format(HCHO[i]))
        print("CH2OH:      {0:11.4f}".format(CH2OH[i]))
        print("H2COO:      {0:11.4f}".format(H2COO[i]))
        print("CO2:        {0:11.4f}".format(CO2))
        print("CO:         {0:11.4f}".format(CO[i]))
        print("H2:         {0:11.4f}".format(H2[i]))
        print("H2O:        {0:11.4f}".format(H2O))
        print("HCOOH:      {0:11.4f}".format(HCOOH[i]))
        print("M:          {0:11.4f}".format(M))
        print("HOCH2CHO:   {0:11.4f}".format(HOCH2CHO[i]))
        print("HOCH2CH2O2: {0:11.4f}".format(HOCH2CH2O2[i]))
        print("k1:         {0:11.9f}".format(k1(tn)))
        print("")

        # Take a step
        tn += dt
        i  += 1

    # Plot the results
    # Time Vector
    time = np.arange(0.0, tm, dt)
    # Ozone
    plt.plot(time, O3)
    # Nitric Oxide
    plt.plot(time, NO)
    # Nitrogen Dioxide
    # Ethane
    # Hydroxyl
    # Nitric Acid

solve()