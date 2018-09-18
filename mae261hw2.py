#! python3.7
import numpy as np
import rungekutta as rk
from matplotlib import pyplot as plt

# Define Constants
O2 = 209500.0
CO2 = 355.0
M = 780300.0
H2O = 2000.0

# Time dependent reaction rate
def kr1(t):
    # t in minutes
    if t <= 600:
        return 0.2053*(t/60.0) - 0.02053*(t/60.0)**2
    else:
        return 0

# Rates (ppm^-1*min^-1)
kr2  = 2.183e-05
kr3  = 26.6
kr4  = 1.2e+04
kr5  = 1.6e+04
kr6  = 1.3e+04
kr7  = 1.1e+04
kr8  = 2.1e+03
kr9  = 2.6e-02
kr10 = 0.034

# PSSA Species
def pssaO(t, NO2):
    return kr1(t)*NO2/(kr2*O2*M)

def pssaOH(HO2, NO, CH2OH, NO2, C2H4):
    nom = kr4*HO2*NO + kr8*CH2OH*O2
    den = kr5*NO2 + kr6*C2H4
    #return nom/den if den > 0.0 else 0.0
    return nom/den

def pssaHO2(HOCH2CH2O2, NO, C2H4, O3):
    nom = 0.28*kr7*HOCH2CH2O2*NO + 0.12*kr9*C2H4*O3
    den = kr4*NO
    #return nom/den if den > 0.0 else 0.0
    return nom/den

def pssaHOCH2CH2O2(C2H4, OH, NO):
    nom = kr6*C2H4*OH
    den = kr7*NO
    #return nom/den if den > 0.0 else 0.0
    return nom/den

def pssaCH2OH(HOCH2CH2O2, NO):
    nom = 0.72*kr7*HOCH2CH2O2*NO
    den = kr8*O2
    #return nom/den if den > 0.0 else 0.0
    return nom/den

# Active Species DiffEq Functions
def diffC2H4(t, C2H4, OH, O3):
    return -kr6*C2H4*OH - kr9*C2H4*O3

def diffO(t, O, NO2):
    return kr1(t)*NO2 - kr2*O*O2*M

def diffO3(t, O3, O, NO, C2H4):
    return kr2*O*O2*M - kr3*O3*NO - kr9*C2H4*O3

def diffNO(t, NO, NO2, O3, HOCH2CH2O2, HO2):
    return kr1(t)*NO2 - kr3*O3*NO - kr4*HO2*NO - kr7*HOCH2CH2O2*NO

def diffNO2(t, NO2, NO, O3, HO2, OH, HOCH2CH2O2):
    return -kr1(t)*NO2 + kr3*NO*O3 + kr4*NO*HO2 - kr5*OH*NO + kr7*HOCH2CH2O2*NO

def diffHNO3(t, HNO3, OH, NO2):
    return kr5*OH*NO2

def diffHCHO(t, HCHO, HOCH2CH2O2, NO, CH2OH, NO2, C2H4, O3):
    return 0.72*kr7*HOCH2CH2O2*NO + kr8*CH2OH*O2 + kr9*C2H4*O3

def diffHOCH2CHO(t, HOCH2CHO, HOCH2CH2O2, NO):
    return 0.28*kr7*HOCH2CH2O2*NO

def diffH2COO(t, H2COO, C2H4, O3):
    return 0.4*kr9*C2H4*O3 - kr10*H2COO*H2O

def diffCO(t, CO, C2H4, O3):
    return 0.42*kr9*C2H4*O3

def diffH2(t, H2, C2H4, O3):
    return 0.12*kr9*C2H4*O3

def diffHCOOH(t, HCOOH, H2COO):
    return kr10*H2COO*H2O

# Some helpful dictionaries
pssaFunc = dict(O     = pssaO,
                OH    = pssaOH,
                HO2   = pssaHO2,
                CH2OH = pssaCH2OH,
                HOCHb = pssaHOCH2CH2O2)

diffFunc = dict(C2H4  = diffC2H4,
                O3    = diffO3,
                NO    = diffNO,
                NO2   = diffNO2,
                HNO3  = diffHNO3,
                HCHO  = diffHCHO,
                H2COO = diffH2COO,
                CO    = diffCO,
                H2    = diffH2,
                HCOOH = diffHCOOH,
                HOCHa = diffHOCH2CHO) #HOCH2CHO

diffArgs = dict(C2H4  = ['OH', 'O3'],
                O3    = ['O', 'NO', 'C2H4'],
                NO    = ['NO2', 'O3', 'HOCHb', 'HO2'],
                NO2   = ['NO', 'O3', 'HO2', 'OH', 'HOCHb'],
                HNO3  = ['OH', 'NO2'],
                HCHO  = ['HOCHb', 'NO', 'CH2OH', 'NO2', 'C2H4', 'O3'],
                H2COO = ['C2H4', 'O3'],
                CO    = ['C2H4', 'O3'],
                H2    = ['C2H4', 'O3'],
                HCOOH = ['H2COO'],
                HOCHa = ['HOCHb', 'NO']) #HOCH2CHO

ind = dict(C2H4  = 0,
           O     = 1,
           O3    = 2,
           OH    = 3,
           NO    = 4,
           NO2   = 5,
           HO2   = 6,
           HNO3  = 7,
           HCHO  = 8,
           CH2OH = 9,
           H2COO = 10,
           CO    = 11,
           H2    = 12,
           HCOOH = 13,
           HOCHa = 14, #HOCH2CHO
           HOCHb = 15) #HOCH2CH2O2

# Single Function for all diff eqs
# t - current time of simulation
# Sys - dictionary
def diffSys(t, SysA, SysB):
    #SysA has concs at start of t
    #SysA gets modified by rk4 (yn+k1*dt/2)
    #SysB has SysA + PSSA concs for t
    #SysB is NOT modified by rk4
    #Evaluate all diff eqs
    outSys = np.zeros_like(SysA)
    for spec,func in diffFunc.items():
        fargs = [SysB[ind[args]] for args in diffArgs[spec]]
        outSys[ind[spec]] = func(t,SysA[ind[spec]],*fargs)
    return outSys

# Solving Procedure
def solve():
    # Simulation parameters
    tn = 0.0 #minutes
    dt = 1.0e-3 #minutes
    tm = 12.0*60.0 #minutes
    st = int(tm/dt)
    N = 16
    # All initialize to zero
    Sys = np.zeros((N,st))
    # Initialize (ppm)
    Sys[ind['C2H4'],0] = 3.0
    Sys[ind['NO'],0]   = 0.375
    Sys[ind['NO2'],0]  = 0.125
    
    # Loop through the simulation period
    tn += dt
    i = 1
    while(i<st):
        #i = int(tn/dt)
        j = i-1

        # Gather PSSA 3ependents for easier reading
        argO     = [tn, Sys[ind['NO2'],j]]
        argHO2   = [Sys[ind['HOCHb'], j], Sys[ind['NO'], j], \
                    Sys[ind['C2H4'],  j], Sys[ind['O3'], j]]
        argOH    = [Sys[ind['HO2'],   j], Sys[ind['NO'], j], \
                    Sys[ind['CH2OH'], j], Sys[ind['NO2'],j], \
                    Sys[ind['C2H4'],  j]]
        argHOCHb = [Sys[ind['C2H4'],  j], Sys[ind['OH'], j], \
                    Sys[ind['NO'],    j]]
        argCH2OH = [Sys[ind['HOCHb'], j], Sys[ind['NO'], j]]
        
        # Solve PSSA
        pssas = np.copy(Sys[:,j])
        pssas[ind['O']]     = pssaO(*argO)
        pssas[ind['HO2']]   = pssaHO2(*argHO2)
        pssas[ind['OH']]    = pssaOH(*argOH)
        pssas[ind['HOCHb']] = pssaHOCH2CH2O2(*argHOCHb)
        pssas[ind['CH2OH']] = pssaCH2OH(*argCH2OH)

        # Step Diff Eqs
        diffs = rk.rk4(tn, Sys[:,j], dt, diffSys, [pssas])
        
        # Combine Results
        for spec in pssaFunc.keys():
            Sys[ind[spec],i] = pssas[ind[spec]]
        for spec in diffFunc.keys():
            Sys[ind[spec],i] = diffs[ind[spec]]

        # print("")
        # print("Current State")
        # print("i:          {0:11d}".format(i))
        # print("tn:         {0:11.7f}".format(tn))
        # print("C2H4:       {0:11.7f}".format(Sys[ind['C2H4'],i]))
        # print("O:          {0:11.7f}".format(Sys[ind['O'],i]))
        # print("O3:         {0:11.7f}".format(Sys[ind['O3'],i]))
        # print("NO:         {0:11.7f}".format(Sys[ind['NO'],i]))
        # print("NO2:        {0:11.7f}".format(Sys[ind['NO2'],i]))
        # print("OH:         {0:11.7f}".format(Sys[ind['OH'],i]))
        # print("HO2:        {0:11.7f}".format(Sys[ind['HO2'],i]))
        # print("HNO3:       {0:11.7f}".format(Sys[ind['HNO3'],i]))
        # print("HCHO:       {0:11.7f}".format(Sys[ind['HCHO'],i]))
        # print("CH2OH:      {0:11.7f}".format(Sys[ind['CH2OH'],i]))
        # print("H2COO:      {0:11.7f}".format(Sys[ind['H2COO'],i]))
        # print("CO:         {0:11.7f}".format(Sys[ind['CO'],i]))
        # print("H2:         {0:11.7f}".format(Sys[ind['H2'],i]))
        # print("HCOOH:      {0:11.7f}".format(Sys[ind['HCOOH'],i]))
        # print("HOCH2CHO:   {0:11.7f}".format(Sys[ind['HOCHa'],i]))
        # print("HOCH2CH2O2: {0:11.7f}".format(Sys[ind['HOCHb'],i]))
        # print("O2:         {0:11.0f}".format(O2))
        # print("H2O:        {0:11.0f}".format(H2O))
        # print("CO2:        {0:11.0f}".format(CO2))
        # print("M:          {0:11.0f}".format(M))
        # print("k1:         {0:11.9f}".format(kr1(tn)))
        # print("")
        tn += dt
        i  += 1

    # It's over!!!
    np.savetxt("Prob4.csv",Sys.T,delimiter=",")
    print("ITS OVER")
    # Plot the results
    # Time Vector
    time = np.arange(0.0, tm, dt)
    # Ozone
    plt.plot(time, Sys[ind['O3'],:])
    plt.xlabel('Time [mins]')
    plt.ylabel('Concentration of Ozone (O3) [ppm]')
    plt.suptitle('Ozone: C2H2=3.0, NO=0.375, NO2=0.125 [ppm]')
    plt.show()
    # # Nitric Oxide
    # Sys2 = np.copy(Sys)
    # Sys2[:,np.where(~Sys2.any(axis=0))[0]] = Sys2[:,np.where(~Sys2.any(axis=0))[0]-1]
    # plt.plot(time, Sys2[ind['O3'],:])
    # plt.xlabel('Time [mins]')
    # plt.ylabel('Concentration of Ozone (O3) [ppm]')
    # plt.suptitle('Ozone: C2H2=3.0, NO=0.375, NO2=0.125 [ppm]')
    # plt.show()
    # plt.plot(time, Sys[ind['NO'],:])
    # plt.xlabel('Time [mins]')
    # plt.ylabel('Concentration of Nitric Oxide (NO) [ppm]')
    # plt.suptitle('Nitric Oxide (C2H4_0 = 1ppm)')
    # plt.show()
    # # Nitrogen Dioxide
    # plt.plot(time, Sys[ind['NO2'],:])
    # plt.xlabel('Time [mins]')
    # plt.ylabel('Concentration of Nitrogen Dioxide (NO2) [ppm]')
    # plt.suptitle('Nitrogen Dioxide (C2H4_0 = 1ppm)')
    # plt.show()
    # # Ethane
    # plt.plot(time, Sys[ind['C2H4'],:])
    # plt.xlabel('Time [mins]')
    # plt.ylabel('Concentration of Ethane (C2H4) [ppm]')
    # plt.suptitle('Ethane (C2H4_0 = 1ppm)')
    # plt.show()
    # # Hydroxyl
    # plt.plot(time, Sys[ind['OH'],:])
    # plt.xlabel('Time [mins]')
    # plt.ylabel('Concentration of Hydroxyl (OH) [ppm]')
    # plt.suptitle('Hydroxyl (C2H4_0 = 1ppm)')
    # plt.show()
    # # Nitric Acid
    # plt.plot(time, Sys[ind['HNO3'],:])
    # plt.xlabel('Time [mins]')
    # plt.ylabel('Concentration of Nitric Acid (HNO3) [ppm]')
    # plt.suptitle('Nitric Acid (C2H4_0 = 1ppm)')
    # plt.show()

solve()