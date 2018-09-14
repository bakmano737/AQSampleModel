#####################################
# A couple of Runge Kutta Methods   #
# Implementation by James V. Soukup #
# For use in CEE 290 at UCI         #
#####################################

import numpy as np

########################################
# The Runge Kutta Fourth Order Method  #
#   This function takes one step with  #
#   the given interval using RK4       #
########################################
########################################
# Runge Kutta Fourth Order: Integrator #
########################################
# Input Dictionary                     #
#    tn - Current Simulation Time      #
#    yn - Current Simulation Value     #
#    dt - Time step to next simulation #
#     f - Model Function               #
# fargs - Model Function Arguments     #
########################################
# This Integrator expects a particular #
# model function type. The model must  #
# represent a differential equation of #
# the form: y' = f(t,y)                #
# Therefore the function must take as  #
# input the current simulation value,  #
# the current simulation time, and any #
# additional inputs. It must output y' #
# for the given time and estimate      #
########################################
def rk4(tn,yn,dt,f,fargs):
    k1 = dt*f(tn,       yn,       *fargs)
    k2 = dt*f(tn+dt/2.0,yn+k1/2.0,*fargs)
    k3 = dt*f(tn+dt/2.0,yn+k2/2.0,*fargs)
    k4 = dt*f(tn+dt,    yn+k3,    *fargs)
    yo = yn+(1.0/6.0)*(k1+2.0*k2+2.0*k3+k4)
    return yo

########################################
# Runge Kutta Fehlberg 45 Integrator   #
########################################
# Input Dictionary                     #
#    tn - Current Simulation Time      #
#    yn - Current Simulation Value     #
#    dt - Time step to next simulation #
#     f - Model Function               #
# fargs - Model Function Arguments     #
########################################
# Compute the slope at 6 near points   #
# Create a fourth order approximation  #
# with k1,k3,k4,k5 and a fifth order   #
# approximation with k1,k3,k4,k5,k6.   #
# Use the second approximation to find #
# the error of the first approximation #
# Use the error to determine if the    #
# step is acceptable, if not, remove   #
# the approximation, reduce the step   #
# size, and begin again. If the error  #
# is acceptable keep the approximation #
# if the error is sufficiently small,  #
# increase the step size to improve    #
# performance and efficiency.          #
########################################
def rkf45(tn,yn,dt,f,fargs):
    # Coefficiencts
    c2  = 1.0/4.0
    a21 = 1.0/4.0
    c3  = 3.0/8.0
    a31 = 3.0/32.0
    a32 = 9.0/32.0
    c4  = 12.0/13.0
    a41 = 1932.0/2197.0
    a42 = -7200.0/2197.0
    a43 = 7296.0/2197.0
    c5  = 1.0
    a51 = 439.0/216.0
    a52 = -8.0
    a53 = 3680.0/513.0
    a54 = -845.0/4104.0
    c6  = 1.0/2.0
    a61 = -8.0/27.0
    a62 = 2.0
    a63 = -3544.0/2565.0
    a64 = 1859.0/4104.0
    a65 = -11.0/40.0
    by1 = 25.0/216.0
    by3 = 1408.0/2565.0
    by4 = 2197.0/4104.0
    by5 = -1.0/5.0
    bz1 = 16.0/135.0
    bz3 = 6656.0/12825.0
    bz4 = 28561.0/56430.0
    bz5 = -9.0/50.0
    bz6 = 2.0/55.0
    tol = 0.001
    dtm = 0.0001
    # First k value
    t1 = tn
    y1 = yn
    k1 = f(t1,y1,*fargs)
    # Second k value
    t2 = tn + c2*dt
    y2 = yn + dt*(a21*k1)
    k2 = f(t2,y2,*fargs)
    # Third k value
    t3 = tn + c3*dt
    y3 = yn + dt*(a31*k1 + a32*k2)
    k3 = f(t3,y3,*fargs)
    # Fourth k value
    t4 = tn + c4*dt
    y4 = yn + dt*(a41*k1 + a42*k2 + a43*k3)
    k4 = f(t4,y4,*fargs)
    # Fifth k value
    t5 = tn + c5*dt
    y5 = yn + dt*(a51*k1 + a52*k2 + a53*k3 + a54*k4)
    k5 = f(t5,y5,*fargs)
    # Sixth k value
    t6 = tn + c6*dt
    y6 = yn + dt*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5)
    k6 = f(t6,y6,*fargs)
    # Fourth Order Approximation
    yk = yn + dt*(by1*k1 + by3*k3 + by4*k4 + by5*k5)
    # Fifth Order Approximation
    zk = yn + dt*(bz1*k1 + bz3*k3 + bz4*k4 + bz5*k5 + bz6*k6)
    # Optimal Step
    df = abs(zk-yk) 
    if df > 0:
        s=((tol*dt)/(2.0*df))**(0.25)
    else:
        s=4
    if s<1:
        # Error is too large. Reduce step
        if (0.8*dt)>=dtm :
            dt=0.8*dt 
            yo=yn
            to=tn
        else:
            dt=dtm
            yo=yk
            to=tn+dt
    elif s>2:
        # Error is too small. Increase step
        to=tn+dt
        yo=yk
        dt=1.25*dt
    else:
        # Goldilocks
        to=tn+dt
        yo=yk
    return [dt,to,yo]
	