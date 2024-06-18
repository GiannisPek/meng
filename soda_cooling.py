import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
import pandas as pd
import warnings


warnings.filterwarnings('ignore')

def vaporP(T: float, fluid: str)-> float:
    """ 
    Function that caclulates the vapor pressure of the water and ehtanol.
    Inputs:
    T (float): ambient temperature [K]
    fluid (str): "water" or "ethanol"

    Output:
    Pv (float): vapor pressure at the given ambiet temperature and fluid [kPa].
    """

    if fluid == "water":
        Pv = 100 * np.exp(13.766) * np.exp(-5132/T) # [kPa]
    elif fluid == "ethanol":
        Pv = 100 * np.exp(14.567) * np.exp(-5101/T) # [kPa]
    else:
        raise ValueError("Please import the proper fluid")
    return Pv

def calcD(D_298: float, T: float):
    """ 
    Function that calculates the mass diffusion coefficient for the given temperature.

    Inputs:
    D_298 (float): mass diffusion coefficient @ 298K [m^2/sec]
    T (float): desired temperature [K]
    """

    D = D_298 * np.power(T/298, 3/2)
    return D

def C2K(T) -> float:
    """ 
    It returns the tempearature from Celsius to Kelvin
    """
    return T + 273

def K2C(T) -> float:
     """ 
    It returns the tempearature from Kelvin to Celsius.
    """
     return T - 273

def rk4(f, self, y0, t0, t_end, h):
    """
    Solves an ordinary differential equation using the 4th order Runge-Kutta method (RK4).
    
    Parameters:
    f (function): The function that defines the ODE dy/dt = f(t, y).
    y0 (float): Initial condition for y at t0.
    t0 (float): Initial time.
    t_end (float): End time.
    h (float): Step size.
    
    Returns:
    t_values (list): List of time values.
    y_values (list): List of y values corresponding to each time value.
    """
    t_values = [t0]
    y_values = [y0]
    
    t = t0
    y = y0
    
    while t < t_end:
        k1 = h * f(t, y, self)
        k2 = h * f(t + h/2, y + k1/2, self)
        k3 = h * f(t + h/2, y + k2/2, self)
        k4 = h * f(t + h, y + k3, self)
        
        y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        t = t + h
        print('process = %1.6f %% '%(t/t_end*100), end= '\r', flush= True)
        t_values.append(t)
        y_values.append(y)
    
    #print('process = %1.6f kg/s '%(t/t_end*100), end= '\r', flush= True)
    return t_values, y_values

def process(T: float, t: float, args)-> float:
    """ 
    The function that contains the differential equation of the process
    
    Inputs:
    t (float): [sec]
    T (float): Temperature [K]
    self (class type object): the class output that contains the parameters

    Returns:
    dTdt (float): first time derivative of temperature [K/s]
    """
    if T != 0:
        T0, fluid, Ri, rhoa, Mo, Mv, Nu, cp_, rhow_, ao, d, U, Tinf_, hfg_, Le = args#[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14]
        Treal = T*T0
        Pvs = vaporP(Treal, fluid)
        rhos_ = Pvs /Ri / Treal / rhoa
        Pvs_ = Mo / Mv * T * rhos_
        dTdt = Nu * (1/cp_/rhow_) * (6 * ao/ d / U) * ((Tinf_ - T) - hfg_ / np.power(Le, 2/3)*(Mv/ Mo) * Pvs_/T)
    else:
        dTdt = 0
    return dTdt
    

def modify_arrays(t, y, Tscc, added_t = 0)-> np.array:
    yindeces    = np.where(y> Tscc)[0]
    yindex      = yindeces[-1]
    tfinal      = t[yindex]
    index_extra = np.where(t == tfinal + added_t)[0]
    tt          = t[1:int(index_extra)]
    yy          = y[1:int(index_extra)]
    return tt, yy, tfinal


class calculate_cooling:
    def __init__(self, U: float, fluid: str, Tamb: float) -> None:
        self.U      = U
        self.fluid  = fluid
        self.Tamb   = C2K(Tamb)
        # a -> stands for the air
        # w -> stands for the water
        # e -> stands for the ethanol
        self.d      = 0.1       # [m]
        self.cpw    = 4.18      # [kJ/kgK]
        self.rhow   = 1000      # [kg/m^3]
        self.Pamb   = 101.325   # [kPa]
        self.T0     = C2K(25)   # reference temperature [K]

        # air properties @ 25 deg Celcius
        self.cpa    = 1         # [kJ/kgK]
        self.rhoa   = 1.187     # [kJ/kgK]
        self.visca  = 1.562e-5  # [m^2/sec]
        self.Pr     = 0.72      # [-]

        # water properties 
        self.hfgw   = 2400      # [kJ/kg]
        self.Dw     = 2.505e-5  # @ 25 deg Celsius [m^2/sec]

        # ethanol properties
        self.hfge   = 900       # [kJ/kg]
        self.De     = 1.35e-5   # @ 25 deg Celsius [m^2/sec]

        if self.fluid == 'water':
            self.Mv = 18
            self.DAB = self.Dw
        elif self.fluid == 'ethanol':
            self.Mv = 46
            self.DAB = self.De
        self.Mo = 29

    def getparams(self) -> None:
        self.Re = self.U * self.d/ self.visca
        self.ao = self.visca /self.Pr # Thermal diffusivity
        if self.fluid == 'water':
            self.Le = self.ao / self.Dw
            self.hfg_   = self.hfgw / self.cpa / self.T0
            self.hfg    = self.hfgw
        elif self.fluid == 'ethanol':
            self.Le = self.ao/self.De
            self.hfg_   = self.hfge / self.cpa / self.T0
            self.hfg    = self.hfge
        
        self.Pinf_  = 0
        self.Tinf_  = self.Tamb / self.T0
        self.cp_    = self.cpw / self.cpa
        self.rhow_  = self.rhow/ self.rhoa
        self.Nu     = 2 + 0.552 * np.power(self.Re, 1/2)* np.power(self.Pr, 1/3)
        self.Ri     = 8.3143/self.Mv

    def Chilton_Colburn(self) -> None:
        F_CC = lambda Ts: self.Tamb - self.hfg/ self.cpa / np.power(self.Le, 2/3) * self.Mv / self.Mo * vaporP(Ts, self.fluid) / self.Pamb - Ts
        Tsguess = C2K(20)
        sol = fsolve(F_CC, Tsguess)
        return sol


results = {'case': [],
           'T_final C.C.': [],
           't_final': [],
           'SS_transient T': [],
           'SS_transietn t': []}
res = pd.DataFrame(results)

cases = {'name': ['water 40K 1m/s', 'water 40K 10m/s', 'water 30K 1m/s', 'water 30K 10m/s', 'ethanol 40K 1m/s','ethanol 40K 10m/s', 'ethanol 30K 1m/s', 'ethanol 30K 10m/s'],
         'fluid': ['water','water','water','water','ethanol','ethanol','ethanol','ethanol'],
         'Tamb': [40,40,30,30,40,40,30,30],
         'Velocity': [1,10,1,10,1,10,1,10]}

casestorun = pd.DataFrame(cases)

added_tlist = [1000, 1000, 1000, 1000, 500, 500, 500, 500]
for i in range(casestorun.shape[0]):
    casee   = casestorun.iloc[i,0]
    fluid   = casestorun.iloc[i,1]
    Tamb    = casestorun.iloc[i,2]
    Vel     = casestorun.iloc[i,3]
    # name = f"fluid: {fluid}, T = {Tamb}$^\circ$C, V = {Vel}m/s"

    par = calculate_cooling(Vel, fluid= fluid, Tamb= Tamb)
    par.getparams()
    Tscc = par.Chilton_Colburn()
    args = [par.T0, par.fluid, par.Ri, par.rhoa, par.Mo, par.Mv, par.Nu, par.cp_, par.rhow_, par.ao, par.d, par.U, par.Tinf_, par.hfg_, par.Le]

    t = np.arange(0, 5000000, 1)
    T = odeint(process, par.Tinf_, t = t,  args = (args,))

    tt_trans = t * par.d/ par.U
    TT_trans = K2C(T * par.T0)

    tt, TT, tfinal = modify_arrays(tt_trans, TT_trans, Tscc = 1.001*K2C(Tscc), added_t= added_tlist[i]*0)


    dftoadd = {'case': casee,
                'T_final C.C.': float(K2C(Tscc)),
                't_final': tfinal,
                'SS_transient T': float(TT_trans[-1]),
                'SS_transietn t': float(tt_trans[-1])}
    res.loc[len(res)] = dftoadd

    # print(yy[-1])
    # print(tfinal)
    indexx = int(1.8*len(tt))
    plt.plot(tt_trans[0:indexx], TT_trans[0:indexx])
    plt.plot([0, tt_trans[indexx]], [TT_trans[indexx],TT_trans[-1]], linestyle='--', color='0.5')
    # plt.plot([tfinal, tfinal], [0, float(K2C(Tscc))], linestyle = '--', color= '0.5')
    # plt.scatter(tfinal, K2C(Tscc), color = 'red', s = 50, marker = 'o', label = ' Chilton-Colburn Temperature')
    plt.title(f"fluid: {fluid}, T = {Tamb}$^\\circ$C, V = {Vel}m/s")
    plt.xlabel("t [sec]")
    plt.ylabel(r'$T_s$ [$^\circ$C]')
    plt.grid()
    figname = f"{fluid}_{Tamb}_{Vel}.png"
    plt.savefig(figname)
    plt.clf()
    
print(res)

