# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 16:01:44 2025

@author: Antonio Oliva Aparicio

Refinement (second) stage. 

Run this after identifying the best candidates in the first design stage.
"""

from scipy.optimize import minimize
import numpy as np
import pandas as pd

# Load the CSV file (adjust the path as needed)
file_path = r"iris_parametric_data_triplets.csv"
df = pd.read_csv(file_path)

df.rename({df.columns[1]: 'o_v'}, axis=1, inplace=True)
df.rename({df.columns[2]: 'w_v'}, axis=1, inplace=True)
df.rename({df.columns[3]: 'S31_v'}, axis=1, inplace=True)
df.rename({df.columns[4]: 'S21_v'}, axis=1, inplace=True)
df.rename({df.columns[5]: 'S32_v'}, axis=1, inplace=True)
df.rename({df.columns[6]: 'phaseTE10_v'}, axis=1, inplace=True)
df.rename({df.columns[7]: 'phaseTE20_v'}, axis=1, inplace=True)

df.rename({df.columns[8]: 'partialMagS31w_v'}, axis=1, inplace=True)
df.rename({df.columns[9]: 'partialMagS21w_v'}, axis=1, inplace=True)
df.rename({df.columns[10]:'partialMagS32w_v'}, axis=1, inplace=True)
df.rename({df.columns[11]:'partialMagS31o_v'}, axis=1, inplace=True)
df.rename({df.columns[12]:'partialMagS21o_v'}, axis=1, inplace=True)
df.rename({df.columns[13]:'partialMagS32o_v'}, axis=1, inplace=True)

df.rename({df.columns[14]:'phaseTE101_v'}, axis=1, inplace=True)
df.rename({df.columns[15]:'partialPhaseTE101w_v'}, axis=1, inplace=True)

# Assign variables dynamically
for col in df.columns:
    globals()[col] = df[col].tolist()  # Convert column to a list and assign it to a variable

def node_add(M, C, ii, jj, beta):
    M[jj, :] = beta * M[ii, :] + M[jj, :]
    C[jj, :] = beta * C[ii, :] + C[jj, :]
    M[:, jj] = beta * M[:, ii] + M[:, jj]
    C[:, jj] = beta * C[:, ii] + C[:, jj]
    return M, C

def node_scale(M, C, ii, alpha):
    M[ii, :] *= alpha
    M[:, ii] *= alpha
    C[ii, :] *= alpha
    C[:, ii] *= alpha
    return M, C

def cost_function(param):
    
    # Variables to optimize
    w, o1, o2   = param  
    

    # Filter constants
    f0  =  10e9       # center frequency
    FBW =  0.02       # fractional bandwidth
    c   =  3e8        # speed of light
    a   =  22.86e-3   # standard waveguide width
    a2  =  40e-3      # cavity width

    # Couplings Triplets (values obtained from filter synthesis)
    fc   = c/(2*a)
    Xeq  = (np.pi/2)/(1-(fc/f0)**2)
    MS1 =  0.7374*np.sqrt(Xeq*FBW)
    M11 =  0.6655
    M1L = -0.5109*np.sqrt(Xeq*FBW)
    MSL =  0.4803*Xeq*FBW
    

    # Input iris 
    ind_input   = 69 # Adjust according to selected best candidates
    w_nom       = w_v[ind_input]
    o_nom       = o_v[ind_input]
    magS32      = S32_v[ind_input]
    magderS32w  = partialMagS32w_v[ind_input]
    magderS32o  = partialMagS32o_v[ind_input]
    magS31      = S31_v[ind_input]
    magderS31w  = partialMagS31w_v[ind_input]
    magderS31o  = partialMagS31o_v[ind_input]
    magS21      = S21_v[ind_input]
    magderS21w  = partialMagS21w_v[ind_input]
    magderS21o  = partialMagS21o_v[ind_input]
    phaseSTE10_sim = phaseTE10_v[ind_input]
    phaseSTE20_sim = phaseTE20_v[ind_input]


    # Update S-parameters of the Input Iris
    S32 = magS32 + (w-w_nom)*magderS32w + (o1-o_nom)*magderS32o
    S31 = magS31 + (w-w_nom)*magderS31w + (o1-o_nom)*magderS31o
    S21 = magS21 + (w-w_nom)*magderS21w + (o1-o_nom)*magderS21o

    # Equivalent circuit of the Output Iris
    KS1 = abs(S32)/abs(S21)
    KS2 = abs(S32)/abs(S31)
    Xs  =-np.sqrt(abs(4*KS1**2*KS2**2/abs(S32)**2-(KS1**2+KS2**2+1)**2))

    # Output iris 
    ind_output   = 45 # Adjust according to selected best candidates
    w_nom       = w_v[ind_output]
    o_nom       = o_v[ind_output]
    magS32      = S32_v[ind_output]
    magderS32w  = partialMagS32w_v[ind_output]
    magderS32o  = partialMagS32o_v[ind_output]
    magS31      = S31_v[ind_output]
    magderS31w  = partialMagS31w_v[ind_output]
    magderS31o  = partialMagS31o_v[ind_output]
    magS21      = S21_v[ind_output]
    magderS21w  = partialMagS21w_v[ind_output]
    magderS21o  = partialMagS21o_v[ind_output]
    phaseLTE10_sim = phaseTE10_v[ind_output]
    phaseLTE20_sim = phaseTE20_v[ind_output]

    # Update S-parameters of the Output Iris
    S32 = magS32 + (w-w_nom)*magderS32w + (o2-o_nom)*magderS32o
    S31 = magS31 + (w-w_nom)*magderS31w + (o2-o_nom)*magderS31o
    S21 = magS21 + (w-w_nom)*magderS21w + (o2-o_nom)*magderS21o

    # Equivalent circuit of the Output Iris
    K1L = abs(S32)/abs(S21);
    K2L = abs(S32)/abs(S31);
    Xl  =-np.sqrt(abs(4*K1L**2*K2L**2/abs(S32)**2-(K1L**2+K2L**2+1)**2));

    # Peer singlet calculation
    epsilon = (1 + MSL**2) / (2 * abs(MSL))
    k = epsilon**2 - 1
    MSS = Xs  
    MLL = Xl  
    xi = 1 + MSS * MLL
    MSL_ = np.sign(MSL) * np.sqrt(xi + 2 * k - np.sqrt(4 * k**2 + 4 * k * xi - (MSS - MLL)**2))

    global phi1, phi2
    phi1 = -np.angle(-(-MSL_**2 + 1j * MLL - 1j * MSS + MLL * MSS + 1) / (MSL_**2 + 1j * MLL + 1j * MSS - MLL * MSS + 1)) / 2
    b1 = -1 / np.tan(phi1)
    J1 = 1 / np.sin(phi1)

    phi2 = -np.angle(-(-MSL_**2 - 1j * MLL + 1j * MSS + MLL * MSS + 1) / (MSL_**2 + 1j * MLL + 1j * MSS - MLL * MSS + 1)) / 2
    b2 = -1 / np.tan(phi2)
    J2 = 1 / np.sin(phi2)

    M = np.array([
        [0, 1, 0, 0, 0, 0, 0],
        [1, b1, J1, 0, 0, 0, 0],
        [0, J1, b1, MS1, MSL, 0, 0],
        [0, 0, MS1, M11, M1L, 0, 0],
        [0, 0, MSL, M1L, b2, J2, 0],
        [0, 0, 0, 0, J2, b2, 1],
        [0, 0, 0, 0, 0, 1, 0]
    ])

    U = np.diag([0, 0, 0, 1.0, 0, 0, 0])
    Mr, Ur = M.copy(), U.copy()

    r = Mr[1, 2] / Mr[2, 2]
    Mr, Ur = node_add(Mr, Ur, 2, 1, -r - np.sqrt(r**2 - 1))
    Mr, Ur = node_add(Mr, Ur, 2, 3, -Mr[1, 3] / Mr[1, 2])
    Mr, Ur = node_add(Mr, Ur, 2, 4, -Mr[1, 4] / Mr[1, 2])

    r = Mr[5, 4] / Mr[4, 4]
    s = Mr[5, 5] / Mr[4, 4]
    Mr, Ur = node_add(Mr, Ur, 4, 5, -r - np.sqrt(r**2 - s))
    Mr, Ur = node_add(Mr, Ur, 4, 3, -Mr[5, 3] / Mr[5, 4])
    Mr, Ur = node_add(Mr, Ur, 4, 2, -Mr[5, 2] / Mr[5, 4])
    Mr, Ur = node_scale(Mr, Ur, 4, 1 / Mr[4, 5])

    M11_          = Mr[3, 3]
    fr            = 0.5 * f0 * FBW * (-M11_ + np.sqrt(M11_**2 + 4 / FBW**2))
    glambda_fr_20 = 1000 * (c / fr) / np.sqrt(1 - (c / (a2 * fr))**2)

    phaseSTE10 = np.angle((KS1**2 - KS2**2 - 1j*Xs - 1) / (KS1**2 + KS2**2 + 1j*Xs + 1))
    phaseLTE10 = np.angle((K1L**2 - K2L**2 - 1j*Xl - 1) / (K1L**2 + K2L**2 + 1j*Xl + 1))
    phaseSTE20 = np.angle((KS2**2 - KS1**2 - 1j*Xs - 1) / (KS2**2 + KS1**2 + 1j*Xs + 1))
    phaseLTE20 = np.angle((K2L**2 - K1L**2 - 1j*Xl - 1) / (K2L**2 + K1L**2 + 1j*Xl + 1))

    glambda_f0_20 = 1000 * (c / f0) / np.sqrt(1 - (c / (a2 * f0))**2)  # mm
    glambda_f0_10 = 1000 * (c / f0) / np.sqrt(1 - (c / (2 * a2 * f0))**2)  # mm

    loadSTE10 = -phaseSTE10_sim + phaseSTE10
    loadLTE10 = -phaseLTE10_sim + phaseLTE10
    loadSTE20 = -phaseSTE20_sim + phaseSTE20
    loadLTE20 = -phaseLTE20_sim + phaseLTE20

    loadTE10 = (loadSTE10 + loadLTE10) / 2
    loadTE20 = (loadSTE20 + loadLTE20) / 2

    global Lcav
    Lcav = (np.pi - loadTE20) * glambda_fr_20 / (2 * np.pi)
    theta = (2 * np.pi / glambda_f0_10) * Lcav + loadTE10

    iter = 5
    for _ in range(iter):
        MSS = Xs - KS1**2 / np.tan(theta)
        MLL = Xl - K1L**2 / np.tan(theta)
        xi = 1 + MSS * MLL
        MSL_ = np.sign(MSL) * np.sqrt(xi + 2 * k - np.sqrt(4 * k**2 + 4 * k * xi - (MSS - MLL)**2))
        
        phi1 = -np.angle(-(-MSL_**2 + 1j * MLL - 1j * MSS + MLL * MSS + 1) / (MSL_**2 + 1j * MLL + 1j * MSS - MLL * MSS + 1)) / 2
        b1 = -1 / np.tan(phi1)
        J1 = 1 / np.sin(phi1)

        phi2 = -np.angle(-(-MSL_**2 - 1j * MLL + 1j * MSS + MLL * MSS + 1) / (MSL_**2 + 1j * MLL + 1j * MSS - MLL * MSS + 1)) / 2
        b2 = -1 / np.tan(phi2)
        J2 = 1 / np.sin(phi2)

        M = np.array([
            [0, 1, 0, 0, 0, 0, 0],
            [1, b1, J1, 0, 0, 0, 0],
            [0, J1, b1, MS1, MSL, 0, 0],
            [0, 0, MS1, M11, M1L, 0, 0],
            [0, 0, MSL, M1L, b2, J2, 0],
            [0, 0, 0, 0, J2, b2, 1],
            [0, 0, 0, 0, 0, 1, 0]
        ])

        U = np.diag([0, 0, 0, 1.0, 0, 0, 0])
        Mr, Ur = M.copy(), U.copy()

        r = Mr[1, 2] / Mr[2, 2]
        Mr, Ur = node_add(Mr, Ur, 2, 1, -r - np.sqrt(r**2 - 1))
        Mr, Ur = node_add(Mr, Ur, 2, 3, -Mr[1, 3] / Mr[1, 2])
        Mr, Ur = node_add(Mr, Ur, 2, 4, -Mr[1, 4] / Mr[1, 2])

        r = Mr[5, 4] / Mr[4, 4]
        s = Mr[5, 5] / Mr[4, 4]
        Mr, Ur = node_add(Mr, Ur, 4, 5, -r - np.sqrt(r**2 - s))
        Mr, Ur = node_add(Mr, Ur, 4, 3, -Mr[5, 3] / Mr[5, 4])
        Mr, Ur = node_add(Mr, Ur, 4, 2, -Mr[5, 2] / Mr[5, 4])
        Mr, Ur = node_scale(Mr, Ur, 4, 1 / Mr[4, 5])

        M11_          = Mr[3, 3]
        fr            = 0.5 * f0 * FBW * (-M11_ + np.sqrt(M11_**2 + 4 / FBW**2))
        glambda_fr_20 = 1000 * (c / fr) / np.sqrt(1 - (c / (a2 * fr))**2)
        Lcav = (np.pi - loadTE20) * glambda_fr_20 / (2 * np.pi)
        theta = (2 * np.pi / glambda_f0_10) * Lcav + loadTE10

    KSL = np.sign(MSL) * -KS1 * K1L / np.sin(theta)
    wr = -M11_
    X = np.tan(2 * np.pi * Lcav / glambda_f0_20 + loadTE20)
    KS2_target = Mr[2, 3] * np.sqrt(abs(X / wr))
    KSL_target = Mr[2, 4]
    K2L_target = Mr[3, 4] * np.sqrt(abs(X / wr))

    KS2 *= np.sign(KS2_target)
    K2L *= np.sign(K2L_target)

    costKS2 = abs((KS2 - KS2_target)/KS2)
    costK2L = abs((K2L - K2L_target)/K2L)
    costKSL = abs((KSL - KSL_target)/KSL)
    cost = costKS2 + costK2L + costKSL

    # Calculate loading phases
    phaseSTE101 = np.angle((KS1**2 + KS2**2 + 1j*Xs - 1) / (KS1**2 + KS2**2 + 1j*Xs + 1))
    phaseLTE101 = np.angle((K1L**2 + K2L**2 + 1j*Xl - 1) / (K1L**2 + K2L**2 + 1j*Xl + 1))
    phaseSTE101+= np.pi
    phaseLTE101+= np.pi
    
    global loadTE101_S,loadTE101_L
    loadTE101_S = phaseSTE101 - (phaseTE101_v[ind_input]  + (w-w_nom)*partialPhaseTE101w_v[ind_input])
    loadTE101_L = phaseSTE101 - (phaseTE101_v[ind_output] + (w-w_nom)*partialPhaseTE101w_v[ind_output])
    
    return cost

ind_input   = 69  # Adjust according to selected best candidates
ind_output  = 45  # Adjust according to selected best candidates

# Initial values of the design parameters
init_values = [w_v[ind_input], o_v[ind_input], o_v[ind_output]]

# Constraints for each design parameter
dw     = 0.2  # Maximum deviation during refinement stage for irises width
do     = 0.2  # Maximum deviation during refinement stage for irises offsets
w      = init_values[0]
o1     = init_values[1]
o2     = init_values[2]
bounds = [(w-dw,w+dw), (o1-do,o1+do), (o2-do,o2+do)]

resultado = minimize(cost_function, init_values, method="SLSQP", bounds=bounds)

print(f"Succesful optimization: {resultado.success}")
print(f"Optimal values for w, o1 and o2: {resultado.x}")
print(f"Best value of cost function: {resultado.fun}")

