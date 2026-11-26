# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 09:57:11 2025

@author: VK
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ------------------------
# Fluid properties (air @ 20 C)
# ------------------------
nu = 1.5e-5       # m^2/s
Pr = 0.71
k = 0.026         # W/mK
beta = 1/293.0    # 1/K
g = 9.81          # m/s^2

# ------------------------
# Duct and flow parameters
# ------------------------
U_inf = 1.0       # m/s (forced horizontal)
L = 10.0          # duct length [m]
B = 0.10          # plate spacing [m]
T_inf = 20.0      # °C
T_w_CWT = 70.0    # °C, constant wall temperature
q_flux = 100.0    # W/m^2, CHF
Re_transition = 5e5
dRe_transition = 1e5
Ra_crit = 1e9

x = np.linspace(1e-6, L, 1200)  # avoid zero

# ------------------------
# Intermittency function
# ------------------------
def gamma_Re(Re):
    return 1.0 / (1.0 + np.exp(-(Re - Re_transition)/dRe_transition))

# ------------------------
# Friction coefficients
# ------------------------
def cf_lam(Re_x):
    return 0.664/np.sqrt(np.maximum(Re_x,1e-12))
def cf_turb(Re_x):
    return 0.0592*np.maximum(Re_x,1e-12)**(-1.0/5.0)

H_lam, H_turb = 2.59, 1.3
Ht_lam, Ht_turb = 2.5, 1.3

# ------------------------
# Coupled von Karman momentum-energy solver
# ------------------------
def rhs_x(xloc, y):
    theta = max(y[0],1e-18)
    theta_t = max(y[1],1e-18)
    Re_x_loc = U_inf*xloc/nu
    g_int = gamma_Re(Re_x_loc)
    cf_local = (1-g_int)*cf_lam(Re_x_loc) + g_int*cf_turb(Re_x_loc)
    dtheta_dx = 0.5*cf_local
    lam_factor = Pr**(-1/3)
    eff_pref = (1-g_int)*lam_factor + g_int*1.0
    ratio = theta_t/(theta+1e-12)
    F = 1.0/(1+ratio)
    dtheta_t_dx = 0.5*cf_local*eff_pref*F
    return [dtheta_dx, dtheta_t_dx]

y0 = [1e-12,1e-12]
sol = solve_ivp(rhs_x,(x[0],x[-1]),y0,t_eval=x,rtol=1e-6,atol=1e-9)
theta = sol.y[0]
theta_t = sol.y[1]

Re_x = U_inf*x/nu
gamma_vals = gamma_Re(Re_x)
H_eff = (1-gamma_vals)*H_lam + gamma_vals*H_turb
Ht_eff = (1-gamma_vals)*Ht_lam + gamma_vals*Ht_turb
delta_h_vk = H_eff*theta
delta_t_h_vk = Ht_eff*theta_t
delta_t_CWT_h = delta_t_h_vk.copy()
delta_t_CHF_h = delta_t_h_vk.copy()

Nu_from_delta_t = np.maximum(x/np.maximum(delta_t_h_vk,1e-12),1e-12)
h_local_chf = k*Nu_from_delta_t/x
Tw_CHF_h = T_inf + q_flux/h_local_chf

# Horizontal average temperature (CHF approximation)
Tavg_h = T_inf + 0.5 * np.minimum(delta_t_CHF_h / B, 1.0) * (Tw_CHF_h - T_inf)

# ------------------------
# Vertical duct (natural convection)
# ------------------------
DeltaT_v = T_w_CWT - T_inf
Gr_x = g*beta*DeltaT_v*x**3/nu**2
Ra_x = Gr_x*Pr

delta_v = np.zeros_like(x)
delta_t_CWT_v = np.zeros_like(x)
delta_t_CHF_v = np.zeros_like(x)
Tw_CHF_v = np.zeros_like(x)
Uavg_v = np.zeros_like(x)
Tavg_v = np.zeros_like(x)

for i,Rax in enumerate(Ra_x):
    if Rax<=0:
        delta_v[i] = 1e-12
        Nu_CWT_v = 1e-12
    elif Rax<Ra_crit:
        delta_v[i] = x[i]*Rax**(-1/4)
        Nu_CWT_v = 0.59*Rax**0.25
    else:
        delta_v[i] = x[i]*Rax**(-0.2)
        Nu_CWT_v = 0.10*Rax**(1/3)
    delta_t_CWT_v[i] = delta_v[i]*Pr**(-1/3)
    Nu_CHF_v = Nu_CWT_v
    delta_t_CHF_v[i] = delta_v[i]*Pr**(-1/3)
    h_x_v = k*Nu_CHF_v/x[i]
    Tw_CHF_v[i] = T_inf + q_flux/h_x_v
    # realistic vertical velocities (~0.01–1 m/s)
    Uavg_v[i] = 0.1*np.sqrt(np.maximum(g*beta*DeltaT_v*x[i],0.0))
    # average temperature fraction
    frac_v = min(delta_t_CHF_v[i]/B,1.0)
    Tavg_v[i] = T_inf + 0.5*frac_v*(Tw_CHF_v[i]-T_inf)

# transitions & merging
x_transition_h = Re_transition*nu/U_inf
x_transition_v = ((Ra_crit/(g*beta*DeltaT_v))*(nu**2)/Pr)**(1.0/3.0) if DeltaT_v!=0 else np.inf
merge_idx_h = np.argmax(delta_h_vk>=B/2)
merge_x_h = x[merge_idx_h] if delta_h_vk[merge_idx_h]>=B/2 else None
merge_idx_v = np.argmax(delta_v>=B/2)
merge_x_v = x[merge_idx_v] if delta_v[merge_idx_v]>=B/2 else None
x_show = L-0.5
i_show = int(np.argmin(np.abs(x-x_show)))
y = np.linspace(0,B,400)
u_profile_h = 1.5*U_inf*(1-(2*y/B-1)**2)
u_profile_v = 1.5*Uavg_v[i_show]*(1-(y/B)**2)

# thermal profiles at x_show
delta_t_h_local = delta_t_h_vk[i_show]
Tw_h_local = Tw_CHF_h[i_show]
T_profile_CWT_h = T_inf + (T_w_CWT-T_inf)*np.exp(-y/np.maximum(delta_t_h_local,1e-6))
T_profile_CHF_h = T_inf + (Tw_h_local-T_inf)*np.exp(-y/np.maximum(delta_t_h_local,1e-6))
delta_t_v_local = delta_t_CHF_v[i_show]
Tw_v_local = Tw_CHF_v[i_show]
T_profile_CWT_v = T_inf + (T_w_CWT-T_inf)*np.exp(-y/np.maximum(delta_t_v_local,1e-6))
T_profile_CHF_v = T_inf + (Tw_v_local-T_inf)*np.exp(-y/np.maximum(delta_t_v_local,1e-6))

# ------------------------
# Plotting
# ------------------------
plt.figure(figsize=(10,6))
plt.plot(x,delta_h_vk*1000,label='Horizontal δ [mm]',color='tab:blue')
plt.plot(x,delta_t_h_vk*1000,'--',label='Horizontal δt [mm]',color='tab:cyan')
plt.plot(x,delta_v*1000,label='Vertical δ [mm]',color='tab:orange')
plt.plot(x,delta_t_CWT_v*1000,'--',label='Vertical δt [mm]',color='tab:red')
plt.axvline(x_transition_h,color='k',linestyle=':',label=f'Forced trans x={x_transition_h:.2f} m')
plt.axvline(x_transition_v,color='k',linestyle='-.',label=f'Natural trans x≈{x_transition_v:.2f} m')
plt.xlabel('x [m]'); plt.ylabel('BL thickness [mm]')
plt.title('Boundary-layer growth (CWT)'); plt.legend(); plt.grid(True)

plt.figure(figsize=(10,6))
plt.plot(x,delta_h_vk*1000,label='Horizontal δ [mm]',color='tab:blue')
plt.plot(x,delta_t_CHF_h*1000,'--',label='Horizontal δt (CHF) [mm]',color='tab:cyan')
plt.plot(x,delta_v*1000,label='Vertical δ [mm]',color='tab:orange')
plt.plot(x,delta_t_CHF_v*1000,'--',label='Vertical δt (CHF) [mm]',color='tab:red')
plt.axvline(x_transition_h,color='k',linestyle=':',label=f'Forced trans x={x_transition_h:.2f} m')
plt.axvline(x_transition_v,color='k',linestyle='-.',label=f'Natural trans x≈{x_transition_v:.2f} m')
plt.xlabel('x [m]'); plt.ylabel('BL thickness [mm]')
plt.title('Boundary-layer growth (CHF)'); plt.legend(); plt.grid(True)

plt.figure(figsize=(10,6))
plt.plot(x,Tw_CHF_h,label='Horizontal Tw(x) CHF',color='tab:cyan')
plt.plot(x,Tw_CHF_v,label='Vertical Tw(x) CHF',color='tab:red')
plt.axhline(T_w_CWT,color='green',linestyle='--',label=f'CWT Tw={T_w_CWT} °C')
plt.xlabel('x [m]'); plt.ylabel('Wall temperature [°C]')
plt.title('Wall temperature (CHF)'); plt.legend(); plt.grid(True)

plt.figure(figsize=(8,6))
plt.plot(y*1000,u_profile_h,label=f'Horizontal velocity at x={x_show:.2f} m')
plt.plot(y*1000,u_profile_v,label=f'Vertical velocity at x={x_show:.2f} m')
plt.xlabel('Distance from heated wall [mm]'); plt.ylabel('Velocity [m/s]')
plt.title(f'Velocity profiles at x={x_show:.2f} m'); plt.legend(); plt.grid(True)

plt.figure(figsize=(8,6))
plt.plot(y*1000,T_profile_CWT_h,label='Horizontal CWT')
plt.plot(y*1000,T_profile_CHF_h,label='Horizontal CHF')
plt.plot(y*1000,T_profile_CWT_v,label='Vertical CWT',linestyle=':')
plt.plot(y*1000,T_profile_CHF_v,label='Vertical CHF',linestyle='--')
plt.xlabel('Distance from heated wall [mm]'); plt.ylabel('Temperature [°C]')
plt.title(f'Thermal profiles at x={x_show:.2f} m'); plt.legend(); plt.grid(True)

plt.figure(figsize=(10,6))
plt.plot(x,Uavg_v,label='Vertical avg velocity (realistic)')
plt.plot(x,np.full_like(x,U_inf),label='Horizontal avg velocity')
plt.xlabel('x [m]'); plt.ylabel('Average velocity [m/s]'); plt.title('Average velocity along duct'); plt.legend(); plt.grid(True)

plt.figure(figsize=(10,6))
plt.plot(x,Tavg_v,label='Vertical avg temperature')
plt.plot(x,Tavg_h,label='Horizontal avg temperature')
plt.xlabel('x [m]'); plt.ylabel('Average temperature [°C]'); plt.title('Average temperature along duct'); plt.legend(); plt.grid(True)

plt.show()

print("Summary:")
print(f"Horizontal duct transition x ≈ {x_transition_h:.2f} m")
print(f"Vertical duct transition x ≈ {x_transition_v:.2f} m")
if merge_x_h: print(f"Horizontal BL merge x ≈ {merge_x_h:.2f} m")
if merge_x_v: print(f"Vertical BL merge x ≈ {merge_x_v:.2f} m")
