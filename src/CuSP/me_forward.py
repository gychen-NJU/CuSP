import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import glob
import time

def VoigtFaradayProfile(
    u: torch.Tensor,
    a: torch.Tensor,
    ynodes=100,
    lim=5.0,
) -> torch.Tensor:
    device = u.device
    dtype = u.dtype
    y = torch.linspace(-lim,lim,ynodes,device=device,dtype=dtype)
    dy = y[1]-y[0]
    u = u.unsqueeze(2)
    a = a.unsqueeze(2)
    y = y[None,None,:]
    numerator = torch.exp(-y**2)*(u-y)
    denominator = (u-y)**2+a**2
    integrand = numerator/denominator
    profile = (1/torch.pi**1.5)*torch.trapz(integrand,dx=dy,dim=-1)
    return profile

def VoigtProfile(
    u: torch.Tensor,
    a: torch.Tensor,
    ynodes=1000,
    lim=10.0,
) -> torch.Tensor:
    device = u.device
    dtype = u.dtype
    y = torch.linspace(-lim,lim,ynodes,device=device,dtype=dtype)
    dy = y[1]-y[0]
    u = u.unsqueeze(2)
    a = a.unsqueeze(2)
    y = y[None,None,:]
    try:
        numerator = torch.exp(-y**2)
        denominator = (u-y)**2+a**2
        integrand = numerator/denominator
        profile = (a[:,:,0]/torch.pi**1.5)*torch.trapz(integrand,dx=dy,dim=-1)
    except:
        print(u.shape,a.shape,y.shape)
        raise
    return profile

class MEForward:
    kB      = 1.380649e-23    # J/K -> Boltzmann constant
    u       = 1.661e-27       # kg  -> atomic mass unit
    c       = 299792458.      # m/s -> speed of light
    e       = 1.602176634e-19 # C   -> elementary electric charge
    Ar      = 55.85           #     -> relative actomic mass of Fe

    def __init__(
        self,
        wavebands: torch.Tensor,
        landeG:float=2.5,
        lambda0:float=630.25,
        wing = None
    ):
        '''
        Initialize the MEForward class
        Input:
            waveband  [nm]  -> waveband of interest
            landeG    []    -> Lande factor
            lambda0   [nm]  -> central wavelength
            wing      [nm]  -> continue spectrum's wing
        '''
        self.wavebands = wavebands
        self.G = landeG
        self.lambda0 = lambda0
        self.wing = wing if wing is not None else wavebands.min().item()
        self.wavebands = torch.cat([torch.tensor([self.wing]).to(wavebands.device).to(wavebands.dtype),self.wavebands])

    def __call__(
        self,
        Dlambda_D:torch.Tensor, # Doppler width include micro turblence and thermal motion
        v_los:torch.Tensor, # line of sight velocity
        eta_0:torch.Tensor, # fraction between selective absorption and continuum absorption
        S10:torch.Tensor, # Source function ratio of S1/S0
        a_damp:torch.Tensor, # damping factor in Lorentz profile
        Bmag:torch.Tensor,# Magnetic field strength
        theta:torch.Tensor,# Magnetic field inclination angle
        phi:torch.Tensor,# Magnetic field azimuth angle
    ):
        '''
        Compute the forward model for the ME spectrum
        Input:
            Dlambda_D [mA] -> Doppler width include micro turblence and thermal motion
            v_los     [m/s] -> velocity of medium
            eta_0     []    -> ratio between the selected absorption and continous absorption
            S10       []    -> S1/S0, the ME atomosphere source function: S = S0 + S1*tau
            a_damping []    -> damping factor for absorption profile
            Bmag      [G]   -> photospherical magnetic field magnitude
            theta     [deg] -> angle respective to the z direction
            phi       [deg] -> angle respective to the +Q direction
        All the input tensors should be on the same device and dtype with shape (N,1)
        Output:
            I,Q,U,V -> Stokes parameters in certian waveband
        '''
        I0,Q0,U0,V0 = self.return_IQUV(Dlambda_D,v_los,eta_0,S10,a_damp,Bmag,theta,phi)
        return I0,Q0,U0,V0

    def return_profile(self,Dlambda_D,u_los,u_B,a_damp):
        '''
        Return the absorption and dispersion profiles
        '''
        wavebands = self.wavebands.unsqueeze(0)
        u0 = (wavebands-self.lambda0)/Dlambda_D
        G  = self.G
        phi_0 = VoigtProfile(u0-u_los      , a_damp)
        phi_B = VoigtProfile(u0-u_los+G*u_B, a_damp)
        phi_R = VoigtProfile(u0-u_los-G*u_B, a_damp)
        psi_0 = VoigtFaradayProfile(u0-u_los      , a_damp)
        psi_B = VoigtFaradayProfile(u0-u_los+G*u_B, a_damp)
        psi_R = VoigtFaradayProfile(u0-u_los-G*u_B, a_damp)
        return phi_0,phi_B,phi_R,psi_0,psi_B,psi_R

    def return_eta_rho(self,eta_0,theta,phi,phi_0,phi_B,phi_R,psi_0,psi_B,psi_R):
        '''
        Return:
        eta_{I,Q,U,V} and rho_{Q,U,V} of the propogation matrix: eta_I, eta_Q, eta_U, eta_V, rho_Q, rho_U, rho_V
        '''
        eta_I = 1+eta_0/2*(phi_0*torch.sin(theta)**2+0.5*(phi_B+phi_R)*(1+torch.cos(theta)**2))
        eta_Q = eta_0/2*(phi_0-0.5*(phi_B+phi_R))*torch.sin(theta)**2*torch.cos(2*phi)
        eta_U = eta_0/2*(phi_0-0.5*(phi_B+phi_R))*torch.sin(theta)**2*torch.sin(2*phi)
        eta_V = eta_0/2*(phi_R-phi_B)*torch.cos(theta)
        rho_Q = eta_0/2*(psi_0-0.5*(psi_B+psi_R))*torch.sin(theta)**2*torch.cos(2*phi)
        rho_U = eta_0/2*(psi_0-0.5*(psi_B+psi_R))*torch.sin(theta)**2*torch.sin(2*phi)
        rho_V = eta_0/2*(psi_R-psi_B)*torch.cos(theta)
        return eta_I, eta_Q, eta_U, eta_V, rho_Q, rho_U, rho_V

    def return_IQUV(self,Dlambda_D,v_los,eta_0,S10,a_damp,Bmag,theta,phi):
        '''
        Return:
        I0/Ic, Q0/Qc, U0/Uc, V0/Vc
        '''
        # Dlambda_D = Dlambda_D*1e-4
        # Dlambda_D = self.lambda0*v_D/self.c
        # print(Dlambda_D)
        Dlambda_los = self.lambda0*v_los/self.c
        u_los = Dlambda_los/Dlambda_D
        lambdaB = 4.67e-12*self.lambda0**2*Bmag
        u_B = lambdaB/Dlambda_D
        phi_0,phi_B,phi_R,psi_0,psi_B,psi_R = self.return_profile(Dlambda_D,u_los,u_B,a_damp)
        eta_I, eta_Q, eta_U, eta_V, rho_Q, rho_U, rho_V = self.return_eta_rho(eta_0,theta,phi,phi_0,phi_B,phi_R,psi_0,psi_B,psi_R)
        Delta   = eta_I**2*(eta_I**2-eta_Q**2-eta_U**2-eta_V**2+rho_Q**2+rho_U**2+rho_V**2)-(eta_Q*rho_Q+eta_U*rho_U+eta_V*rho_V)**2
        Delta_r = 1/Delta
        
        I_tau0  = 1+Delta_r*eta_I*(eta_I**2+rho_Q**2+rho_U**2+rho_V**2)*S10
        Q_tau0  = Delta_r*(eta_I**2*eta_Q+eta_I*(eta_V*rho_U-eta_U*rho_V)+rho_Q*(eta_Q*rho_Q+eta_U*rho_U+eta_V*rho_V))*S10
        U_tau0  = Delta_r*(eta_I**2*eta_U+eta_I*(eta_Q*rho_V-eta_V*rho_Q)+rho_U*(eta_Q*rho_Q+eta_U*rho_U+eta_V*rho_V))*S10
        V_tau0  = Delta_r*(eta_I**2*eta_V+eta_I*(eta_U*rho_Q-eta_Q*rho_U)+rho_V*(eta_Q*rho_Q+eta_U*rho_U+eta_V*rho_V))*S10
        
        Ic      = I_tau0[:,0:1]
        I0      = I_tau0[:,1:]/Ic
        Q0      = Q_tau0[:,1:]/Ic
        U0      = U_tau0[:,1:]/Ic
        V0      = V_tau0[:,1:]/Ic  
        return I0,Q0,U0,V0
