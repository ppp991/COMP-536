import numpy as np
import os
import matplotlib.pyplot as plt
import Physical_Constants_5
import matplotlib.animation as animation
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from my_numerical_methods_5 import SolveODE, ODEIntegrator, Nbody

###### PROJECT EXTENSION #######
class KerrMetricFull():
    def __init__(self, M, a, Q, E, L):
        self.M = M    # Black hole mass
        self.a = a    # Spin parameter (a = J/M)
        self.Q = Q    # Carter constant
        self.E = E    # Energy per unit mass
        self.L = L    # Angular momentum along spin axis

    def geodesics(self, tau, state):
        t, r, r_dot, theta, theta_dot, phi = state
        M, a, E, L, Q = self.M, self.a, self.E, self.L, self.Q

        # Clamp theta away from poles for safety
        theta = np.clip(theta, 1e-6, np.pi - 1e-6)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin2 = sin_theta**2
        cos2 = cos_theta**2

        # Metric quantities
        Delta = r**2 - 2*M*r + a**2
        Sigma = r**2 + a**2 * cos2

        # Effective potentials (Carter)
        R = ((E * (r**2 + a**2) - a * L)**2 - Delta * (r**2 + (L - a * E)**2 + Q))
        Theta = Q - cos2 * (a**2 * (1 - E**2) + L**2 / sin2)

        # Derivatives of coordinates
        dt_dtau = ((r**2 + a**2) * (E * (r**2 + a**2) - a * L) / Delta + a * (L - a * E)) / Sigma
        dphi_dtau = ((L / sin2 - a * E) + a * (E * (r**2 + a**2) - a * L) / Delta) / Sigma

        dr_dtau = r_dot
        dtheta_dtau = theta_dot

        # Derivatives of effective potentials (stabilized)
        dR_dr = (2 * (E * (r**2 + a**2) - a * L) * (2 * r * E)
                 - (2*r - 2*M) * (r**2 + (L - a * E)**2 + Q)
                 - 2 * Delta * r)

        dTheta_dtheta = (
            2 * cos_theta * sin_theta * (a**2 * (1 - E**2) + L**2 / sin2)
            - 2 * L**2 * cos2 / sin2**2
        )

        # Accelerations
        dr_dot_dtau = dR_dr / (2 * Sigma)
        dtheta_dot_dtau = dTheta_dtheta / (2 * Sigma)

        # Clamp second derivative if numerically unstable
        if not np.isfinite(dtheta_dot_dtau) or abs(dtheta_dot_dtau) > 1e5:
            dtheta_dot_dtau = np.clip(dtheta_dot_dtau, -1e5, 1e5)

        # Return full 6D system
        return np.array([
            dt_dtau,
            dr_dtau,
            dr_dot_dtau,
            dtheta_dtau,
            dtheta_dot_dtau,
            dphi_dtau
        ])


    # def geodesics(self, tau, state):
    #     t, r, r_dot, theta, theta_dot, phi = state
    #     M, a, E, L, Q = self.M, self.a, self.E, self.L, self.Q

    #     # Clamp theta to avoid sin(theta) → 0
    #     theta = np.clip(theta, 1e-6, np.pi - 1e-6)

    #     # Clamp theta_dot to avoid overshooting
    #     theta_dot = np.clip(theta_dot, -1e3, 1e3)

    #     # Trig functions
    #     sin_theta = np.sin(theta)
    #     cos_theta = np.cos(theta)
    #     sin2 = sin_theta**2
    #     cos2 = cos_theta**2

    #     # Metric components
    #     Delta = r**2 - 2*M*r + a**2
    #     Sigma = r**2 + a**2 * cos2

    #     # Effective potentials
    #     R = ((E * (r**2 + a**2) - a * L)**2 - Delta * (r**2 + (L - a * E)**2 + Q))
    #     Theta = Q - cos2 * (a**2 * (1 - E**2) + L**2 / sin2)

    #     # Coordinate derivatives
    #     dt_dtau = ((r**2 + a**2) * (E * (r**2 + a**2) - a * L) / Delta + a * (L - a * E)) / Sigma
    #     dphi_dtau = ((L / sin2 - a * E) + a * (E * (r**2 + a**2) - a * L) / Delta) / Sigma

    #     dr_dtau = r_dot
    #     dtheta_dtau = theta_dot

    #     # Derivatives of R and Theta potentials
    #     def R_derivative(r, E, L, Q, M, a, Delta):
    #         dDelta_dr = 2*r - 2*M
    #         term1 = 2 * (E * (r**2 + a**2) - a * L) * (2 * r * E)
    #         term2 = -dDelta_dr * (r**2 + (L - a * E)**2 + Q)
    #         term3 = -2 * Delta * r
    #         return term1 + term2 + term3

    #     def Theta_derivative(theta, E, L, Q, a):
    #         sin_theta = np.sin(theta)
    #         cos_theta = np.cos(theta)
    #         sin2 = sin_theta**2
    #         cos2 = cos_theta**2

    #         if sin2 < 1e-12:  # Avoid divide by zero
    #             sin2 = 1e-12

    #         dTheta_dtheta = (
    #             2 * cos_theta * sin_theta * (a**2 * (1 - E**2) + L**2 / sin2)
    #             - 2 * L**2 * cos_theta**2 / sin2**2
    #         )
    #         return dTheta_dtheta

        # # Accelerations (second-order derivatives)
        # dr_dot_dtau = 0.5 * R_derivative(r, E, L, Q, M, a, Delta) / Sigma
        # dtheta_dot_dtau = 0.5 * Theta_derivative(theta, E, L, Q, a) / Sigma

        # # Clamp thetä to prevent explosion
        # dtheta_dot_dtau = np.clip(dtheta_dot_dtau, -1e5, 1e5)

        # # Optional: detect numerical instability
        # if not np.isfinite(dtheta_dot_dtau):
        #     print(f"[WARNING] Non-finite θ̈ detected! θ = {theta}, d²θ/dτ² = {dtheta_dot_dtau}, Θ = {Theta}")

        # return np.array([dt_dtau, dr_dtau, dr_dot_dtau, dtheta_dtau, dtheta_dot_dtau, dphi_dtau])
###### EXTENSION END #######


class SchwarzschildMetric():
    def __init__(self, M, E, L):
        self.M = M # Black hole mass
        self.E = E # Particle energy (conserved)
        self.L = L # Particle angular momentum (conserved)

    def effective_potential(self, r, L=None):
        """
        Return the effective potential (V_eff) of the Schwarzschild geodesic

        Parameters:
        r: float
            Radial value
        L: float   
            Test particle angular momentum
        """
        if L is None:
            L = self.L
        return (1 - 2*self.M / r) * (1 + L**2 / r**2)

    def geodesics(self, tau, state):
        """
        Return the geodesics of each dimensional quantity

        Parameters:
        tau: float
            Current proper time value
        state: array
            Array of the current state variables ([r, rdot, phi, phidot]

        Returns:
        geodesics: array
            Array of the ODE of each state variable ([dt_dtau, dr_dtau, drdot_dtau, dphi_dtau])
        """
        # Assign variables from state array
        t, r, rdot, phi = state
        M, E, L = self.M, self.E, self.L # Constants

        # Reduced first-order ODEs
        dt_dtau = E / (1 - (2 * M / r))
        dr_dtau = rdot
        drdot_dtau = (-M / r**2) + (L**2 * (r - 3 * M)) / r**4
        dphi_dtau = L / r**2

        geodesics = np.array([dt_dtau, dr_dtau, drdot_dtau, dphi_dtau])
        return geodesics


class NewtonianGravity():
    def __init__(self, M):
        self.M = M # Black hole mass

    def geodesics(self, t, state):
        """
        Return the radial and angular geodesics for the Newtonian equation of motion

        Parameters:
        t: float
            Current time value
        state: array
            Array of the current state variables ([r, rdot, phi, phidot]

        Returns:
        geodesics: array
            Array of the ODE of each state variables
        """
        # Assign variables from state array
        r, rdot, phi, phidot = state
        M = self.M # Constant

        # ODEs for each state variable
        dr_dt = rdot
        dphi_dt = phidot
        drdot_dt = r * phidot**2 - M / r**2
        dphidot_dt = -2 * rdot * phidot / r

        geodesics = np.array([dr_dt, drdot_dt, dphi_dt, dphidot_dt])    
        return geodesics


class InitialConditions():
    def __init__(self, M):
        self.M = M # Black hole mass

    def circular_orbit_sc(self, r0=7.0):
        """
        Returns the initial conditions and the E (energy of the (particle-BH) system) and L (nngular momentum of the system)
        for the ISCO case

        Parameters:
        M: float
            Mass of the BH 
        r0: float
            Initial radius for this case (6M for ISCO)

        Returns:
        initial_conditions: array
            Array of initial conditions for this case ([t0, r0, theta0, phi0])
        conserved_quantities: array
            Array of the E and L values for this case ([E, L])
        
        """
        M = self.M 
        # Compute L_c for circular orbit at r0
        L = r0 * np.sqrt(M / (r0 - 3 * M))  
    
        # Compute E_c from V_eff using that L
        V_eff = (1 - 2 * M / r0) * (1 + L**2 / r0**2)
        E = np.sqrt(V_eff)  # Ensures E^2 = V_eff(r0)

        # Create arrays for the respected groups of values
        conserved_quantities = np.array([E, L])
        initial_conditions = np.array([0.0, r0, 0.0, 0.0])
        return initial_conditions, conserved_quantities

    def circular_orbit_newt(self, r0=7.0):
        """
        Returns the initial conditions and the E (energy of the (particle-BH) system) and L (angular momentum of the system)
        for the newtonian model

        Parameters:
        M: float
            Mass of the BH 
        r0: float
            Initial radius for this case 

        Returns:
        initial_conditions: array
            Array of initial conditions for this case ([r, rdot, phi, phidot])
        conserved_quantities: array
            Array of the E and L values for this case ([E, L])
        """
        M = self.M
        v_phi = np.sqrt(M / r0)  # orbital velocity magnitude
        phi_dot = v_phi / r0  # angular velocity
        E = 0.5 * v_phi**2 - M / r0
        L = v_phi * r0
        
        # Create arrays for the respected groups of values
        initial_state = np.array([r0, 0.0, 0.0, phi_dot])  # [r, rdot, phi, phidot]
        conserved_quantities = np.array([E, L])
        return initial_state, conserved_quantities
    
    def plunging_orbit_sc(self, r0=3.5, E_plunge=1.05, L_plunge=3.0):
        """
        Returns the initial conditions and the E (energy of the (particle-BH) system) and L (nngular momentum of the system)
        for the plunging orbit case

        Parameters:
        M: float
            Mass of the BH 
        r0: float
            Initial radius for this case (4.5M for plunging orbit)
        E: float
            Systen energy (E>1 for plunging)
        L: float
            System angular momentum

        Returns:
        initial_conditions: array
            Array of initial conditions for this case ([t0, r0, r0dot, phi0])
        conserved_quantities: array
            Array of the E and L values for this case ([E, L])
        """
        M = self.M 

        # Create arrays for the respected groups of values
        conserved_quantities = np.array([E_plunge, L_plunge])
        initial_conditions = np.array([0.0, r0, 0.0, 0.0])
        return initial_conditions, conserved_quantities

    def precessing_orbit_sc(self, r0=9.0, E_precess=0.92, L_precess=0.95):
        """
        Returns the initial conditions and the E (energy of the (particle-BH) system) and L (nngular momentum of the system)
        for the precessing orbit case

        Parameters:
        M: float
            Mass of the BH 
        r0: float
            Initial radius for this case (8M for precessing orbit)

        Returns:
        initial_conditions: array
            Array of initial conditions for this case ([t0, r0, r0dot, phi0])
        conserved_quantities: array
            Array of the E and L values for this case ([E, L])      
        """
        M = self.M 

        # Compute E_c and L_c (circular values)
        E_c = (1 - 2*M/r0)**0.5 / (1 - 3*M/r0)**0.5
        L_c = (M * r0)**0.5 / (1 - 3*M/r0)**0.5
    
        # Lower energy slightly to induce oscillation between turning points
        E = E_c * E_precess
        L = L_c * L_precess # Reduce to make precession for apparent

        # Create arrays for the respected groups of values
        conserved_quantities = np.array([E, L])
        initial_conditions = np.array([0.0, r0, 0.0, 0.0])
        return initial_conditions, conserved_quantities
    
    ###### PROJECT EXTENSION #######
    def kerr_equatorial(self, r0=8.0, a=0.9):
        """
        Returns the initial conditions and the E (energy of the (particle-BH) system) and L (angular momentum of the system)
        for the equatorial Kerr orbit case (theta = pi/2). Q = 0 since theta is fixed

        Parameters:
        M: float
            Mass of the BH 
        r0: float
            Initial radius for this case (6M - 12M)
        a: float
            BH spin parameter

        Returns:
        initial_conditions: array
            Array of initial conditions for this case ([t0, r0, r0dot, phi0])
        conserved_quantities: array
            Array of the E and L values for this case ([E, L])      
        """
        M = self.M
        Q = 0.0 
        theta0 = np.pi / 2

        # Compute E and L (for a circular orbit)
        E_top = r0**1.5 - 2*M*r0**0.5 + a*M**0.5
        E_bott = r0**0.75 * np.sqrt(r0**1.5 - 3*M*r0**0.5 + 2*a*M**0.5)
        L_top = M**0.5 * (r0**2 - 2*a*r0**0.5 + a**2)
        L_bott = r0**0.75 * np.sqrt(r0**1.5 - 3*M*r0**0.5 + 2*a*M**0.5)

        E = E_top / E_bott
        L = L_top / L_bott

        # Create arrays for the respected groups of values
        conserved_quantities = np.array([E, L, Q])
        initial_conditions = np.array([0.0, r0, 0.0, theta0, 0.0, 0.0])
        return initial_conditions, conserved_quantities
    
    def kerr_3D(self, r0=9.0, a=0.9, Q=10.0, E=0.95, L=3.0, theta0=np.pi/3, inward=True):
        """
        Returns the initial conditions and the E (energy of the (particle-BH) system) and L (angular momentum of the system)
        for the 3D Kerr orbit case (theta = variable). Q =! 0

        Parameters:
        M: float
            Mass of the BH 
        r0: float
            Initial radius for this case (8M - 10M)
        a: float
            BH spin parameter
        Q: float
            Carter constant
        theta0: float
            Initial (non-equatorial) theta value

        Returns:
        initial_conditions: array
            Array of initial conditions for this case ([t0, r0, r0dot, theta0, theta0dot, phi0])
        conserved_quantities: array
            Array of the E and L values for this case ([E, L])      
        """
        M = self.M
        cos_theta0 = np.cos(theta0)
        sin_theta0 = np.sin(theta0)

        # Metric components
        Sigma = r0**2 + a**2 * cos_theta0**2
        Delta = r0**2 - 2*M*r0 + a**2

        # Radial and polar potentials
        R = ((E * (r0**2 + a**2) - a * L)**2 - Delta * (r0**2 + (L - a * E)**2 + Q))
        Theta = Q - cos_theta0**2 * (a**2 * (1 - E**2) + L**2 / sin_theta0**2)

        # Derive initial velocities
        rdot0 = np.sqrt(np.abs(R)) / Sigma
        thetadot0 = np.sqrt(np.abs(Theta)) / Sigma

        if inward:
            rdot0 *= -1  # Start falling inward

        # Pack state
        initial_conditions = np.array([0.0, r0, rdot0, theta0, thetadot0, 0.0])
        conserved_quantities = np.array([E, L, Q])
        return initial_conditions, conserved_quantities
    ###### EXTENSION END #######


def plot_effective_potential(ic_class, metric_class, orbit_name, method, r_range=(2.1, 50), color='tab:blue'):
    """
    Function for plotting the effective potential curves for each plot. E^2 for each
    case is also plotted to see if it is accurate with the necessary requirements:

    Circular: E^2 = V_eff(r0)
    Plunging: E^2 = max(V_eff)
    Precessing: min(V_eff) < E^2 < 1    
    """
    initial_state, conserved = method # = (state, (E, L))
    E, L = conserved
    r0 = initial_state[1] # Initial radial value

    metric = metric_class(M=ic_class.M, E=E, L=L) # = SchwarzschildMetric
    r = np.linspace(2, 50, 1000) # Range of radial values for V_eff
    V_eff = metric.effective_potential(r) # Effective Potential

    # Plot V_eff curve and E^2 and r0 lines
    plt.plot(r, V_eff, label=f'{orbit_name} (L={L:.2f})', color=color)
    plt.axhline(y=E**2, color=color, linestyle='--', label=f'$E^2$ ({orbit_name})')
    plt.axvline(x=r0, color=color, linestyle=':', label=f'$r_0$ ({orbit_name})')

# Instantiate initial conditions generator
ic = InitialConditions(M=1.0)

# Get each orbital case
init_circ, conserved_circ = ic.circular_orbit_sc(r0=6.0)
init_plunge, conserved_plunge = ic.plunging_orbit_sc(r0=3.5)
init_prec, conserved_prec = ic.precessing_orbit_sc(r0=12.0)

# Plot all
plt.figure(figsize=(10, 6))

plot_effective_potential(ic, SchwarzschildMetric, "Circular", (init_circ, conserved_circ), color='tab:blue')
plot_effective_potential(ic, SchwarzschildMetric, "Plunging", (init_plunge, conserved_plunge), color='tab:red')
plot_effective_potential(ic, SchwarzschildMetric, "Precessing", (init_prec, conserved_prec), color='tab:orange')

plt.title("Effective Potential $V_{\\text{eff}}(r)$ in Schwarzschild Spacetime")
plt.xlabel("Radius $r$")
plt.ylabel("$V_{\\text{eff}}(r)$")
plt.grid(True)
plt.ylim(0, 2)
plt.xlim(2, 10)
plt.legend()
plt.show()

def main():
    """
    Function to run orbit simulations for each case

    Returns:
    - 2D plot of Newtonian test particle orbit
    - 2D plot of Schwarzschild test particle circular orbit
    - 2D plot of Schwarzschild test particle plunging orbit
    - 2D plot of Schwarzschild test particle precessing orbit
    """
    M = 1.0 # Black Hole Mass
    init_cond = InitialConditions(M=M) # Instance of initial conditions

    # # Orbital period and simulation time
    # T_orbit = 2 * np.pi * np.sqrt(12.0**3 / M)
    # total_time = 2 * T_orbit
    # steps = 2000
    # dt = total_time / steps
    # time_array = np.linspace(0, total_time, steps)

    # Function to ensure plunging simulation stops at plunge
    def make_plunge_condition(M):
        def plunging_condition(state):
            r = state[1]  # Schwarzschild state = [t, r, rdot, phi]
            return r <= 2.001 * M  
        return plunging_condition

    # Orbit configurations
    orbit_config = {
    "Newtonian": {
        "init_func": init_cond.circular_orbit_newt,
        "metric_class": NewtonianGravity,
        "use_tau": False,
        "color": "tab:blue",
        "r0": 12.0
    },
    "Circular_sc": {
        "init_func": init_cond.circular_orbit_sc,
        "metric_class": SchwarzschildMetric,
        "use_tau": True,
        "color": "tab:green",
        "r0": 12.0
    },
    "Plunging_sc": {
        "init_func": init_cond.plunging_orbit_sc,
        "metric_class": SchwarzschildMetric,
        "use_tau": True,
        "color": "tab:red",
        "r0": 5.5,
        "E_plunge": 1.05,
        "L_plunge": 3.0,
        "termination_condition": make_plunge_condition(M)
    },
    "Precessing_sc": {
        "init_func": init_cond.precessing_orbit_sc,
        "metric_class": SchwarzschildMetric,
        "use_tau": True,
        "color": "tab:orange",
        "E_precess": 0.92,
        "L_precess": 0.94,
        "r0": 12.0
    },
    "Kerr_Equatorial": {
        "init_func": init_cond.kerr_equatorial,
        "metric_class": KerrMetricFull,
        "use_tau": True,
        "color": "tab:red",
        "r0": 6.0,
        "a": 0.9
    },
    "Kerr_3D": {
        "init_func": init_cond.kerr_3D,
        "metric_class": KerrMetricFull,
        "use_tau": True,
        "color": "tab:red",
        "r0": 9.0,
        "a": 0.9,
        "Q": 10.0,
        "E": 0.95,
        "L": 3.0,
        "theta0": np.pi/3
    }
    }

    # Main loop to assign each orbital states
    for name, cfg in orbit_config.items():
        # Orbital period and simulation time
        T_orbit = 2 * np.pi * np.sqrt(cfg["r0"]**3 / M)
        total_time = 20 * T_orbit
        steps = 20000
        dt = total_time / steps
        time_array = np.linspace(0, total_time, steps)

        # Get initial state and conserved quantities
        if name == "Plunging_sc":
            initial_state, (E, L) = cfg["init_func"](r0=cfg["r0"], E_plunge=cfg["E_plunge"], L_plunge=cfg["L_plunge"])
        elif name == "Precessing_sc":
            initial_state, (E, L) = cfg["init_func"](r0=cfg["r0"], E_precess=cfg["E_precess"], L_precess=cfg["L_precess"])
        elif name == "Kerr_Equatorial":
            initial_state, (E, L, Q) = cfg["init_func"](r0=cfg["r0"], a=cfg["a"])
        elif name == "Kerr_3D":
            initial_state, (E, L, Q) = cfg["init_func"](r0=cfg["r0"], a=cfg["a"], Q=cfg["Q"], E=cfg["E"], L=cfg["L"], theta0=cfg["theta0"])
        else:
            initial_state, (E, L) = cfg["init_func"](r0=cfg["r0"])

        # Metric instance
        if name == "Newtonian":
            metric = cfg["metric_class"](M=M)
        elif name == "Circular_sc" or name == "Precessing_sc" or name == "Plunging_sc":
            metric = cfg["metric_class"](M=M, E=E, L=L)
        elif name == "Kerr_Equatorial" or name == "Kerr_3D":
            metric = cfg["metric_class"](M=M, a=cfg["a"], Q=Q, E=E, L=L)

        # Terminating condition for plunging case
        termination_condition = cfg.get("termination_condition", None)

        # Solver and integrator
        solver = SolveODE(
            deriv_func=metric.geodesics,
            current_state=initial_state,
            time=time_array[0],
            h=dt,
            t_stop=time_array[-1],
            termination_condition=termination_condition
        )

        integrator = ODEIntegrator(solver, method="rk45_step")
        times, states = integrator.integrate()

        # Extract r, theta, and phi then convert to cartesian
        if name == "Newtonian":
            r = states[:, 0]
            phi = states[:, 2]
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            z = np.ones_like(x) * np.random.uniform(-2, 2)  # random z values 
        elif name == "Circular_sc" or name == "Precessing_sc" or name == "Plunging_sc":
            r = states[:, 1]
            phi = states[:, 3]
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            z = np.ones_like(x) * np.random.uniform(-2, 2)  # random z values 
        elif name == "Kerr_Equatorial" or name == "Kerr_3D":
            r = states[:, 1]
            theta = states[:, 3]
            phi = states[:, 5]
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

        # Plot orbit in 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(x, y, z, color=cfg["color"], alpha=0.8)

        ax.scatter(0, 0, 0, color="black", s=100, label="Black Hole")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_zlim(-15, 15)
        ax.set_title("Orbit in 3D")
        ax.legend(loc='upper right', fontsize='small')
        plt.tight_layout()
        plt.show()

    # Star Cluster in GR
    M_bh = 1.0            # Black hole mass (used in both Newtonian and GR)
    N_stars = 10          # Number of orbiting stars
    r_min, r_max = 4.0*M_bh, 12.0*M_bh  # Range for radial distribution (in units of M)
    T_orbit = 2 * np.pi * np.sqrt(12.0**3 / M_bh)
    total_time = 20 * T_orbit
    steps = 20000
    dt = total_time / steps
    time_array = np.linspace(0, total_time, steps)

    # Generate radii and angles for initial star positions
    np.random.seed(42)
    radii = np.random.uniform(r_min, r_max, size=N_stars)
    angles = np.random.uniform(0, 2 * np.pi, size=N_stars)

    # Initialize
    init_cond = InitialConditions(M=M_bh)
    gr_states = [] # Array of each star's state array
    gr_params = [] # Array of each star's conserved quantities

    # Obtain states and conserved quantities for each star
    for i in range(N_stars):
        r0 = radii[i]
        phi0 = angles[i]

        # Use circular orbit method for each star at r0
        init_state, (E, L) = init_cond.circular_orbit_sc(r0=r0)

        # Manually enable precessions
        E *= np.random.uniform(0.95, 1.0)
        L *= np.random.uniform(0.95, 1.0)

        # Adjust phi0 for individual position
        init_state[3] = phi0

        # Append states and parameters
        gr_states.append(init_state)
        gr_params.append((E, L))

    # Plot cluster in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, (state, (E, L)) in enumerate(zip(gr_states, gr_params)):
        metric = SchwarzschildMetric(M=M_bh, E=E, L=L)
        
        # ODE solver and integration
        solver = SolveODE(
            deriv_func=metric.geodesics,
            current_state=state,
            time=0.0,
            h=dt,
            t_stop=total_time
        )
        
        integrator = ODEIntegrator(solver, method="rk45_step")
        tau, star = integrator.integrate()

        # Star = [t, r, rdot, phi]
        r = star[:, 1]
        phi = star[:, 3]

        # Convert to cartesian
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = 0  

        ax.plot(x, y, z, label=f"Star {i+1}", alpha=0.8)

    ax.scatter(0, 0, 0, color='black', s=100, label="Black Hole")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_zlim(-15, 15)
    ax.set_title("GR Cluster Orbits in 3D (Projected)")
    ax.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
