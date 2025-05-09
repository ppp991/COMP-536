import numpy as np
import os
import matplotlib.pyplot as plt
import Physical_Constants_5
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
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

        # Simplified trig functions
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

        # Return full 6D system
        return np.array([
            dt_dtau,
            dr_dtau,
            dr_dot_dtau,
            dtheta_dtau,
            dtheta_dot_dtau,
            dphi_dtau
        ])
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
        E_precess: float
            Scalar to alter E for precession
        L_precess: float
            Scalar to alter L for precession

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

        # if inward:
        #     rdot0 *= -1  # Start falling inward

        # Pack state and conserved quantities
        initial_conditions = np.array([0.0, r0, rdot0, theta0, thetadot0, 0.0])
        conserved_quantities = np.array([E, L, Q])
        return initial_conditions, conserved_quantities
    ###### EXTENSION END #######

# Helper Functions
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

def schwarzschild_to_cartesian(r, phi, z_offset=0.0):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = np.ones_like(x) * z_offset
    return x, y, z

def extract_states(name, states):
    if name.startswith("Newtonian"):
        r = states[:, 0]
        phi = states[:, 2]
        return r, None, phi
    elif "Kerr" in name:
        r = states[:, 1]
        theta = states[:, 3]
        phi = states[:, 5]
        return r, theta, phi
    else:
        r = states[:, 1]
        phi = states[:, 3]
        return r, None, phi

def initialize_orbit(cfg, init_cond):
    init_func = cfg["init_func"]
    r0 = cfg.get("r0", 6.0)

    if "E_plunge" in cfg:
        return init_func(r0=r0, E_plunge=cfg["E_plunge"], L_plunge=cfg["L_plunge"])
    elif "E_precess" in cfg:
        return init_func(r0=r0, E_precess=cfg["E_precess"], L_precess=cfg["L_precess"])
    elif "Kerr" in cfg["init_func"].__name__:
        return init_func(
            r0=r0, a=cfg["a"], Q=cfg.get("Q", 0.0),
            E=cfg.get("E", None), L=cfg.get("L", None),
            theta0=cfg.get("theta0", np.pi / 2)
        )
    else:
        return init_func(r0=r0)
    
# Function to ensure plunging simulation stops at plunge
def make_plunge_condition(M):
    def plunging_condition(state):
        r = state[1]  # Schwarzschild state = [t, r, rdot, phi]
        return r <= 2.0001 * M # Event horizon
    return plunging_condition

def classify_sc_orbit(r, rdot, L, M):
    # Dynamic classification
    v_r = rdot[-1]
    v_phi = L / r[-1]**2
    v_squared = v_r**2 + (r[-1]**2 * v_phi**2)
    E_kin = 0.5 * v_squared
    E_total = E_kin - M / r[-1]
    final_r = r[-1]
    dr = r[-1] - r[-2]

    if final_r <= 2.05:
        dynamic_label = "plunging"
    elif E_total > 0 and dr > 0:
        dynamic_label = "escaping"
    else:
        dynamic_label = "bound"
    return dynamic_label
    
def animate_orbit_2D(x, y, name, black_hole=True, color='tab:blue', filename='orbit.gif', frame_step=10):
    fig, ax = plt.subplots(figsize=(6, 6))
    horizon_circle = patches.Circle((0, 0), radius=2.0, color='black', alpha=0.2, label='Event Horizon')
    ax.add_patch(horizon_circle)
    ax.set_xlim(np.min(x)-1, np.max(x)+1)
    ax.set_ylim(np.min(y)-1, np.max(y)+1)
    ax.set_aspect('equal')
    ax.set_title(f"{name}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    line, = ax.plot([], [], color=color)
    point, = ax.plot([], [], 'o', color=color)
    if black_hole:
        ax.scatter(0, 0, color="black", s=100, label="Black Hole")

    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point

    def update(frame_idx):
        idx = frame_idx * frame_step
        if idx >= len(x):
            idx = len(x) - 1
        line.set_data(x[:idx], y[:idx])
        point.set_data([x[idx]], [y[idx]])
        return line, point

    num_frames = len(x) // frame_step
    ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval=20)
    ani.save(filename, writer='pillow', fps=30)
    plt.close()

def animate_3D_particles_motion(particle_data, filename="schwarzschild_particles.gif", duration_sec=10):
    print(f"Saving animated 3D particle motion to {filename}...")

    fps = 30
    total_frames = fps * duration_sec  # e.g. 300 frames for 10 seconds
    print(f"Generating animation: {total_frames} frames at {fps} fps")

    # Prepare uniform index mapping to stretch or compress each orbit
    frame_indices_list = []
    for pdata in particle_data:
        n = len(pdata['x'])
        frame_indices = np.linspace(0, n - 1, total_frames).astype(int)
        frame_indices_list.append(frame_indices)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Event horizon (sphere at r = 2M)
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    r_h = 2.0  # Schwarzschild radius
    x_h = r_h * np.cos(u) * np.sin(v)
    y_h = r_h * np.sin(u) * np.sin(v)
    z_h = r_h * np.cos(v)
    ax.plot_surface(x_h, y_h, z_h, color='black', alpha=0.1, zorder=0)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_zlim(-15, 15)
    ax.set_title("3D Schwarzschild Particle Motion")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.scatter(0, 0, 0, color="black", s=100, label="Black Hole")

    trails = []
    points = []

    for pdata in particle_data:
        trail, = ax.plot([], [], [], color=pdata['color'], alpha=0.6, label=pdata['label'])
        point, = ax.plot([], [], [], 'o', color=pdata['color'])
        trails.append(trail)
        points.append(point)

    def init():
        for trail, point in zip(trails, points):
            trail.set_data([], [])
            trail.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
        return trails + points

    def update(frame_idx):
        for i, pdata in enumerate(particle_data):
            idx = frame_indices_list[i][frame_idx]
            x, y, z = pdata['x'], pdata['y'], pdata['z']
            trails[i].set_data(x[:idx], y[:idx])
            trails[i].set_3d_properties(z[:idx])
            points[i].set_data([x[idx]], [y[idx]])
            points[i].set_3d_properties([z[idx]])
        ax.view_init(elev=30, azim=frame_idx * 1.5 % 360)  # slow camera rotation
        return trails + points

    writer = FFMpegWriter(fps=fps, bitrate=1800)
    ani = animation.FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=False)
    ani.save(filename, writer=writer)
    plt.close()

# Initialize particle data array
particle_data = []
def simulate_nbody_schwarzschild(particle_configs, init_cond, M, steps):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Event horizon (sphere at r = 2M)
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    r_h = 2.0  # Schwarzschild radius
    x_h = r_h * np.cos(u) * np.sin(v)
    y_h = r_h * np.sin(u) * np.sin(v)
    z_h = r_h * np.cos(v)
    ax.plot_surface(x_h, y_h, z_h, color='black', alpha=0.1, zorder=0)

    for i, cfg in enumerate(particle_configs):
        termination_condition = None
        T_orbit = 2 * np.pi * np.sqrt(cfg["r0"]**3 / M)
        total_time = 20 * T_orbit
        dt = total_time / steps

        if cfg['type'] == 'circular':
            state0, (E, L) = init_cond.circular_orbit_sc(r0=cfg['r0'])
        elif cfg['type'] == 'plunging' or cfg['type'] == 'escaping':
            state0, (E, L) = init_cond.plunging_orbit_sc(r0=cfg['r0'], E_plunge=cfg['E_plunge'], L_plunge=cfg['L_plunge'])
            termination_condition = make_plunge_condition(M)
        elif cfg['type'] == 'precessing':
            state0, (E, L) = init_cond.precessing_orbit_sc(r0=cfg['r0'], E_precess=cfg['E_precess'], L_precess=cfg['L_precess'])
        else:
            raise ValueError("Unknown particle type")

        metric = SchwarzschildMetric(M, E, L)
        solver = SolveODE(metric.geodesics, state0, 0.0, dt, total_time, termination_condition=termination_condition)
        integrator = ODEIntegrator(solver, method='rk45_step')
        times, states = integrator.integrate()

        r = states[:, 1]
        rdot = states[:, 2]
        phi = states[:, 3]

        dynamic_label = classify_sc_orbit(r, rdot, L, M)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = 0.0

        if isinstance(z, (float, int)):
            z = np.full_like(x, z)

        # Store for animation
        particle_data.append({
            'x': x,
            'y': y,
            'z': z,
            'color': cfg.get('color', 'tab:gray'),
            'label': f"Particle {i+1} ({dynamic_label})"
        })


        ax.plot(x, y, z, label=f"{dynamic_label}")
        # # Run animation
        # animate_3D_particles_motion(particle_data, filename="schwarzschild_multi_orbits.mp4", duration_sec=10)

    ax.scatter(0, 0, 0, color='black', s=100, label="Black Hole")
    ax.set_title("N-body Schwarzschild Orbits (RK4)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Particle ID")
    ax.legend()
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_zlim(-15, 15)
    plt.tight_layout()
    plt.show()

def main():
    M = 1.0  # Black Hole Mass
    init_cond = InitialConditions(M=M)

    # Orbit configuration for each case
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
        "Escaping_sc": {
            "init_func": init_cond.plunging_orbit_sc, 
            "metric_class": SchwarzschildMetric,
            "use_tau": True,
            "color": "tab:purple",
            "r0": 10.0,
            "E_plunge": 1.08,
            "L_plunge": 2.5 
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

    # Plot effective potentials
    def plot_all_effective_potentials():
        plt.figure(figsize=(10, 6))
        plot_effective_potential(init_cond, SchwarzschildMetric, "Circular",
                                init_cond.circular_orbit_sc(r0=6.0), color='tab:blue')
        plot_effective_potential(init_cond, SchwarzschildMetric, "Plunging",
                                init_cond.plunging_orbit_sc(r0=3.5), color='tab:red')
        plot_effective_potential(init_cond, SchwarzschildMetric, "Precessing",
                                init_cond.precessing_orbit_sc(r0=12.0), color='tab:orange')
        
        plt.title("Effective Potential $V_{\\text{eff}}(r)$ in Schwarzschild Spacetime")
        plt.xlabel("Radius $r$")
        plt.ylabel("$V_{\\text{eff}}(r)$")
        plt.grid(True)
        plt.ylim(0, 2)
        plt.xlim(2, 10)
        plt.legend()
        plt.show()

    plot_all_effective_potentials()

    # Main orbital simulation loop
    for name, cfg in orbit_config.items():
        steps = 20000
        T_orbit = 2 * np.pi * np.sqrt(cfg["r0"]**3 / M)
        total_time = 20 * T_orbit
        dt = total_time / steps
        time_array = np.linspace(0, total_time, steps)

        # Get initial state and conserved quantities
        initial_state, conserved = initialize_orbit(cfg, init_cond)
        if "Kerr" in name:
            E, L, Q = conserved
        else:
            E, L = conserved
            Q = cfg.get("Q", 0.0)

        # Instantiate metric
        if name == "Newtonian":
            metric = cfg["metric_class"](M=M)
        elif "sc" in name:
            metric = cfg["metric_class"](M=M, E=E, L=L)
        elif "Kerr" in name:
            metric = cfg["metric_class"](M=M, E=E, L=L, a=cfg.get("a", 0.0), Q=Q)

        # Termination condition
        termination_condition = cfg.get("termination_condition", None)
        if termination_condition is None and "Plunging" in name:
            termination_condition = make_plunge_condition(M)

        # Solve ODE
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

        # Extract and convert to Cartesian
        r, theta, phi = extract_states(name, states)
        if "Kerr" in name and theta is not None:
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
        else:
            z_offset = np.random.uniform(-2, 2)
            x, y, z = schwarzschild_to_cartesian(r, phi, z_offset=z_offset)

        # Plot individual orbit cases
        if "Kerr" in name:
            # 3D PLOT for Kerr orbits
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            # Event horizon (sphere at r = 2M)
            u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
            r_h = 2.0  # Schwarzschild radius
            x_h = r_h * np.cos(u) * np.sin(v)
            y_h = r_h * np.sin(u) * np.sin(v)
            z_h = r_h * np.cos(v)
            ax.plot_surface(x_h, y_h, z_h, color='black', alpha=0.1, zorder=0)
            ax.plot(x, y, z, color=cfg["color"], label=name)
            ax.scatter(0, 0, 0, color="black", s=100, label="Black Hole")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_title(f"3D Orbit Projection: {name}")
            ax.legend(fontsize='small')
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)
            ax.set_zlim(-15, 15)
            plt.tight_layout()
            plt.show()
        else:
            # 2D PLOT for others
            fig, ax = plt.subplots(figsize=(8, 6))
            horizon_circle = patches.Circle((0, 0), radius=2.0, color='black', alpha=0.2, label='Event Horizon')
            ax.add_patch(horizon_circle)
            ax.plot(x, y, color=cfg["color"], alpha=0.8, label=name)
            ax.scatter(0, 0, color="black", s=100, label="Black Hole")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)
            ax.set_title(f"2D Orbit Projection: {name}")
            ax.set_aspect('equal')
            ax.legend(fontsize='small')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # if "Kerr" not in name:  # Only animate non-Kerr 2D orbits
        #     animate_orbit_2D(x, y, name=name, color=cfg["color"], filename=f"{name}_orbit.gif")

    # 3D Schwarzschild multi-orbit visualization
    particle_configs = [
        {'type': 'circular',   'r0': 12.0, 'color': 'tab:blue'},
        {'type': 'precessing', 'r0': 6.0, 'E_precess': 1.02, 'L_precess': 3.8, 'color': 'tab:orange'}, # Minimum stable r0
        {'type': 'plunging',   'r0': 5.5, 'E_plunge': 1.1,  'L_plunge': 2.9, 'color': 'tab:red'},
        {'type': 'precessing', 'r0': 12.0, 'E_precess': 0.92, 'L_precess': 0.94, 'color': 'tab:green'}, # Maximum stable r0
        {'type': 'escaping',   'r0': 10.0, 'E_plunge': 1.08, 'L_plunge': 2.5, 'color': 'tab:purple'} # r0 too large to plunge
    ]

    # Simulation the differe Sc orbits for the given particle configuration
    simulate_nbody_schwarzschild(particle_configs, init_cond, M=M, steps=20000)

if __name__ == "__main__":
    main()