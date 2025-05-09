import numpy as np
import Physical_Constants_5
from itertools import combinations # Produces the sets of combinations for pairs
import copy


class Integrators:

    
    def __init__(self, N, f=np.sin, a=0, b=np.pi):
        """
        Initialize function, bounds, and step attributes
        """
        self.f = f
        self.a = a
        self.b = b - 1e-6  # Avoid singularity at z=1
        self.N = N

    
    def rectangle(self):
        """ 
        Rectangle rule approximates a function f from lower bound a to upper bound b
        by implementing a for loop.

        Parameters:
            f : function
                The function to be integrated, defined within a program with one argument
                (function input, e.g x). Should support numpy arrays.

            a : float
                Lower bound in which you are integrating f over

            b : float
                Upper bound in which you are integrating f over

            N : int
                The number of rectangles to sum

        Returns:
            rect: float
                Evaluated integral using the rectangle method
        """

        a, b = self.a, self.b
        f = self.f
        N = self.N
        
        h = (b - a) / N  # Width of rectangle
        sum = 0

        for i in range(1, N):
            sum += f(a + i * h)
        rect = sum * h
        
        return rect



    def trapezoidal(self):
        """ 
        Trapezoidal rule approximates a function f from lower bound a to upper bound b
        by computing the areas of trapezoids under the function.

        Parameters:
            f : function
                The function to be integrated, defined within a program with one argument
                (function input, e.g x). Should support numpy arrays.

            a : float
                Lower bound in which you are integrating f over

            b : float
                Upper bound in which you are integrating f over

            N : int
                The number of rectangles to sum

            Returns:
            trap: float
                Evaluated integral using the trapezoidal method
        """

        a, b = self.a, self.b
        f = self.f
        N = self.N
        
        h = (b - a) / N # Width of triangle
        sum = 0.0

        for i in range(1, N):
            sum += f(a + i * h)
        trap = (h * (f(a) + f(b) + 2.0 * sum)) / 2.0
        
        return trap

        
    def simpsons(self):
        """ 
        Simpson's 1/3 rule approximates the integrand of a function f with a second order polynomial.

        Parameters:
            f : function
                The function to be integrated, defined within a program with one argument
                (function input, e.g x). Should support numpy arrays.

            a : float
                Lower bound in which you are integrating f over

            b : float
                Upper bound in which you are integrating f over

            N : int
                The number of rectangles to sum

        Returns:
            simp: float
                Evaluated integral using simpson's method
        """

        a, b = self.a, self.b
        f = self.f
        N = self.N
        
        h = (b - a) / N  # Step size
        sum1 = f(a) + f(b)  # First and last term
    
        sum2 = 0.0  # Sum for odd indices 
        sum3 = 0.0  # Sum for even indices 
    
        # Sum over odd indices
        for i in range(1, N, 2):
            sum2 += f(a + i * h)
    
        # Sum over even indices
        for j in range(2, N, 2):
            sum3 += f(a + j * h)
    
        simp = (1/3) * (sum1 + 4 * sum2 + 2 * sum3) * h
        return simp

    


class RootFinders:

    def __init__(self, f, a, b, tol=1e-6, max_iter=100):
        
        """Initialize function, bounds, and step attributes"""
        self.f = f
        self.a = a
        self.b = b
        self.tol = tol
        self.max_iter = max_iter


    def bisection(self):
        """
        The bisection method is a simple, robust method to find roots.
        It works by iteratively narrowing down the interval [a,b] where the
        function f(x) changes sign. 

        Parameters:
        f : function
            The function to be searched for roots, defined within a program with one argument
            (function input, e.g x). Should support numpy arrays.

        a : float
            Lower bound of the function f

        b : float
            Upper bound of the function f

        tol : float
            Desired tolerance. When reached, the computation has achieved sufficient accuracy.

        Returns:
            Tuple of the root found and the number of iterations needed
        """

        a, b, f, tol = self.a, self.b, self.f, self.tol  # Use local copies
        i = 0  # Initialize iteration count
    
        if f(a) * f(b) >= 0.0:
            print("f(a) and f(b) must have different signs")
            return None, i
    
        while abs(b - a) > tol:
            mid = (a + b) / 2.0
    
            if abs(f(mid)) < tol:
                if abs(mid) < 1e-8:  # If the root is too close to zero, continue searching
                    print("Bisection method found x ≈ 0, ignoring...")
                    a = mid + tol  # Shift away from 0
                else:
                    return mid, i
            
            elif f(a) * f(mid) < 0.0:
                b = mid
            else:
                a = mid
            
            i += 1  # Update iteration count
    
        root = (a + b) / 2.0
        return root if abs(root) > 1e-8 else None, i  # Return None if root ≈ 0



    def newton(self, df, x0):
        """
        Uses the function’s derivative f'(x)to approximate the root. It iteratively refines a guess
        using a recursion formula

        Parameters:
        f : function
            The function to be searched for roots, defined within a program with one argument
            (function input, e.g x). Should support numpy arrays.

        tol : float
            Desired tolerance. When reached, the computation has achieved sufficient accuracy.

        df : function
            The derivative of the function f for a particular value of x

        x0 : float
            Initial guess of the root

        max_iter:
            Maximum amount of iterations

        Returns:
            Tuple of the root found and the number of iterations needed
        """
        
        f, tol, max_iter = self.f, self.tol, self.max_iter  # Use local copies
    
        for i in range(max_iter):
            if abs(df(x0)) < 1e-12:  # Avoid division by zero
                print(f"Newton's method: Derivative too small at x = {x0}")
                return None, i
    
            x_new = x0 - f(x0) / df(x0)
    
            if abs(x_new - x0) < tol:  # Stopping condition
                return x_new, i
            
            x0 = x_new  # Update guess
    
        return x0 if abs(x0) > 1e-8 else None, i  # Return None if root ≈ 0



    def secant(self, x0, x1):
        """
        Secant method is similar to Newton’s method but doesn’t require the derivative
        The derivative is approximated by using the function f(x) at two nearby points.

       Parameters:
        f : function
            The function to be searched for roots, defined within a program with one argument
            (function input, e.g x). Should support numpy arrays.

        tol : float
            Desired tolerance. When reached, the computation has achieved sufficient accuracy.

        x0 : float
            First nearby point

        x1 : float
            Second nearby point

        max_iter:
            Maximum amount of iterations

        Returns:
            Tuple of the root and the number of iterations completed
        """


        f, tol, max_iter = self.f, self.tol, self.max_iter  # Use local copies
    
        for i in range(max_iter):
            if abs(f(x1) - f(x0)) < 1e-12:  # Avoid division by zero
                print("Secant method: Division by zero risk")
                return None, i
    
            x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
    
            if abs(x2 - x1) < tol:  # Stopping condition
                return x2, i
            
            x0, x1 = x1, x2  # Update values
    
        return x2 if abs(x2) > 1e-6 else None, i  # Return None if root ≈ 0




class SolveODE:
    
    def __init__(self, deriv_func, current_state, time, h, t_stop, termination_condition=None):
        """
        Parameters
        deriv_func: function
            The derivative function (ODE) of consideration (e.g dPdr)
        current_state: array
            Array of the current dependent variable values (e.g [P, m])
        time: float
            Current value of the independent variable (e.g r)
        h: float
            Step size of the independent variable (e.g dt)
        t_stop:
            Terminating value of the independent variable (e.g rstop)
        termination_condition: function
            Checks to see if the current simulation should terminate (Plunging orbit case)
        
        """
        self.dfdt = deriv_func
        self.f = np.array(current_state, dtype=float)
        self.t = time
        self.dt = h # Change depending on problem
        self.t_stop = t_stop
        self.termination_condition = termination_condition

    def euler(self):
        """
        Performs one Euler step on the current state.
        Assumes the state matrix has shape (n_bodies, 2 * d).
        
        Returns:
        self.t : float
            Updated time
        self.f : np.ndarray
            Updated state matrix
        """
        # Forward Euler: f(t + dt) = f(t) + dt * df/dt
        self.f = self.f + self.dt * self.dfdt(self.t, self.f)
        self.t += self.dt
        return self.t, self.f

    def rk4(self):
        """
        Computes one step of the 4th Order Runge-Kutta method 
        
        Parameters
        dfdx: function
            The derivative function (ODE) of consideration (e.g dPdr)
        f: array
            State array of the current dependent variable values (e.g [P, m])
        h: float
            Step size of the independent variable (e.g dt)

        Returns:
        f_new: array
            Updated state array

        """

        k1 = self.dfdt(self.t, self.f)
        k2 = self.dfdt(self.t + 0.5*self.dt, self.f + 0.5*self.dt*k1)
        k3 = self.dfdt(self.t + 0.5*self.dt, self.f + 0.5*self.dt*k2)
        k4 = self.dfdt(self.t + self.dt, self.f + self.dt*k3)

        f_prime = (1/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.f = self.f + self.dt * f_prime
        self.t += self.dt
        return self.t, self.f

    def leapfrog(self):
        """
        Performs one step of the Leapfrog integration method.
        
        Supports 1D or 2D state arrays:
            - For 1D: [x0, x1, ..., v0, v1, ...]
            - For 2D: [[x0, x1, ..., v0, v1, ...], ...]

        Returns:
        f_new: array
            Updated state array [x_new, v_new]
        """

        n, d = self.f.shape            # Number of bodies and dimensionality
        d = d // 2                    # Dimension of the states array

        # Current position and velo
        x = self.f[:, :d]
        v = self.f[:, d:]

        # Acceleration at current state
        a0 = self.dfdt(self.t, self.f)[:, d:]
        
        # Error handling
        if not hasattr(self, 'v_half'):
            # Initialize v_half using a half-step of Euler
            self.v_half = v + 0.5 * self.dt * a0
    
        # Update position
        x_new = x + self.dt * self.v_half
    
        # Estimate acceleration at new position
        new_state = np.hstack([x_new, self.v_half])
        a_new = self.dfdt(self.t, new_state)[:, d:]
    
        # Update half-step velocity to full-step velocity
        v_new = self.v_half + 0.5 * self.dt * a_new

        # Update half-step velocity
        self.v_half += self.dt * a_new

        # Time step and state
        self.t += self.dt
        self.f = np.hstack([x_new, v_new])
        return self.t, self.f
    
    def rk45_step(self):
        """
        Performs a single RK45 step using Dormand-Prince method.
        Uses internal time (self.t), state (self.f), and step size (self.dt).
        
        Returns:
        y_next : np.array
        error_estimate : float
        """
        f = self.dfdt
        t = self.t
        y = self.f
        h = self.dt

        # Dormand-Prince coefficients
        c2 = 1/5
        c3 = 3/10
        c4 = 4/5
        c5 = 8/9
        c6 = 1.0
        c7 = 1.0

        a21 = 1/5
        a31 = 3/40; a32 = 9/40
        a41 = 44/45; a42 = -56/15; a43 = 32/9
        a51 = 19372/6561; a52 = -25360/2187; a53 = 64448/6561; a54 = -212/729
        a61 = 9017/3168; a62 = -355/33; a63 = 46732/5247; a64 = 49/176; a65 = -5103/18656
        a71 = 35/384; a73 = 500/1113; a74 = 125/192; a75 = -2187/6784; a76 = 11/84

        b1 = 35/384; b3 = 500/1113; b4 = 125/192; b5 = -2187/6784; b6 = 11/84; b7 = 0
        bs1 = 5179/57600; bs3 = 7571/16695; bs4 = 393/640; bs5 = -92097/339200; bs6 = 187/2100; bs7 = 1/40

        k1 = f(t, y)
        k2 = f(t + c2*h, y + h*a21*k1)
        k3 = f(t + c3*h, y + h*(a31*k1 + a32*k2))
        k4 = f(t + c4*h, y + h*(a41*k1 + a42*k2 + a43*k3))
        k5 = f(t + c5*h, y + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
        k6 = f(t + c6*h, y + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))
        k7 = f(t + c7*h, y + h*(a71*k1 + a73*k3 + a74*k4 + a75*k5 + a76*k6))

        y_next = y + h * (b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6)
        z_next = y + h * (bs1*k1 + bs3*k3 + bs4*k4 + bs5*k5 + bs6*k6 + bs7*k7)

        error = np.linalg.norm(y_next - z_next)

        return y_next, error


class ODEIntegrator:
    
    def __init__(self, solver, method='rk4'):
        self.solver = solver  # A SolveODE instance
        self.method = method  # Integration method

    def integrate(self, tol=1e-6, h_min=1e-6, h_max=0.1):
        """
        Integrates the ODE using adaptive RK45 or fixed-step methods.

        Parameters:
        tol : float
            Error tolerance for RK45
        h_min, h_max : float
            Minimum and maximum step sizes

        Returns:
        times, states : arrays
            Integrated time and state values

        """
        times = [self.solver.t]
        states = [self.solver.f.copy()]

        max_steps = 10000
        step_count = 0
        while self.solver.t < self.solver.t_stop and step_count < max_steps:
            if self.solver.termination_condition and self.solver.termination_condition(self.solver.f):
                print("Termination condition met — stopping integration.")
                break

            if self.method == 'rk4':
                self.solver.t, self.solver.f = self.solver.rk4()
                times.append(self.solver.t)
                states.append(self.solver.f.copy())
            elif self.method == 'leapfrog':
                self.solver.t, self.solver.f = self.solver.leapfrog()
                times.append(self.solver.t)
                states.append(self.solver.f.copy())
            elif self.method == 'rk45_step':
                y_next, err = self.solver.rk45_step()
                
                # Adaptive control
                if err < tol:
                    # Accept step
                    self.solver.t += self.solver.dt
                    self.solver.f = y_next
                    times.append(self.solver.t)
                    states.append(self.solver.f.copy())

                    # Increase step size if error is small
                    self.solver.dt *= min(2.0, 0.9 * (tol / err)**0.25)
                else:
                    # Reject step, try again with smaller dt
                    self.solver.dt *= max(0.1, 0.9 * (tol / err)**0.25)

                # Clamp h
                self.solver.dt = np.clip(self.solver.dt, h_min, h_max)

            else:
                raise ValueError(f"Unknown method: {self.method}")
            step_count += 1
        return np.array(times), np.array(states)


class Nbody:

    def __init__(self, state_0=None, masses=None, use_numba=False):

        # Initial State and Masses
        self.state_0 = state_0        # Set initial state matrix
        self.masses = masses          # Set the body masses array
    
        # Call the gravitational constant
        self.G = Physical_Constants_5.G # cgs units

        # Combinations of bodies
        self.pairs = list(combinations(range(self.state_0.shape[0]), 2))
        
        # Results of the simulation
        self.state_over_time = None
        self.all_energies = None

        # Flag to switch to numba method
        self.use_numba = use_numba  

        
    def state_deriv(self, t, state):
        """
        Compute state derivative for the input state
    
        Parameters:
        t: float
            Current time value
        state: array
            [x, y, vx, vy]
    
        Returns:
        state_deriv: array
            [vx, vy, ax, ay]
        """
    
        n, d = state.shape            # Number of bodies and dimensionality
        d = d // 2                    # Dimension of the states array
        pos_arr = state[:, :d]        # Sub matrix of positions ([x1, y1, z1],[x1, y2, z2])
        vel_arr = state[:, d:]        # Sub matrix of velocities ([vx1, vy1, z1],[vx1, vy2, vz2])
        epsilon = 1e11                # Softening parameter (cm)
    
        # Initialize state derivative array
        state_deriv = np.zeros_like(state)
        state_deriv[:, :d] = vel_arr   # Initial velocities same as input state

        # NumPy-based calculation
        acc_arr = np.zeros_like(pos_arr)
        for i, j in self.pairs:
            r_vec = pos_arr[j] - pos_arr[i]
            r_mag = np.linalg.norm(r_vec)
            softened_r = np.sqrt(r_mag**2 + epsilon**2)
            denom = softened_r ** 3
            a_i = self.G * self.masses[j] * r_vec / denom
            a_j = -self.G * self.masses[i] * r_vec / denom
            acc_arr[i] += a_i
            acc_arr[j] += a_j
    
        state_deriv[:, d:] = acc_arr
        return state_deriv
    
    
    def compute_energies(self, state):
        """
        Returns the different energies of the system
        
        Parameters:
        state: array
            The current state 
        masses: array
            Array of masses for each body
        G: float
            Gravitational constant
            
        Returns:
        KE: float
            Kinetic energy of the system 
        PE: float
            Potential energy of the system 
        """
    
        n, d = state.shape            # Number of bodies and dimensionality
        d = d // 2                    # Dimension of the states array
        pos_arr = state[:, :d]        # Sub matrix of positions ([x1, y1, z1],[x1, y2, z2])
        vel_arr = state[:, d:] # Sub matrix of velocities ([vx1, vy1, z1],[vx1, vy2, vz2])
    
        # Compute combined Kinetic Energy of the bodies
        KE = 0
        for i in range(n):
            KE += 0.5 * self.masses[i] * np.linalg.norm(vel_arr[i]) ** 2
    
        # Compute combined Gravitational Potential Energy of the bodies
        PE = 0
        for body_i, body_j in self.pairs:
            r = np.linalg.norm(pos_arr[body_j] - pos_arr[body_i]) # Distance between bodies
            PE -= self.masses[body_i] * self.masses[body_j] / r
        PE *= self.G 

        E_tot = KE + PE
    
        return KE, PE, E_tot


    def simulation(self, period, t0, dt, method="rk4"):
        """
        Runs the simulation using the ODEIntegrator class.
        
        Parameters:
            period : float
                Total integration time
            t0 : float
                Initial time
            dt : float
                Time step
            method : str
                Integration method: 'leapfrog' or 'rk4'
        
        Returns:
            state_over_time : np.ndarray
                Trajectory over time
        """
    
        # Total number of steps (used only for energy and storage)
        it = int(period / dt)
    
        # Set up the solver
        solver = SolveODE(
            deriv_func=self.state_deriv,
            current_state=copy.deepcopy(self.state_0),
            time=t0,
            h=dt,
            t_stop=period)
    
        # Use ODEIntegrator to integrate
        integrator = ODEIntegrator(solver, method=method)
        times, states = integrator.integrate()
    
        # Store results
        self.state_over_time = states
    
        # Compute energy at each time step
        self.all_energies = np.zeros((len(states), 3))
        for i, state in enumerate(states):
            KE, PE, E_tot = self.compute_energies(state)
            self.all_energies[i] = np.array([KE, PE, E_tot])
    
        return self.state_over_time

                
        