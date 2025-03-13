import numpy as np
from scipy import interpolate
from scipy import optimize

class ConsumptionSavingModel:

    def __init__(self, mp):
        ''' Initialize the model object

        Args:
            mp (SimpleNamespace) : model parameters
        
        Returns
            (ConsumptionSavingModel): model object
        '''
        
        # a. Parse parameters
        self.rho = mp.rho
        self.kappa = mp.kappa
        self.nu = mp.nu
        self.r = mp.r
        self.beta = mp.beta
        self.Delta = mp.Delta
        self.y_prb_hi = mp.y_prb_hi
        self.max_debt = mp.max_debt
        self.tau = mp.tau
        self.gamma = mp.gamma
        self.ybar = mp.ybar

        # b. Containers
        self.sim_m1 = []

    def utility(self, c):
        ''' Calculate flow utility of consumption level c

        Args:
            c (ndarray): level of consumtion

        Returns:
            (ndarray): flow utility of consumption
        '''

        return (c**(1-self.rho))/(1-self.rho)

    def bequest(self, m, c):
        ''' Calculate flow utility of leaving bequest given residual consumption

        Args:
            m (ndarray): cash-on-hand
            c (ndarray): level of consumtion

        Returns:
            (ndarray): utility of bequests
        '''

        return (self.nu*(m-c + self.kappa)**(1-self.rho))/(1-self.rho)


    def v2(self, c2, m2):
        ''' Compute state specific value of consumption choice and bequests in period 2

        Args:
            c2 (ndarray): level of consumtion in period 2
            m2 (ndarray): cash-on-hand in period 2

        Returns:
            (ndarray): value of comsumption and bequests
        '''

        return self.utility(c2) + self.bequest(m2,c2)

    def v1(self, c1, s1, m1, v2_interp):
        ''' Compute state specific value of consumption choice in period 1

        Args:
            c1 (ndarray): level of consumtion in period 1
            m1 (ndarray): cash-on-hand in period 1
            s1 (ndarray): study choice in period 1
            v2_interp (RegularGridInterpolator): interpolator between m in period 2 and value function

        Returns:
            (ndarray): state specific value of consumption choice in period 1
        '''

        # a.1 Initialize variables
        expected_v2 = 0.0
        low_y = self.ybar + self.gamma*s1 - self.Delta
        high_y = self.ybar + self.gamma*s1 + self.Delta

        # a.3 Assets at the end of period 1
        a1 = m1 - c1 - self.tau*s1*np.ones_like(m1)

        # b. Compute expectation of v2 given the set of possible interest rate and income realizations 
        m2_low_y = (1+self.r)*a1 + low_y
        v2_low_y = (1-self.y_prb_hi)*v2_interp([m2_low_y])

        m2_high_y = (1+self.r)*a1 + high_y
        v2_high_y = self.y_prb_hi*v2_interp([m2_high_y])

        expected_v2 = v2_low_y + v2_high_y

        # c. Return value v1 of consumption c1 and expected v2
        return self.utility(c1) + self.beta*expected_v2

    def solve_period_2(self):
        ''' Solve the consumption problem of period 2

        Returns:
            m2s (ndarray): cash-on-hand levels in period 2
            v2s (ndarray): value function in period 2
            c2s (ndarray): consumption function in period 2 (ie policy function)
        '''

        # a. grids
        m2s = np.linspace(1e-4,5,500)
        v2s = np.empty(500)
        c2s = np.empty(500)

        # b. solve for each m2 in grid
        for i,m2 in enumerate(m2s):

            # i. objective
            obj = lambda x: -self.v2(x[0],m2)

            # ii. initial value (consume half)
            x0 = m2/2

            # iii. optimizer
            result = optimize.minimize(obj,[x0],method='L-BFGS-B',
            bounds=((1e-8,m2),))

            # iv. save
            v2s[i] = -result.fun
            c2s[i] = result.x
            
        return m2s,v2s,c2s
    
    def solve_period_1_full(self, v2_interp):
        ''' Solve the consumption problem of period 1

        Args:
            v2_interp (RegularGridInterpolator): interpolator between m in period 2 and value function

        Returns:
            m1s (ndarray): cash-on-hand levels in period 1
            v1s (ndarray): value function in period 1
            c1s (ndarray): consumption function in period 1 (ie policy function)
        '''
        # m grid
        m1s = np.linspace(1e-8, 4, 100)

        # solve period 1 cons with no study
        v1s_no_study, c1s_no_study = self.solve_period_1_cons(0, m1s, v2_interp)
        # solve period 1 cons with study
        v1s_study, c1s_study = self.solve_period_1_cons(1, m1s, v2_interp)

        # decide study values: for each m1, compare v1s with and without study
        v1s = np.maximum(v1s_no_study, v1s_study)
        c1s = np.where(v1s_no_study >= v1s_study, c1s_no_study, c1s_study)
        s1s = np.where(v1s_no_study >= v1s_study, 0, 1)

        # return m1 grid, v1 grid (function of both c and s), optimal c1 and s1 policies
        return m1s, v1s, c1s, s1s
    
    def solve_period_1_cons(self, study_choice, m_grid, v2_interp):
        ''' Solve the consumption problem of period 1 for a specific study choice

        Args:
            v2_interp (RegularGridInterpolator): interpolator between m in period 2 and value function
            study_choice (int): study choice in period 1
            m_grid (ndarray): cash-on-hand grid in period 1

        Returns:
            v1s (ndarray): value function in period 1
            c1s (ndarray): consumption function in period 1 (ie policy function)
        '''

        # a. grids
        v1s = np.empty(100)
        c1s = np.empty(100)

        # b. solve for each m1s in grid
        for i, m1 in enumerate(m_grid):
            # Check if education is feasible when study_choice is 1
            if study_choice == 1 and m1 < self.tau:
                # Education is infeasible, set value to -inf and consumption to 0
                v1s[i] = float('-inf')
                c1s[i] = 0
            else:
                # i. objective
                def obj(x): return -self.v1(x[0], study_choice, m1, v2_interp)

                # ii. initial guess (consume half of available funds)
                x0 = (m1 - self.tau*study_choice)/2

                # Make sure initial guess is positive
                x0 = max(1e-12, x0)

                # iii. optimize
                result = optimize.minimize(
                    obj, [x0], method='L-BFGS-B', 
                    bounds=((1e-12, m1 - self.tau*study_choice + self.max_debt),))
                
                # iv. save
                v1s[i] = -result.fun
                c1s[i] = result.x[0]

        return v1s, c1s
    
    def solve(self):
        ''' Solve the consumption savings problem over all periods

        Returns:
            m1 (ndarray): cash-on-hand levels in period 1
            v1 (ndarray): value function in period 1
            c1 (ndarray): optimal consumption function in period 1 (ie policy function)
            s1 (ndarray): optimal education function in period 1 (ie policy function)
            m2 (ndarray): cash-on-hand levels in period 2
            v2 (ndarray): value function in period 2
            c2 (ndarray): optimal consumption function in period 2 (ie policy function)
        '''

        # a. solve period 2
        m2, v2, c2 = self.solve_period_2()

        # b. construct interpolator
        v2_interp = interpolate.RegularGridInterpolator([m2], v2,
                                                        bounds_error=False, fill_value=None)

        # b. solve period 1
        m1, v1, c1, s1 = self.solve_period_1_full(v2_interp)

        return m1, c1, s1, v1, m2, c2, v2

    def simulate(self):
        ''' Simulate choices in period 1 and 2 based on model solution and random draws of income.
        
        Returns:
            sim_c1 (ndarray): simulated consumption choices in period 1
            sim_s1 (ndarray): simulated education choices in period 1
            sim_c2 (ndarray): simulated consumption choices in period 2
            sim_m2 (ndarray): simulated cash-on-hand in period 2
        '''
            
        # a. solve the model at current parameters
        m1, c1, s1, _, m2, c2, _ = self.solve()
        
        # b. construct interpolators
        c1_interp = interpolate.RegularGridInterpolator([m1], c1,
                                                    bounds_error=False, fill_value=None)
        
        s1_interp = interpolate.RegularGridInterpolator([m1], s1,
                                                    bounds_error=False, fill_value=None)
        
        c2_interp = interpolate.RegularGridInterpolator([m2], c2,
                                                    bounds_error=False, fill_value=None)
        
        # c. sim period 1 based on draws of initial m and solution
        sim_c1 = c1_interp(self.sim_m1)
        sim_s1 = np.round(s1_interp(self.sim_m1)).astype(int)  # Ensure education choice is binary (0 or 1)
        
        # d. calculate period 1 assets (subtracting education cost if applicable)
        sim_a1 = self.sim_m1 - sim_c1 - self.tau * sim_s1
        
        # e. transition to period 2 m based on random draws of income
        # Create income based on education choice and random draws
        income_draws = np.random.random(sim_a1.shape) < self.y_prb_hi
        
        # Base income affected by education choice
        y2_base = self.ybar + self.gamma * sim_s1
        
        # Add income shock
        y2 = np.where(income_draws, 
                    y2_base + self.Delta,  # High income
                    y2_base - self.Delta)  # Low income
        
        # f. Based on random draws of income, simulate period 2 cash on hand
        sim_m2 = (1+self.r)*sim_a1 + y2
        
        # g. Simulate period 2 consumption choice based on model solution
        sim_c2 = c2_interp(sim_m2)
        
        return sim_c1, sim_s1, sim_c2, sim_m2

