

import numpy as np


__author__ = 'Konstantinos Tatsis'
__email__ = 'konnos.tatsis@gmail.com'


class Integrator:

    """
    """

    def __init__(self):
        pass


    def setStep(self):
        pass


    def setInterval(self):
        pass


    def setSolver(self):
        pass


    def setInitialConditions(self):
        pass


    def integrate(self):
        pass



class Solver:

    """
    Class ...

    Parameters
    ----------
    solver: str
        ...

    Methods
    -------
    set(solver)
        Specify the solver to be used.
    RK1(system, y, x, dx, *parameters)
        First-order Runge-Kutta scheme.
    RK2(system, y, x, dx, *parameters)
        Second-order Runge-Kutta scheme.

    Atrributes
    ----------
    solver: callable
        ...
    """

    def __init__(self, solver='RK2'):

        self.__solvers = ['RK1', 'RK2', 'RK4']
        self.__default = self.__solvers[1]

        self.set(solver)


    def set(self, solver):

        """
        Specify the integration solver.

        Parameters
        ----------
        solver: {'RK1', 'RK2', 'RK4'}
            The name of the integration solver, specified as one of the 
            values below.
            
            'RK1'
                First order Runge-Kutta, also known as Euler's method. 
            'RK2'
                Second order Runge-Kutta.
            'RK4'
                Fourth order Runge-Kutta.

        Raises
        ------
        ValueError
            If an invalid integration solver is specified.
        """

        if solver not in self.__solvers:
            raise ValueError('Invalid integration scheme.')

        self.solver = getattr(self, solver)


    @staticmethod
    def RK1(system, y, x, dx, *parameters):

        k1 = dx*system(y, x, *parameters)
        y += k1

        return y


    @staticmethod
    def RK2(system, y, x, dx, *parameters):

        k1 = dx*system(x, y, *parameters)
        k2 = dx*system(x+0.5*dx, y+0.5*k1, *parameters)
        y += k2

        return y


    @staticmethod
    def RK4(system, y, x, dx, *parameters):
        
        k1 = dx*system(y, x, *parameters)
        k2 = dx*system(y+0.5*k1, x+0.5*dx, *parameters)
        k3 = dx*system(y+0.5*k2, x+0.5*dx, *parameters)
        k4 = dx*system(y+k3, x+dx, *parameters)
        y += k1/6+k2/3+k3/3+k4/6

        return y


    @staticmethod
    def RK5(system, y, x, dx, *parameters):

        """
        Fifth-order Runge-Kutta method.
        """

        k1 = dx*system(y, x, *parameters)
        k2 = dx*system(y+1/5*k1, x+1/5*dx)
        k3 = dx*system(y+3/40*k1+9/40*k2, x+3/10*dx)
        k4 = dx*system(y+44/45*k1-56/15*k2+32/9*k3, x+4/5*dx)
        dy = 19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4
        k5 = dx*system(y+dy, x+8/9*dx)
        dy = 9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5
        k6 = dx*system(y+dy, x+dx)
        y += 35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6

        return y