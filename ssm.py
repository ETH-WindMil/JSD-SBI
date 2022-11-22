"""
Provides classes for linear and non-linear state-space modeling
"""

import abc
import copy
import odeint

import numpy as np


__author__ = 'Konstantinos Tatsis'
__email__ = 'konnos.tatsis@gmail.com'


class StateSpace(abc.ABC):

    """
    Class for the modeling of state-space systems. Represents the system as 
    continuous-time first order differential equation or discrete-time 
    difference equation.

    Attributes
    ----------
    dt: float
        The sampling time of the system, equal to None for systems in 
        continuous-time domain.
    inputSize: int
        The size of input vector.
    outputSize: int
        The size of output vector.
    stateSize: int
        The size of state vector.
    """

    @abc.abstractproperty
    def dt(self):
        pass

    @abc.abstractproperty
    def inputSize(self):
        pass

    @abc.abstractproperty
    def outputSize(self):
        pass

    @abc.abstractproperty
    def stateSize(self):
        pass


    @abc.abstractmethod
    def getState(self):
        pass

    @abc.abstractmethod
    def getOutput(self):
        pass


class Linear(StateSpace):

    """
    Class for the modeling of linear state-space systems. Represents
    the system as continuous-time first order differential equation or 
    discrete-time difference equation. [1] [2] [3]

    Parameters
    ----------
    A: ndarray
        The state matrix of the system, also referred to as system matrix.
    B: ndarray
        The input matrix of the state-space system. 
    C: ndarray
        The output matrix of the state-space system.
    D: ndarray
        The feedthrough matrix of the state-space system, also referred to as
        direct transmission matrix.
    dt: float, optional
        Sampling time [s] of the discrete-time state-space system. By default,
        the sampling time is initialized to None, which corresponds to a
        continuous-time system.

    Attributes
    ----------
    domain: str
        The time domain, "continuous" or "discrete", of the system.
    dt: float
        The sampling time of the system, equal to None for systems in 
        continuous-time domain.
    inputSize: int
        The size of input vector.
    outputSize: int
        The size of output vector.
    shape: tuple
        The shape of the system, equal to (inputSize, outputSize, stateSize).
    stateSize: int
        The size of state vector.

    Methods
    -------
    getOutput(state, inpt)
        Evaluate the measurement equation.
    getState(state, inpt)
        Evaluate the state equation.
    simulate(inpt, time, initial, method)
        Simulate the time response to arbitrary inputs.
    toAugmented(**parameters)
        Convert system to augmented state-space.
    toContinuous(method)
        Convert system to continuous-time state-space.
    toDiscrete(dt, method)
        Convert system to discrete-time state-space.

    Raises
    ------
    ValueError
        If the shapes of system matrices are not aligned.

    Examples
    --------
    >>> A = np.array([[0, 1], [0, 0]])
    >>> B = np.array([[0], [1]])
    >>> C = np.array([[1, 0]])
    >>> D = np.array([[0]])
    >>> ss = StateSpace(A, B, C, D)
    >>> print(ss)
    Continuous-time State-space system, dt: None
      A: (2x2) State matrix
      B: (2x1) Input matrix
      C: (1x2) Output matrix
      D: (1x1) Feedthrough matrix
    >>> ss.A
    array([[ 0.,  1.],
           [ 0.,  0.]])
    >>> ss.B
    array([[0],
           [1]])

    References
    ----------
     .. [1] M. Verhaegen, V. Verdult, "Filtering and System Identification,
        A Least Squares Approach", Cambridge University Press, 2007.

     .. [2] G. F. Franklin, D. J. Powell, M. L. Workman, "Digital Control of 
        Dynamic Systems", Prentice Hall, 1997.

     .. [3] K. J. Astrom, B. Wittenmark, "Computer-Controlled Systems:
        Theory and Design", Prentice Hall, 1990.
    """

    def __init__(self, A, B, C, D, dt=None):

        itr = (('A', 'B', 0), ('A', 'C', 1), ('C', 'D', 0), ('B', 'D', 1))

        for m, n, j in itr:
            try:
                if eval(m+'.shape[j]') != eval(n+'.shape[j]'):
                    error = 'The shapes of {} and {} are not aligned.'
                    raise ValueError(error.format(m, n))
            except IndexError:
                error = 'System matrices must be two-dimensional arrays.'
                raise IndexError(error)

        self._stateSize = A.shape[1]
        self._inputSize = B.shape[1]
        self._outputSize = C.shape[0]
        self.shape = (self.stateSize, self.inputSize, self.outputSize)
        self.domain = 'continuous' if dt is None else 'discrete'
        self._dt = dt

        self.A, self.B, self.C, self.D = A, B, C, D


    @property
    def dt(self):
        return self._dt        

    @property
    def inputSize(self):
        return self._inputSize

    @property
    def outputSize(self):
        return self._outputSize

    @property
    def stateSize(self):
        return self._stateSize


    def isContinuous(self):

        """
        Check if the state-space model is a continuous-time system.

        Returns
        -------
        flag: bool
            A boolean flag which is True if the system is a continuous-time
            state-space model and False otherwise.
        """

        flag = True if self.domain == 'continuous' else False

        return flag


    def isDiscrete(self):

        """
        Check if the state-space model is a discrete-time system.

        Returns
        -------
        flag: bool
            A boolean flag which is True if the system is a discrete-time
            state-space model and False otherwise.
        """

        flag = not(self.isContinuousTime())

        return flag


    def __str__(self):

        dgt = str(max([len(str(item)) for item in self.shape]))
        string = '{0}-time Linear State-space, dt: {1}\n'+\
                 '  A: ({2:'+dgt+'}x{3:<'+dgt+'}) State matrix\n'+\
                 '  B: ({4:'+dgt+'}x{5:<'+dgt+'}) Input matrix\n'+\
                 '  C: ({6:'+dgt+'}x{7:<'+dgt+'}) Output matrix\n'+\
                 '  D: ({8:'+dgt+'}x{9:<'+dgt+'}) Feedthrough matrix'
        string = string.format(self.domain.capitalize(), self.dt, 
                self.shape[0], self.shape[0], self.shape[0], self.shape[1], 
                self.shape[2], self.shape[0], self.shape[2], self.shape[1])

        return string


    def __eq__(self, other):

        if self.shape == other.shape and self.dt == other.dt:
            AnB = self.A == other.A and self.B == other.B
            CnD = self.C == other.C and self.D == other.D

            if AnB and CnD:
                value = True

        return value


    def toDiscrete(self, dt, method='zoh', terms=None):

        """
        Convert system to discrete-time StateSpace.

        Parameters
        ----------
        dt: float
            Sampling time [s] of the discrete-time state-space system.
        method: {'zoh', 'foh', 'polynomial', 'bilinear'}, optional
            Discretization method, specified as one of the values below. If 
            not specified, 'zoh' is used by default.

            'zoh'
                Zero-order hold which assumes piecewise constant signals over 
                the sample time dt.
            'polynomial'
                Based on the Taylor series expansion of the matrix exponential.
            'foh'
                First-order hold which assumes piecewise linear signals  over 
                the sample time dt.
            'bilinear' 
                Bilinear (Tustin) method.

        terms: int, optional
            The number of terms for the Taylor series expansion of the matrix
            exponential when using polynomial discretization. By number of 
            terms defaults to 2, which yields the Rectangular method also 
            known as Euler's forward formula, and it should always be greater
            than or equal to 2.

        Returns
        -------
        system: StateSpace
            Discrete-time state-space of the current system.

        Raises
        ------
        TypeError
            If an invalid discretization method is specified or if the number 
            of terms for polynomial discretization is not an integer.
        ValueError
            If the number of terms for polynomial discretization is smaller
            than 2.
        LinAlgError
            If the system matrix is singular and a zoh method is selected for
            discretization.

        Examples
        --------
        >>> A = np.array([[0, 1], [1, 0]])
        >>> B = np.array([[0], [1]])
        >>> C = np.array([[1, 0]])
        >>> D = np.array([[0]])
        >>> ss = StateSpace(A, B, C, D)
        >>> dss = ss.toDiscrete(dt=0.1)
        >>> print(dss)
        Discrete-time State-space, dt: 0.1
          A: (2x2) State matrix
          B: (2x1) Input matrix
          C: (1x2) Output matrix
          D: (1x1) Feedthrough matrix
        >>> dss.A
        array([[1.00500417, 0.10016675],
               [0.10016675, 1.00500417]])
        >>> dss.B
        array([[0.00500417],
               [0.10016675]])

        >>> dss = ss.toDiscrete(dt=0.1, method='polynomial')
        >>> dss.A
        array([[ 1. ,  0.1],
               [ 0.1,  1. ]])
        >>> dss.B
        array([[ 0. ],
               [ 0.1]])
        """

        if self.domain is 'discrete':
            string = 'System is already discretized with sampling time {}.\n'
            sys.stdout.write(string.format(self.dt))
            system = self
        else:
            if method is 'zoh':
                I = np.eye(self.stateSize)
                try:
                    A = sp.linalg.expm(self.A*dt)
                    B = (A-I).dot(np.linalg.inv(self.A).dot(self.B))
                except np.linalg.LinAlgError:
                    string = 'Matrix A is singular, try polynomial method.'
                    raise np.linalg.LinAlgError(string)
                C, D = self.C, self.D

            elif method is 'polynomial':
                if terms == None:
                    terms = 2
                else:
                    if not isinstance(terms, int):
                        error = 'the number of terms must be an integer.'
                        raise TypeError(error)
                    if terms < 2:
                        error = 'the number of terms cannot be smaller than 2.'
                        raise ValueError(error)

                product = np.eye(self.A.shape[0])
                A, B = product, np.zeros(self.A.shape)
                C, D = self.C, self.D

                for m in range(1, terms):
                    B += product*dt**m/math.factorial(m)
                    product = product.dot(self.A)
                    A += product*dt**m/math.factorial(m)

                B = B.dot(self.B)

            elif method is 'foh':
                sts, ins = self.stateSize, self.inputSize
                augm = np.zeros((sts+2*ins, sts+2*ins))
                augm[:sts, :sts] = self.A*dt
                augm[:sts, sts:sts+ins] = self.B*dt
                augm[sts:sts+ins, sts+ins:] = np.eye(ins)
                exp = sp.linalg.expm(augm)
                G1 = exp[:sts, sts:sts+ins]
                G2 = exp[:sts, sts+ins:]

                A = exp[:sts, :sts]
                B = G1+A.dot(G2)-G2
                C = self.C
                D = self.D+self.C.dot(G2)

            elif method is 'bilinear':
                I = np.eye(self.stateSize)
                inv = np.linalg.inv(I-self.A*dt/2)
                A = (I+self.A*dt/2).dot(inv)
                B = np.linalg.inv(self.A).dot(A-I).dot(self.B)
                C = self.C.dot(inv)
                D = self.D+self.C.dot(inv).dot(self.B)*dt/2

            else:
                raise TypeError('Invalid discretization method.')

            system = StateSpace(A, B, C, D, dt)

        return system


    def toContinuous(self, method='zoh', terms=None):
        
        """
        Convert system to continuous-time State-Space.

        Parameters
        ----------
        method: {'zoh', 'polynomial', 'bilinear'}, optional
            Discretization method, specified as one of the values below. If 
            not specified, 'zoh' is used by default.

            'zoh':          Zero-order hold which assumes piecewise constant 
                            signals over the sample time dt.

            'polynomial':   Based on the Taylor series expansion of the 
                            matrix exponential.

            'foh':          First-order hold which assumes piecewise linear
                            signals  over the sample time dt.

            'bilinear':     Bilinear (Tustin) method, which is equivalent to
                            the trapezoid rule.
        terms: int, optional
            The number of terms for the Taylor series expansion of the matrix
            exponential when using polynomial discretization. The number of 
            terms defaults to 2, which yields the Rectangular method also 
            known as Euler's forward formula, and it should always be greater
            than or equal to 2.

        Returns
        -------
        system: StateSpace
            Continuous-time state-space of the current system.

        Raises
        ------
        TypeError
            If an invalid discretization method is specified or if the number 
            of terms for polynomial discretization is not an integer.
        ValueError
            If the number of terms for polynomial discretization is smaller
            than 2.
        LinAlgError
            If the system matrix is singular and a zoh method is selected for
            discretization.

        Examples
        --------
        >>> css = ss.toContinuous(method='polynomial')
        >>> print(css)
        Continuous-time State-space, dt: None
          A: (2x2) State matrix
          B: (2x1) Input matrix
          C: (1x2) Output matrix
          D: (1x1) Feedthrough matrix
        >>> css.A
        array([[ 0.,  1.],
               [ 0.,  0.]])
        >>> css = ss.B
        array([[-0.05],
               [ 1.  ]])
        """

        if self.domain is 'continuous':
            sys.stdout.write('System is already in continuous-time domain.\n')
            system = self
        else:
            if method == 'zoh':
                I = np.eye(self.stateSize)

                try:
                    A = sp.linalg.logm(self.A)/self.dt
                    B = np.linalg.inv(self.A-I).dot(A).dot(self.B)
                except np.linalg.LinAlgError:
                    string = 'Matrix A-I is singular, try polynomial method.'
                    raise np.linalg.LinAlgError(string)

                C, D = self.C, self.D
            elif method == 'polynomial':
                if terms == None:
                    terms = 2
                else:
                    if not isinstance(terms, int):
                        error = 'the number of terms must be an integer.'
                        raise TypeError(error)
                    if terms < 2:
                        error = 'the number of terms cannot be smaller than 2.'
                        raise ValueError(error)

                I = np.eye(self.stateSize)
                product = self.A-I
                A, B, C, D = product, I, self.C, self.D

                for m in range(1, terms):
                    B += product*(-1)**m/(m+1)
                    product = product.dot(self.A-I)
                    A += product.dot(self.A-I)*(-1)**m/(m+1)

                A = A/self.dt
                B = B.dot(self.B)/self.dt
            elif method is 'foh':
                I = np.eye(self.stateSize)
                A = sp.linalg.logm(self.A)/self.dt
                B1 = np.linalg.matrix_power(A*self.dt, 2)
                B2 = np.linalg.matrix_power(self.A-I, -2)
                B = B1.dot(B2).dot(self.B)/self.dt
                
                C = self.C
                inv = np.linalg.inv(self.A-I)
                D = self.D+C.dot(inv).dot(self.dt*A.dot(inv)-I).dot(self.B)
            elif method is 'bilinear':
                I = np.eye(self.stateSize)
                A = 2*np.linalg.inv(self.A+I).dot(self.A-I)/self.dt
                B = np.linalg.inv(self.A-I).dot(A).dot(self.B)
                C = self.C.dot(I-A*self.dt/2)
                D = self.D-C.dot(np.linalg.inv(I-A*self.dt/2)).dot(B)*self.dt/2
            else:
                raise TypeError('Invalid discretization method.')

            system = StateSpace(A, B, C, D)

        return system


    def toAugmented(self, **parameters):

        """
        Convert system representation to augmented state-space, where the
        state vector is augmented with additional variables.

        Parameters
        ----------
        **parameters: arguments
            The augmented state-space can be instantiated with 0, 2 or 5
            arguments, as described below:

            - 0: The state vector is augmented with the input vector by
            assuming zero transition matrix in the continuous-time domain 
            and identity transition matrix in discrete-time domain.

            - 2: The state vector is augmented with the input vector by 
            specifying the transition matrices 
                * A21 relating input to state
                * A22 modeling input evolution.

            - 5: The state vector is augmented with a parameter vector by 
            specifying the transition matrices 
                * A12 relating parameter(s) to state.
                * A21 relating state to parameter(s).
                * A22 modeling parameter(s) evolution.
                * B2  relating input to parameter(s).
                * C2  relating parameter(s) to output.

        Returns
        -------
        system: StateSpace
            The augmented state-space representation of the current system.

        Examples
        --------
        >>> A = np.array([[0, 1], [1, 0]])
        >>> B = np.array([[0], [1]])
        >>> C = np.array([[1, 0]])
        >>> D = np.array([[0]])
        >>> ss = StateSpace(A, B, C, D)
        >>> dss = ss.toDiscrete(dt=0.1)
        >>> augss = dss.toAugmented()
        >>> print(augss)
        Discrete-time State-space, dt: 0.1
          A: (3x3) State matrix
          B: (3x0) Input matrix
          C: (1x3) Output matrix
          D: (1x0) Feedthrough matrix
        >>> augss.A
        array([[1.00500417, 0.10016675, 0.00500417],
               [0.10016675, 1.00500417, 0.10016675],
               [0.        , 0.        , 1.        ]])
        >>> augss.B
        array([], shape=(3, 0), dtype=float64)
        >>> augss.C
        array([[1, 0, 0]])
        >>> augss.D
        array([], shape=(1, 0), dtype=float64)

        Notes
        -----
        Whether the state is augmented with the input or other system 
        parameters, the additional variables are appended at the end of state
        vector.
        """
        
        if len(parameters) in (0, 2):

            if len(parameters) == 0:
                A21 = np.zeros((self.inputSize, self.stateSize))
                A22 = np.eye(self.inputSize)

                if self.domain is 'continuous':
                    A22 = np.zeros((self.inputSize, self.inputSize))

            else:
                for key in parameters.keys():
                    if key not in ['A21', 'A22']:
                        error = 'Invalid transition matrix {}.'
                        raise TypeError(error.format(key))

                    if not isinstance(parameters[key], np.ndarray):
                        error = 'Invalid type of transition matrix {}.'
                        raise TypeError(error.format(key))

                if A21.shape != (self.inputSize, self.stateSize):
                    error = 'Invalid shape of transition matrix A21.'
                    raise ValueError(error)

                if A22.shape != (self.inputSize, self.inputSize):
                    error = 'Invalid shape of transition matrix A22.'
                    raise ValueError(error)

            A = np.vstack((np.hstack((self.A, self.B)), np.hstack((A21, A22))))
            B = np.zeros((A.shape[0], 0))
            C = np.hstack((self.C, self.D))
            D = np.zeros((C.shape[0], 0))

        elif len(parameters) == 5:

            for key in parameters.keys():
                if key not in ['A12', 'A22', 'A21', 'B2', 'C2']:
                    error = 'Invalid transition matrix {}.'
                    raise TypeError(error.format(key))

                if not isinstance(parameters[key], np.ndarray):
                    error = 'Invalid type of transition matrix {}.'
                    raise TypeError(error.format(key))

            if A22.shape[0] != A22.shape[1]:
                raise ValueError('A22 is not square matrix.')

            error = 'Invalid shape of transition matrix {}.'
            if A12.shape != (self.stateSize, A22.shape[0]):
                raise ValueError(error.format('A12'))

            if A21.shape != (A22.shape[0], self.stateSize):
                raise ValueError(error.format('A21'))

            if B2.shape != (A22.shape[0], self.inputSize):
                raise ValueError(error.format('B2'))

            if C2.shape != (self.outputSize, A22.shape[0]):
                raise ValueError(error.format('C2'))

            A = np.vstack((np.hstack((self.A, A12)), np.hstack((A21, A22))))
            B = np.vstack((self.B, B2))
            C = np.hstack((self.C, C2))
            D = self.D

        else:
            raise TypeError('Invalid number of arguments.')            

        system = StateSpace(A, B, C, D, self.dt)

        return system


    def getState(self, x, p):

        """
        Evaluate the state equation and get the state derivative, if the
        system is a continuous-time model, or the state at next time step, 
        if the system is a discrete-time model.

        Parameters
        ----------
        x: ndarray
            The current state vector.
        p: ndarray
            The current input vector.

        Returns
        -------
        x: ndarray
            The state at next time step or state derivative at current step.

        Examples
        --------
        >>> import syst
        >>> A = np.array([[0, 1], [1, 0]])
        >>> B = np.array([[0], [1]])
        >>> C = np.array([[1, 0]])
        >>> D = np.array([[0]])
        >>> ss = syst.StateSpace(A, B, C, D)
        >>> dss = ss.toDiscrete(dt=0.1)

        >>> x = np.ones(2)
        >>> p = np.array([1])
        >>> dss.getState(x, p)
        array([1.11017509, 1.20533767])
        """
        
        x = self.A.dot(x)+self.B.dot(p)

        return x


    def getOutput(self, x, p):

        """
        Evaluate the output equation and get the system output at current 
        time step, given the system state and input.

        Parameters
        ----------
        x: ndarray
            The current state vector.
        p: ndarray
            The current input vector.

        Returns
        -------
        output: ndarray
            The system output at current time step.

        Examples
        --------
        >>> import syst
        >>> A = np.array([[0, 1], [1, 0]])
        >>> B = np.array([[0], [1]])
        >>> C = np.array([[1, 0]])
        >>> D = np.array([[0]])
        >>> ss = syst.StateSpace(A, B, C, D)
        >>> dss = ss.toDiscrete(dt=0.1)

        >>> x = np.ones(2)
        >>> p = np.array([1])
        >>> dss.getOutput(x, p)
        array([1.])
        """
        
        y = self.C.dot(x)+self.D.dot(p)

        return y


    def simulate(self, p, t, initial=None, method='zoh'):

        """
        Simulate the time response of a continuous- or discrete-time system 
        to arbitrary inputs.

        Parameters
        ----------
        p: ndarray
            The input history, which has as many rows as the system inputs and 
            as many columns as the time samples.
        t: ndarray
            The regularly-spaced time samples of simulation.
        initial: ndarray
            The initial condition of system states.
        method: {'zoh', 'foh', 'bilinear'}
            The input interpolation method between samples.

        Returns
        -------
        y: ndarray
            The system output, sampled at the same time intervals as the input.
        x: ndarray
            The state trajectories, which has as many rows as the system 
            states and as many columns as the time samples.
        t: ndarray
            The time samples used for simulation.

        Raises
        ------
        ValueError
            If the shape of input or initial conditions is not aligned with
            the system matrices or if an invalid interpolation method is 
            specified.

        Notes
        -----
        If the system is represented in continuous-time domain, the simulation
        is performed upon tranformation to discrete-time domain, using by
        default a zero-order hold.
        """

        if p.shape[0] != self.inputSize:
            raise ValueError('Invalid shape of input history.')

        if p.shape[1] != t.shape[0]:
            raise ValueError('Non-aligned shapes of input and time vectors.')

        if initial is None:
            initial = np.zeros(self.stateSize)
        else:
            if initial.shape != (self.stateSize, ):
                raise ValueError('Invalid shape of initial conditions.')

        if method not in ['zoh', 'foh', 'bilinear']:
            raise ValueError('Invalid interpolation method.')

        # Check if time samples are regularly-spaced.

        step = t[1]-t[0]
        if not np.allclose(np.roll(t, -1)[:-1]-t[:-1], step):
            raise ValueError('Non regularly-spaced time samples.')

        # Check if the system is in continuous- or discrete-time domain.
        # In the latter case, an additional check is required, to verify that
        # equal sampling rates are used for the system and the time vector.

        if self.domain == 'continuous':
            system = self.toDiscrete(step, method)
            A, B = system.A, system.B
            C, D = system.C, system.D
        else:
            if self.dt != step:
                error = 'Non-equal sampling rates used for system and time.'
                raise ValueError(error)

            A, B = self.A, self.B
            C, D = self.C, self.D


        steps = t.shape[0]
        x = np.zeros((self.stateSize, steps))
        x[:, 0] = initial

        y = np.zeros((self.outputSize, steps))
        y[:, 0] = C.dot(x[:, 0])+D.dot(p[:, 0])

        for k in range(1, steps):
            x[:, k] = A.dot(x[:, k-1])+B.dot(p[:, k-1])
            y[:, k] = C.dot(x[:, k])+D.dot(p[:, k])

        return y, x, t



class Nonlinear(StateSpace):

    """
    Class for the modeling of nonlinear state-space systems. Represents the 
    system as continuous-time first-order differential equation or 
    discrete-time difference equation.  [?] ....

    Parameters
    ----------
    dt: float, optional
        Sampling time [s] of the discrete-time state space system. By default,
        the sampling time is initialized to None, which corresponds to a
        continuous-time system.
    ni: int, positive
        The number of system inputs.
    no: int, positive
        The number of system outputs.
    ns: int, positive
        The number of system states.
    output: callable
        The output(x, t, ...) equation function or method.
    state: callable
        The state(x, t, ...) equation function or method.

    Attributes
    ----------
    domain: str
        The time domain, "continuous" or "discrete", of the system.
    dt: float
        The sampling time of the system, equal to None for systems in
        continuous-time domain.

    Methods
    -------
    getOutput(x, t, p, *parameters)
        Evaluate the output (measurement) equation.
    getState(x, t, p, *parameters)
        Evaluate the state equation.
    simulate(t, p, initial, method, *settings)
        Simulate the time response to arbitrary inputs.
    toAugmented(**parameters)
        Convert system to augmented state-space.
    toContinuous(method)
        Convert system to continuous-time state-space.
    toDiscrete(dt, method)
        Convert system to discrete-time state-space.
    """

    def __init__(self, state, output, ns, ni, no, dt=None):

        self.state = state
        self.output = output

        self._stateSize = ns
        self._inputSize = ni
        self._outputSize = no

        self.domain = 'continuous' if dt is None else 'discrete'
        self._dt = dt


    def __str__(self):

        dgt = str(max([len(str(item)) for item in self.shape]))
        string = '{0}-time Nonlinear State-space, dt: {1}\n'+\
                 '  A: ({2:'+dgt+'}x{3:<'+dgt+'}) State matrix\n'+\
                 '  B: ({4:'+dgt+'}x{5:<'+dgt+'}) Input matrix\n'+\
                 '  C: ({6:'+dgt+'}x{7:<'+dgt+'}) Output matrix\n'+\
                 '  D: ({8:'+dgt+'}x{9:<'+dgt+'}) Feedthrough matrix'
        string = string.format(self.domain.capitalize(), self.dt, 
                self.shape[0], self.shape[0], self.shape[0], self.shape[1], 
                self.shape[2], self.shape[0], self.shape[2], self.shape[1])

        return string

    @property
    def dt(self):
        return self._dt        

    @property
    def inputSize(self):
        return self._inputSize

    @property
    def outputSize(self):
        return self._outputSize

    @property
    def stateSize(self):
        return self._stateSize


    def isContinuous(self):

        """
        Check if the state-space model is a continuous-time system.

        Returns
        -------
        flag: bool
            A boolean flag which is True if the system is a continuous-time
            state-space model and False otherwise.
        """

        flag = True if self.dt == None else False

        return flag


    def isDiscrete(self):

        """
        Check if the state-space model is a discrete-time system.

        Returns
        -------
        flag: bool
            A boolean flag which is True if the system is a discrete-time
            state-space model and False otherwise.
        """

        flag = not(self.isContinuous())

        return flag

    def toDiscrete(self, dt, method='RK4', *settings):
        
        """
        Convert system to discrete-time state-space.

        Parameters
        ----------
        dt: float
            Sampling time [s] of the discrete-time system.
        method: {'RK2', 'RK4'}
            Discretization method, specified as one of the values below.
            'RK1'
                First-order Runge-Kutta, also known as Euler's method.
            'RK2'
                Second-order Runge-Kutta, also known as midpoint method.
            'RK4'
                Fourth-order Runge-Kutta method.
        settings: non-keyworded argument list, optional
            The settings of time-integration scheme.

        Returns
        -------
        system: Nonlinear
            Discrete-time state-space of the current system.

        Raises
        ------
        ValueError
            If an invalid discretization method is specified.

        Examples
        --------
        >>> 
        """

        if self.domain is 'discrete':
            string = 'System is already discretized with sampling time {}.\n'
            sys.stdout.write(string.format(self.dt))
            system = self

        else:
            slvr = odeint.Solver(method)

            def model(x, t, *parameters):
                x = slvr.solver(self.state, x, t, dt, *parameters)
                return x

            ns, ni, no = self.stateSize, self.inputSize, self.outputSize
            system = Nonlinear(model, self.output, ns, ni, no, dt)

        return system


    def toAugmented(self, **parameters):
        
        """
        Convert system to an augmented state-space, where the state vector is
        augmented with additional variables.

        Parameters
        ----------
        **parameters: keyworded arguments
            The augmented state-space can be instantiated with 0, 2 or 4
            arguments, as described below:

            - 0: The state vector is augmented with the input vector by
            assuming zero transition matrix in the continuous-time domain
            and identity transition matri in discrete-time domain.

            - 1: The state vector is augmented with the input vector by
            spacifying the transition function
                * function(x, p, *parameters)

            - 3: The state vector is augmented with a parameter vector by
            specifying the following transition functions
                * function(m, *parameters)          # relating parameter(s) to state
                * function(x, m, p, *parameters)    # relating parameter(s), state and input to parameter(s)
                * function(m, *parameters)          # relating parameter(s) to output

        Returns
        -------
        system: Nonlinear
            The augmented state-space representation of the current system.

        Notes
        -----
        Whether the state is augmented with the input or other sytem-related
        parameters, the additional variables are appended to the end of the
        state vector.
        """

        if len(parameters) in (0, 1):

            if len(parameters) == 0:

                # augment the self.state function with the default function
                pass

            else:

                # augment the self.state function with the input function
                pass

        elif len(parameters) == 3:

            # augment with the given functions, both state and output
            pass

        else:
            raise ValueError('Invalid number of arguments')

        system = Nonlinear()

        return system


    def getState(self, x, t=None, p=None, *parameters):
        
        """
        Evaluate the state equation and get the state derivative, if the 
        system is represented by a continuous-time model, or the state at next 
        time step, if system is represented by a discrete-time model.

        Parameters
        ----------
        p: ndarray
            The input vector evaluated at time t.
        t: float, positive
            The time instant at which the state equation is to be evalauted.
        x: ndarray
            The state vector evaluated at time t.
        parameters: non-keyworded arguments, optional
            ...

        Returns
        -------
        x: ndarray
            The state derivative if the model is in continuous-time or the
            state vector at next time step if the model is in discrete-time 
            domain.
        """

        x = self.state(x, t, p, *parameters)

        return x


    def getOutput(self, x, t=None, p=None, *parameters):
        
        """
        Evaluate the output equation and get the system output, given the 
        system state and input.

        Parameters
        ----------
        p: ndarray
            The input vector evalauted at time t.
        t: float
            The time instant at which the output equation is to be evalauted.
        x: ndarray
            The state vector evaluated at time t.
        parameters: non-keyworded arguments, optional
            ...

        Returns
        -------
        y: ndarray
            The system output vector.
        """

        y = self.output(x, t, p, *parameters)

        return y


    def simulate(self, t, p=None, initial=None, method='RK4', *settings):
        
        """
        Simulate the time response of the continuous- or discrete-time system 
        to arbitrary inputs.

        Parameters
        ----------
        initial: ndarray
            The initial condition of the system state.
        p: ndarray
            The input history, which has as many rows as the system inputs and 
            as many columns as the time samples contained in the time vector.
        t: ndarray
            The time samples of simulation.
        method: str
            The integration scheme.
        settings: non-keyworded arguments, optional
            ...

        Returns
        -------
        y: ndarray
            The system output, sampled at the same time intervals as the input.
        x: ndarray
            The state trajectories matrix, which has as many rows as the
            system states and as many columns as the time samples.
        t: ndarray
            The time samples used for simulation.

        Raises
        ------
        ValueError
            If an invalid integration method is specified.
            If the shapes of input and time vectors are not aligned.
            If an invalid shape of initial conditions vector is specified.
        """

        if p.shape[0] != self.inputSize:
            raise ValueError('Invalid shape of input history.')

        if p.shape[1] != t.shape[0]:
            raise ValueError('Non-aligned shapes of input and time vectors.')

        if initial is None:
            initial = np.zeros(self.stateSize)
        else:
            if initial.shape != (self.stateSize, ):
                raise ValueError('Invalid shape of initial conditions.')


        steps = t.shape[0]
        x = np.zeros((self.stateSize, steps))
        x[:, 0] = initial

        y = np.zeros((self.outputSize, steps))
        y[:, 0] = self.getOutput(x[:, 0], t[0], p[:, 0])


        if self.domain == 'continuous':
            func = self.getState
            slvr = odeint.Solver(method)

            for k in range(1, steps):

                dt = t[k]-t[k-1]
                x[: ,k] = slvr.solver(func, x[:, k-1], t[k-1], dt, p[:, k-1])
                y[:, k] = self.getOutput(x[:, k], t[k], p[:, k])

        else:
            if self.dt != t[1]-t[0]:
                error = 'Non-aligned sampling rates between system and time.'
                raise ValueError(error)

            if not np.allclose(np.roll(t, -1)[:-1] - t[:-1], t[1]-t[0]):
                raise ValueError('Non-regularly spaced time samples.')

            for k in range(1, steps):
                x[:, k] = self.getState(x[:, k-1], t[k-1], p[:, k-1])
                y[:, k] = self.getOutput(x[:, k], t[k], p[:, k])


        return y, x, t