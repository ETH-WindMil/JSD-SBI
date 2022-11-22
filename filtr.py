"""
Provides classes for recursive input, state and parameter estimation.
"""

import abc
import sys
import copy
import inspect

import numpy as np
import scipy as sp
import collections as cl
import itertools as it


__author__ = 'Konstantinos Tatsis'
__email__ = 'konnos.tatsis@gmail.com'


class Base:

    """
    Base class used as a substitute of dict instances, which is instantiated 
    with any number of keyword arguments.

    Parameters
    ----------
    **kwargs: arguments
        The key-value pairs to be assigned as attributes of the class.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Filter(abc.ABC):

    @abc.abstractmethod
    def __init__(self, model):

        if type(model).__name__ not in self._models:
            error = 'model must be an ssm.{} instance.'
            raise ValueError(error.format('/'.join(self._models)))
        else:
            if model.isContinuous():
                raise TypeError('model must be a discrete-time system.')

            self.model = model


        s = self.model.stateSize
        o = self.model.outputSize

        self._timer = it.count(0, self.model.dt)
        self.time = next(self._timer)

        self.state = Base(mean=np.zeros(s), cov=1e5*np.eye(s))
        state = Base(mean=np.zeros(s), cov=np.zeros((s, s)))
        self.correction = Base(state=state)

        state = Base(mean=np.zeros(s), cov=1e-5*np.eye(s))
        output = Base(mean=np.zeros(o), cov=1e-5*np.eye(o))
        self.noise = Base(state=state, output=output)

        self.prediction = np.zeros(o)
        self.innovation = Base(mean=np.zeros(o), cov=1e-5*np.eye(o))
        self.residual = Base(mean=np.zeros(o), cov=np.eye(o))


    @abc.abstractmethod
    def predict(self, p=None):
        pass


    @abc.abstractmethod
    def correct(self, y, p=None):
        pass


    def setInitialState(self, mean, covariance):

        """
        Specify the initial value of the state estimate and the state
        estimation error covariance.

        Parameters
        ----------
        mean: ndarray
            The initial state estimate.
        covariance: ndarray
            The initial state estimation error covariance.

        Raises
        ------
        TypeError
            If the shape of mean or covariance is not aligned with the state 
            size.

        Examples
        --------
        >>> A = np.array([[0, 1], [0, 0]])
        >>> B = np.array([[0], [1]])
        >>> C = np.array([[1, 0]])
        >>> D = np.array([[0]])
        >>> model = syst.StateSpace(A, B, C, D, dt=0.02)
        >>> kf = filt.KalmanFilter(model)
        >>> kf.setInitialState(np.ones(2), 1e5*np.eye(2))
        >>> kf.state.mean
        array([ 1.,  1.])
        >>> kf.state.cov
        array([[ 100.,    0.],
               [   0.,  100.]])
        """

        error = 'Invalid shape of initial state {}.'

        if mean.shape != self.state.mean.shape:
            raise TypeError(error.format('vector'))

        if covariance.shape != self.state.cov.shape:
            raise TypeError(error.format('covariance matrix'))

        self.state.mean = mean
        self.state.cov = covariance


    def setProcessNoise(self, covariance):

        """
        Specify the process noise covariance matrix.

        Parameters
        ----------
        covariance: ndarray
            The process noise covariance matrix.

        Raises
        ------
        TypeError
            If the shape of covariance matrix is not aligned with the size of
            state.

        Examples
        --------
        >>> A = np.array([[0, 1], [0, 0]])
        >>> B = np.array([[0], [1]])
        >>> C = np.array([[1, 0]])
        >>> D = np.array([[0]])
        >>> model = syst.StateSpace(A, B, C, D, dt=0.02)
        >>> kf = filt.KalmanFilter(model)
        >>> kf.setInitialState(np.ones(2), 1e5*np.eye(2))
        >>> kf.setProcessNoise(1e-5*np.eye(2))
        >>> kf.processNoise
        array([[  1.00000000e-05,   0.00000000e+00],
               [  0.00000000e+00,   1.00000000e-05]])
        """

        if covariance.shape != self.noise.state.cov.shape:
            raise TypeError('Invalid shape of covariance matrix.')

        self.noise.state.cov = covariance


    def setMeasurementNoise(self, covariance):

        """
        Specify the measurement noise covariance matrix.

        Parameters
        ----------
        covariance: ndarray
            The measurement noise covariance matrix.

        Raises
        ------
        TypeError
            If the shape of covariance matrix is not aligned with the number
            of measurements.

        Examples
        --------
        >>> A = np.array([[0, 1], [0, 0]])
        >>> B = np.array([[0], [1]])
        >>> C = np.array([[1, 0]])
        >>> D = np.array([[0]])
        >>> model = syst.StateSpace(A, B, C, D, dt=0.02)
        >>> kf = filt.KalmanFilter(model)
        >>> kf.setInitialState(np.ones(2), 1e5*np.eye(2))
        >>> kf.setProcessNoise(1e-5*np.eye(2))
        >>> kf.setMeasurementNoise(1e-2*np.eye(1))
        >>> kf.measurementNoise
        array([[  1.00000000e-05]])
        """

        if covariance.shape != self.noise.output.cov.shape:
            raise TypeError('Invalid shape of covariance matrix.')

        self.noise.output.cov = covariance



class KF:

    """
    Kalman Filter (KF) class for online state estimation of discrete-time
    linear time-invariant (LTI) systems [1]. The Kalman filter is defined as 
    a recursive linear state estimator designed to be optimal in an unbiased 
    minimum-variance sense [2]. The algorithm works in a two-step process. 
    In the prediction step, the Kalman filter estimates the current state 
    variables along with their uncertainties. Once the next measurement is 
    observed, the estimates are updated using a weighted average and the 
    process advances to the next step.

    Parameters
    ----------
    model: ssm.Linear
        The discrete-time state-space model describing the system dynamics.

    Attributes
    ----------
    model: smm.Linear
        The discrete-time state-space model describing the system dynamics.
    state: Base
        The state of the model which is by default initialized with a zero 
        mean and identity error covariance matrix.
    prediction: ndarray
        The predicted output.
    innovation: Base
        The difference between actual measurements and prior measurement 
        predictions, also referred to as pre-fit residual.
    residual: Base
        The difference between actual measurements and posterior measurement
        prediction, also referred to as post-fit residual.
    correction: Base
        The state mean and state covariance correction.
    noise: Base
        The process and measurement noise terms which are by default 
        initialized with zero mean and unit covariance matrices.

    Methods
    -------
    setInitialState(mean, covariance)
        Specify the initial state estimate.
    setProcessNoise(covariance)
        Specify the process noise covariance matrix.
    setMeasurementNoise(covariance)
        Specify the measurement noise covariance matrix.
    updateProcessNoise(delta)
        Update the process noise covariance matrix.
    updateMeasurementNoise(delta)
        Update the measurement noise covariance matrix.
    predictState(input)
        Predict the state at next time step using the system model.
    correctState(output, input)
        Correct the state using measured system outputs.
    smoothState()
        Smooth the state by conditioning past estimates on future measurements.
    
    Raises
    ------
    ValueError
        If model is not an ssm.Linear instance.
    TypeError
        If model is not a discrete-time state-space system.

    References
    ----------
     .. [1] R. E. Kalman, "A new approach to linear filtering and prediction
        problems", Transactions of the ASME Journal of Basic Engineering, 
        82, pp. 35-45, 1960.

     .. [2] G. Welch, G. Bishop, "An introduction to Kalman filter",
        University of North Carolina Chapel Hill, 2001.

     .. [3] G. F. Franklin, J. D. Powell, M. L. Workman, "Digital Control of
        Dynamic Systems", Second Edition, Adison-Wesley, 1990.

    Examples
    --------
    >>> import syst
    >>> import filt
    >>> A = np.array([[0, 1], [0, 0]])
    >>> B = np.array([[0], [1]])
    >>> C = np.array([[1, 0]])
    >>> D = np.array([[0]])
    >>> model = syst.StateSpace(A, B, C, D, dt=0.02)
    >>> kf = filt.KalmanFilter(model)
    >>> kf.state.mean
    array([ 0.,  0.])
    >>> kf.state.cov
    array([[ 0.,  0.],
           [ 0.,  0.]])
    """


    def __init__(self, model):

        self.__models = ['Linear']

        self.predictionObservers = []
        self.correctionObservers = []



    def setInitialState(self, mean, covariance):

        """
        Specify the initial value of the state estimate and the state
        estimation error covariance.

        Parameters
        ----------
        mean: ndarray
            The initial state estimate.
        covariance: ndarray
            The initial state estimation error covariance.

        Raises
        ------
        TypeError
            If the shape of mean or covariance is not aligned with the state 
            size.

        Examples
        --------
        >>> A = np.array([[0, 1], [0, 0]])
        >>> B = np.array([[0], [1]])
        >>> C = np.array([[1, 0]])
        >>> D = np.array([[0]])
        >>> model = syst.StateSpace(A, B, C, D, dt=0.02)
        >>> kf = filt.KalmanFilter(model)
        >>> kf.setInitialState(np.ones(2), 1e5*np.eye(2))
        >>> kf.state.mean
        array([ 1.,  1.])
        >>> kf.state.cov
        array([[ 100.,    0.],
               [   0.,  100.]])
        """

        error = 'Invalid shape of initial state {}.'

        if mean.shape != self.state.mean.shape:
            raise TypeError(error.format('vector'))

        if covariance.shape != self.state.cov.shape:
            raise TypeError(error.format('covariance matrix'))

        self.state.mean = mean
        self.state.cov = covariance



    def setProcessNoise(self, covariance):

        """
        Specify the process noise covariance matrix.

        Parameters
        ----------
        covariance: ndarray
            The process noise covariance matrix.

        Raises
        ------
        TypeError
            If the shape of covariance matrix is not aligned with the size of
            state.

        Examples
        --------
        >>> A = np.array([[0, 1], [0, 0]])
        >>> B = np.array([[0], [1]])
        >>> C = np.array([[1, 0]])
        >>> D = np.array([[0]])
        >>> model = syst.StateSpace(A, B, C, D, dt=0.02)
        >>> kf = filt.KalmanFilter(model)
        >>> kf.setInitialState(np.ones(2), 1e5*np.eye(2))
        >>> kf.setProcessNoise(1e-5*np.eye(2))
        >>> kf.processNoise
        array([[  1.00000000e-05,   0.00000000e+00],
               [  0.00000000e+00,   1.00000000e-05]])
        """

        if covariance.shape != self.noise.state.cov.shape:
            raise TypeError('Invalid shape of covariance matrix.')

        self.noise.state.cov = covariance


    def setMeasurementNoise(self, covariance):

        """
        Specify the measurement noise covariance matrix.

        Parameters
        ----------
        covariance: ndarray
            The measurement noise covariance matrix.

        Raises
        ------
        TypeError
            If the shape of covariance matrix is not aligned with the number
            of measurements.

        Examples
        --------
        >>> A = np.array([[0, 1], [0, 0]])
        >>> B = np.array([[0], [1]])
        >>> C = np.array([[1, 0]])
        >>> D = np.array([[0]])
        >>> model = syst.StateSpace(A, B, C, D, dt=0.02)
        >>> kf = filt.KalmanFilter(model)
        >>> kf.setInitialState(np.ones(2), 1e5*np.eye(2))
        >>> kf.setProcessNoise(1e-5*np.eye(2))
        >>> kf.setMeasurementNoise(1e-2*np.eye(1))
        >>> kf.measurementNoise
        array([[  1.00000000e-05]])
        """

        if covariance.shape != self.noise.output.cov.shape:
            raise TypeError('Invalid shape of covariance matrix.')

        self.noise.output.cov = covariance



    def predict(self, p=None):

        """
        Predict the state and state estimation error covariance at the next 
        time step based on the system model.

        Parameters
        ----------
        p: ndarray, optional
            The system input. Note that even when the system is excited by a
            single input, p argument should be an ndarray instance and not
            a scalar.

        Examples
        --------
        >>> A = np.array([[0, 1], [0, 0]])
        >>> B = np.array([[0], [1]])
        >>> C = np.array([[1, 0]])
        >>> D = np.array([[0]])
        >>> model = syst.StateSpace(A, B, C, D, dt=0.02)
        >>> kf = filt.KalmanFilter(model)
        >>> kf.setInitialState(np.ones(2), 1e5*np.eye(2))
        >>> kf.setProcessNoise(1e-5*np.eye(2))
        >>> kf.setMeasurementNoise(1e-2*np.eye(1))
        >>> kf.predictState(np.array([2]))
        >>> kf.state.mean
        array([ 1.,  2.])
        >>> kf.state.cov
        array([[  2.00000000e-05,   0.00000000e+00],
               [  0.00000000e+00,   1.00000000e-05]])
        """

        self.step = next(self.counter)
        A, B = self.model.A, self.model.B
        p = np.zeros(B.shape[1]) if p is None else p

        self.state.mean = A.dot(self.state.mean)+B.dot(p)
        self.state.cov = A.dot(self.state.cov).dot(A.T)
        self.state.cov += self.noise.state.cov

        C, D = self.model.C, self.model.D
        self.prediction = C.dot(self.state.mean)+D.dot(p)

        for observer in self.predictionObservers:
            observer(self.state.mean, self.state.cov)



    def correct(self, output, p=None):

        """
        Correct the state and state estimation error covariance using 
        measured system outputs.

        Parameters
        ----------
        p: ndarray, optional
            The system exogenous input.
        y: ndarray
            The measured system output.

        Examples
        --------
        >>> A = np.array([[0, 1], [0, 0]])
        >>> B = np.array([[0], [1]])
        >>> C = np.array([[1, 0]])
        >>> D = np.array([[0]])
        >>> model = syst.StateSpace(A, B, C, D, dt=0.02)
        >>> kf = filt.KalmanFilter(model)
        >>> kf.setInitialState(np.ones(2), 1e5*np.eye(2))
        >>> kf.setProcessNoise(1e-5*np.eye(2))
        >>> kf.setMeasurementNoise(1e-2*np.eye(1))
        >>> kf.getStatePrediction(np.array([2]))
        >>> kf.correctState(np.array([0.1]))
        >>> kf.state.mean
        array([ 0.99820359,  2.        ])
        >>> kf.state.cov
        array([[  1.99600798e-05,   0.00000000e+00],
               [  0.00000000e+00,   1.00000000e-05]])
        """

        C, D = self.model.C, self.model.D

        leftGain = self.state.cov.dot(C.T)
        leftRightGain = C.dot(self.state.cov)
        rightGain = leftRightGain.dot(C.T)+self.noise.output.cov
        self.gain = leftGain.dot(np.linalg.inv(rightGain))

        p = np.zeros(self.model.B.shape[1]) if p is None else p
        self.prediction = C.dot(self.state.mean)+D.dot(p)
        self.innovation.mean = output-self.prediction
        self.innovation.cov = rightGain

        self.correction.state.mean = self.gain.dot(self.innovation.mean)
        self.correction.state.cov = -self.gain.dot(leftRightGain)

        self.state.mean += self.correction.state.mean
        self.state.cov += self.correction.state.cov

        prediction = C.dot(self.state.mean)+D.dot(p)
        self.residual.mean = output-prediction
        self.residual.covariance = C.dot(self.state.cov).dot(C.T)
        self.residual.covariance += self.noise.output.cov

        self.prediction = prediction

        self.scorrectionSequence.append(self.scorrection.mean)
        self.innovationSequence.append(self.innovation.mean)

        for observer in self.correctionObservers:
            observer(self.state.mean, self.state.cov)


class UKF(Filter):

    """
    Unscented Kalman Filter (UKF) class for online state estimation of 
    discrete-time nonlinear systems [1]. Based on the Unscented Transformation 
    (UT), the UKF postulates that the probability distribution of the state 
    at each time instant is approximated by propagating a set of discrete 
    state-points, referred to as the sigma points, through the exact nonlinear 
    system and measurement functions. The posterior statistics are then 
    approximated by means of a weighted sample mean and covariance of the
    output sigma points [2,3].

    Parameters
    ----------
    model: ssm.Nonlinear
        The 'UnscentedKalmanFilter' class is instantiated with a nonlinear 
        state-space model. As such, the model must contain the following 
        methods:

            1. getState(state, input)
            2. getOutput(state, input)

        The model must additionally have the following attributes:

            1. stateSize:   The size of state vector.
            2. inputSize:   The number of inputs.
            3. outputSize:  The number of measured outputs.
    alpha: 
        Parameter of unscented transformation, which determines the spread of 
        sigma points around mean. Default value is 1.
    beta: 
        Parameter of unscented transformation related to the state 
        distribution. Default value is 0.
    kappa:
        Parameter of unscented transformation related to the spread of sigma 
        points around mean. Default value is 0.

    Attributes
    ----------
    model: ssm.Nonlinear
        The discrete-time state-space model describing the system dynamics.
    state: Base
        The state of the model which is a Base instance with attributes 
        mean and cov. By default, the state is initialized with a zero mean 
        and an indentity error covariance matrix.
    prediction: ndarray
        The predicted output.
    innovation: ndarray
        The difference between actual measurements and prior measurement 
        predictions, also referred to as pre-fit residual.
    correction: Base
        The state mean and state covariance correction.
    noise: Base
        The process and measurement noise terms, which are both instances of
        the Base class and initialized with a zero mean and unit covariance 
        matrix.

    Methods
    -------
    setInitialState(mean, covariance):
        Specify the initial value of the state estimate and the state
        estimation error.
    setProcessNoise(covariance):
        Specify the process noise covariance matrix.
    setMeasurementNoise(covariance):
        Specify the measurement noise covariance matrix.
    getParticles(mean, covariance):
        Generate particles (sigma points) according to the given statistics, 
        mean and covariance, of a random variable.
    getHyperparameters(particles):
        Calculate the hyperparameters, mean and covariance, of a random
        variable with given particles as the weighted sample mean and 
        covariance of the particles.
    predictState(input):
        Predict the state and state estimation error covariance at the next
        time step.
    correctState(output):
        Correct the state and state estimation error covariance using
        measured system outputs.

    References
    ----------
     .. [1] E. A. Wan, R. van der Merwe, "The Unscented Kalman Filter for 
        Nonlinear Estimation", IEEE Adaptive Systems for Signal Processing,
        Communications and Control Symposium, pp. 153-158, 2000.

     .. [2] S. J. Julier, J. K. Uhlmann, "Unscented filtering and nonlinear
        estimation", Proceedings of the IEEE, 92(3), pp. 401-422, 2004.

     .. [3] S. Mariani, A. Ghisi, "Unscented Kalman filtering for nonlinear
        structural dynamics", Nonlinear Dynamics, No. 49, pp. 131-150, 2007.

    """


    def __init__(self, model, alpha=1, beta=2, kappa=0):

        self._models = ['Linear', 'Nonlinear']
        super(UKF, self).__init__(model)

        s = self.model.stateSize
        o = self.model.outputSize

        self.alpha = alpha
        self.beta = beta
        self.kappa = 0 #  3-s
        self.lamda = self.alpha**2*(s+self.kappa)-s
        self.gamma = np.sqrt(s+self.lamda)

        p = 2*s+1
        self.particles = self.getParticles(self.state.mean, self.state.cov)
        self.outputParticles = np.zeros((o, p))

        weights = np.repeat(1/(2*(s+self.lamda)), s)
        weights = np.hstack((weights, self.lamda/(s+self.lamda), weights))
        self.weights = Base(mean=weights, cov=weights)
        self.weights.cov[s] += 1-self.alpha**2+self.beta



    def setParameters(self, **parameters):

        """
        Specify the unscented transformation parameters, alpha, beta and 
        kappa, which are related to the state distribution and the spread of 
        sigma points around the mean. 

        Parameters
        ----------
        **parameters: keyworded arguments
            The unscented transformation parameters, specified as follows

            ===========  ====================================================
            Parameter    Description
            ===========  ====================================================
            alpha        Determines the spread of sigma points around mean.
            beta         Related to the state distribution.
            kappa        Related to the spread of sigma points around mean.
            ===========  ====================================================

        Raises
        ------
        ValueError
            If an invalid parameter is specified.

        Examples
        --------
        >>> 
        >>> ukf.setParameters(alpha=1, beta=2, kappa=0)
        """

        for key in parameters.keys():
            if key not in ['alpha', 'beta', 'kappa']:
                raise ValueError('Invalid parameter name "{}".'.format(key))

        for key in parameters.keys():
            setattr(self, key, parameters[key])

        s = self.model.stateSize
        self.lamda = self.alpha**2*(s+self.kappa)-s
        self.gamma = np.sqrt(s+self.lamda)

        weights = np.repeat(1/(2*(s+self.lamda)), s)
        weights = np.hstack((weights, self.lamda/(s+self.lamda), weights))
        self.weights = Base(mean=weights, cov=weights)
        self.weights.cov[s] += 1-self.alpha**2+self.beta



    def getParticles(self, mean, covariance):

        """
        Generate particles (sigma points) according to the given statistics, 
        mean and covariance, of a random variable. The particles are 
        symmetrically distributed around the mean value.

        Parameters
        ----------
        mean: ndarray
            The mean value of the random variable.
        covariance: ndarray
            The covariance matrix of the random variable.

        Returns
        -------
        particles: ndarray
            The particle matrix of size (s x p), where s is the state size 
            and p is the number of particles equal to 2s+1.
        """

        mean = mean[np.newaxis].T
        distance = np.real(sp.linalg.sqrtm(covariance))*self.gamma

        if np.any(np.isnan(distance)):
            raise ValueError('Singular covariance matrix.')

        particles = np.hstack((mean-distance, mean, mean+distance))

        return particles



    def getHyperparameters(self, particles):

        """
        Calculate the hyperparameters, mean and covariance, of a random
        variable as the weighted sample mean and covariance of the particles.

        Parameters
        ----------
        particles: ndarray
            The sampled random variable of size (d x p), where d is the 
            variable dimension and p is the number of particles (samples).

        Returns
        -------
        mean: ndarray
            The mean value calculated as a weighted sample of the particles.
        covariance: ndarray
            The covariance matrix calculated as a weighted sample of the 
            particles.
        """

        s, p = particles.shape
        mean = np.sum(self.weights.mean*particles, 1)

        deviation = particles-mean[np.newaxis].T
        leftMatrix = (self.weights.cov*deviation).reshape((s, 1, p))
        rightMatrix = deviation.T.reshape((1, p, s))
        covariance = leftMatrix.dot(rightMatrix).squeeze()

        return mean, covariance



    def predict(self, p=None, *parameters):

        """
        Predict state and state estimation error covariance at the next time
        step.

        Parameters
        ----------
        p: ndarray, optional
            The system input at current time step.
        """

        self.time = next(self._timer)
        self.particles = self.getParticles(self.state.mean, self.state.cov)

        for k, state in enumerate(self.particles.T):
            self.particles[:, k] = self.model.getState(state, self.time, p, *parameters)

        self.state.mean, covariance = self.getHyperparameters(self.particles)
        self.state.cov = covariance+self.noise.state.cov
        self.state.cov = (self.state.cov+self.state.cov.T)/2




    def correct(self, output, p=None, *parameters):

        """
        Correct the system state and its associated error covariance matrix 
        using measured system outputs. 

        Parameters
        ----------
        output: ndarray
            The measured system output.
        p: ndarray
            The system input at current time step.
        """

        func = self.model.getOutput

        for k, state in enumerate(self.particles.T):
            self.outputParticles[:, k] = func(state, self.time, p, *parameters)


        outputMean, covariance = self.getHyperparameters(self.outputParticles)
        outputCovariance = covariance+self.noise.output.cov
        (s, p), o = self.particles.shape, self.outputParticles.shape[0]

        stateDeviation = self.particles-self.state.mean[np.newaxis].T
        leftMatrix = (self.weights.cov*stateDeviation).reshape((s, 1, p))
        outputDeviation = self.outputParticles-outputMean[np.newaxis].T
        rightMatrix = outputDeviation.T.reshape((1, p, o))

        crossCovariance = leftMatrix.dot(rightMatrix).reshape(s, o)
        rightGain = np.linalg.inv(outputCovariance)
        gain = crossCovariance.dot(rightGain)

        self.prediction = outputMean
        self.innovation.mean = output-outputMean
        self.innovation.cov = rightGain
        self.correction.state.mean = gain.dot(self.innovation.mean)
        self.correction.state.cov = gain.dot(outputCovariance).dot(gain.T)

        self.state.mean += self.correction.state.mean
        self.state.cov -= self.correction.state.cov
        self.state.cov = (self.state.cov+self.state.cov.T)/2



class PF(Filter):

    """
    Particle Filter (PF) class for particle-based sequential Bayesian
    inference on discrete-time nonlinear systems.

    Parameters
    ----------
    model: ...
        ...
    N: float, positive
        The number of particles.
    likelihood: {'Normal'}
        The likelihood function.
    """

    def __init__(self, model, N=None, likelihood='Normal'):
        
        self._models = ['Linear', 'Nonlinear']
        super(PF, self).__init__(model)


        self.setNumberOfParticles(N)
        self.setResamplingStrategy('default')
        self.setResamplingThreshold(2)

        self.setLikelihood(likelihood)
        self.weights = np.repeat(1/N, N)


    def normalizeWeights(self):

        """
        Normalize the importance weights so that their sum is equal to one.
        """

        weight = np.sum(self.weights)

        if weight > 0:
            self.weights /= weight
        else:
            sys.stdout.write('All particle weights are zero.\n')
            n = self.weights.size
            self.weights = np.ones(n)/n


    def resetWeights(self):

        """
        Reset weights to uniform values.
        """

        self.weights = np.repeat(1/self.N, self.N)


    def setNumberOfParticles(self, N):

        """
        Specify the number of particles.

        Parameters
        ----------
        N: int, positive
            The number of particles.

        Raises
        ------
        TypeError
            If the number of particles is not an integer.
        ValueError
            If the number of particles is non-positive.
        """

        if type(N) not in [int]:
            raise TypeError('Non-integer number of particles.')

        if N < 1:
            raise ValueError('Non-positive number of particles.')

        self.N = N


    def setLikelihood(self, likelihood='Normal'):

        """
        Specify the likelihood function.

        Parameters
        ----------
        likelihood: {'Normal', 'Lognormal', 'Exponential'}
            The name of the likelihood function.

        Raises
        ------
        TypeError
            If an invalid likelihood function is specified.
        """

        if likelihood is 'Normal':
            def getLikelihood(mean, cov):
                num = np.exp(-0.5*mean.T.dot(np.linalg.inv(cov)).dot(mean))
                denom = np.sqrt((2*np.pi)**mean.size*np.linalg.det(cov))
                return num/denom

        elif likelihood is 'Lognormal':
            def getLikelihood(mean, cov):
                pass

        elif likelihood is 'Exponential':
            def getLikelihood(mean, cov):
                pass

        else:
            raise TypeError('Invalid likelihood function')

        self._getLikelihood = getLikelihood


    def setResamplingStrategy(self, strategy):
        
        """
        Specify the resampling strategy.

        Parameters
        ----------
        strategy: str
            The name of resampling strategy.

        Raises
        ------
        ValueError
            If an invalid resampling strategy is specified.
        """

        if strategy == 'default':
            def resample():
                cumsum = np.cumsum(self.weights)

                for j in range(self.N):
                    index = np.where(cumsum >= np.random.rand(1))[0][0]
                    self.particles[:, j] = self.particles[:, index]

        else:
            raise ValueError('Invalid remsampling strategy.')

        self._resample = resample


    def setResamplingThreshold(self, threshold):

        """
        Description

        Parameters
        ----------
        threshold: float, positive
            The degeneracy threshold, whose exceedance triggers the 
            resampling of particles.
        """
        
        self.threshold = threshold


    def getParticles(self, mean, covariance):
        
        """
        Generate particles according to the given statistics, mean and
        covariance, of a random variable.

        Parameters
        ----------
        mean: ndarray
            The mean value of the random variable.
        covariance: ndarray
            The covariance matrix of the random variable.

        Returns
        -------
        particles: ndarray
            The particle matrix of size (d x p), where d is the dimension of
            the variable and p is the number of particles.
        """

        mean = mean[np.newaxis].T
        samples = np.random.randn(mean.size, self.N)
        distance = np.linalg.cholesky(covariance).dot(samples)

        if np.any(np.isnan(distance)):
            raise ValueError('Singular covariance matrix.')

        particles = mean+distance

        return particles


    def getHyperparameters(self, particles, weights):

        """
        Calculate the hyperparameters, mean and covariance, of a random
        variable as the weighted sample mean and covariance of the particles.

        Parameters
        ----------
        particles: ndarray
            The sampled random variable of size (d x p), where d is the 
            variable dimension and p is the number of particles.
        weights: ndarray
            The weight matrix.

        Returns
        -------
        mean: ndarray
            The mean value calculated as a weighted sample of the particles.
        covariance: ndarray
            The covariance matrix calculated as a weighted sample of the 
            particles.
        """

        s, p = particles.shape
        mean = np.sum(self.weights*particles, 1)

        deviation = particles-mean[np.newaxis].T
        leftMatrix = (self.weights*deviation).reshape((s, 1, p))
        rightMatrix = deviation.T.reshape((1, p, s))
        covariance = leftMatrix.dot(rightMatrix).squeeze()

        return mean, covariance



    def resample(self):
        
        """
        Resample particles and reset weights to uniform values.
        """

        if 1/np.sum(self.weights**2) < self.threshold:
            self._resample()
            self.resetWeights()




    def predict(self, p=None, *parameters):

        """
        Predict the state and state estimation error covariance at the next
        time step.

        Parameters
        ----------
        p: ndarray
            The exogenous system input.
        """

        self.time = next(self._timer)
        self.particles = self.getParticles(self.state.mean, self.state.cov)

        for k, state in enumerate(self.particles.T):
            self.particles[:, k] = self.model.getState(state, self.time, p, *parameters)
        
        mean, covariance = self.noise.state.mean, self.noise.state.cov
        noiseParticles = self.getParticles(mean, covariance)
        particles = self.particles+noiseParticles

        mean, covariance = self.getHyperparameters(particles, self.weights)
        self.state.mean, self.state.cov = mean, covariance
        self.state.cov = (self.state.cov+self.state.cov.T)/2


    def correct(self, output, p=None, *parameters):
        
        """

        Parameters
        ----------

        """

        R = self.noise.output.cov

        for k, state in enumerate(self.particles.T):
            prediction = self.model.getOutput(state, self.time, p, *parameters)

            likelihood = self._getLikelihood(output-prediction, R)+1e-99
            self.weights[k] *= likelihood


        self.normalizeWeights()
        self.resample()

        mean, covariance = self.getHyperparameters(self.particles, self.weights)
        self.state.mean, self.state.cov = mean, covariance
        self.state.cov = (self.state.cov+self.state.cov.T)/2


class MPF(PF):

    """
    Parameters
    ----------
    model: ...
        ...
    N: int, positive
        The number of particles.
    pr: float, optional
        Probability of replacement.
    pm: float, optional
        Probability of mutation.
    likelihood: {'Normal'}
        The likelihood function.
    """

    def __init__(self, model, N=None, likelihood='Normal', **parameters):

        super(MPF, self).__init__(model, N, likelihood)

        self.mutation = Base(pr=1, pm=1, states=[])
        self.setMutationParameters(**parameters)


    def setMutationParameters(self, **parameters):

        """
        Specify the mutation parameters.

        Parameters
        ----------
        **parameters: keyworded-arguments
            ...
        pr: float, [0, 1]
            Probability of replacement.
        pm: float
            Probability of mutation.
        index: list, ndarray
            The indices of state components to be mutated.
        """

        for key, parameter in parameters.items():
            if key not in ['pr', 'pm', 'states', 'radius']:
                raise TypeError('Invalid mutation parameter.')

            if key in ['pr', 'pm']:
                if parameter > 1 or parameter < 0:
                    raise ValueError('Invalid "{}" value.'.format(key))
            elif key == 'states':
                if not isinstance(parameter, (list, tuple, np.ndarray)):
                    raise TypeError('Invalid "states" instance.')

                if max(parameter) > self.model.stateSize-1 or min(parameter) < 0:
                    raise ValueError('Invalid "state" values.')
            else:
                pass

            setattr(self.mutation, key, parameter)


    def setResamplingStrategy(self, strategy):
        
        """
        Specify the resampling strategy.

        Parameters
        ----------
        strategy: str
            The name of resampling strategy.

        Raises
        ------
        ValueError
            If an invalid resampling strategy is specified.
        """

        if strategy == 'default':

            def resample(prior):

                """
                Description
                """

                resampledIndex = []
                cumsum = np.cumsum(self.weights)

                for j in range(self.N):
                    index = np.where(cumsum >= np.random.rand(1))[0][0]
                    self.particles[:, j] = self.particles[:, index]
                    resampledIndex.append(index)

                resampledIndex = np.sort(resampledIndex)


                random = np.random.rand(len(self.mutation.states))
                p2replace = np.where(random < self.mutation.pr)[0]

                self.particles[:, p2replace] = prior[np.newaxis].T

                repetitions = resampledIndex[:-1] == resampledIndex[1:]
                p2mutate = resampledIndex[np.where(repetitions)[0]+1]


                rows, columns = len(self.mutation.states), len(p2mutate)
                span = self.mutation.radius.flatten()
                mutations = (np.random.rand(rows, columns)-0.5)*span
                mutations *= np.random.rand(rows, columns) < self.mutation.pm
                mutations = mutations.flatten()


                particles = self.particles
                self.particles[self.mutation.states, p2mutate] *= 1+mutations


                denominator = np.sqrt(np.sum((self.particles-particles)**2, 0)/np.sum(particles**2, 0)+1)
                self.weights /= denominator

        else:
            raise ValueError('Invalid remsampling strategy.')

        self._resample = resample



    def resample(self, *parameters):
        
        """
        Resample particles and normalize weights.
        """

        if 1/np.sum(self.weights**2) < self.threshold:
            self._resample(*parameters)
            self.normalizeWeights()



    def correct(self, output, p=None, *parameters):
        
        """

        Parameters
        ----------

        """

        R = self.noise.output.cov

        for k, state in enumerate(self.particles.T):
            prediction = self.model.getOutput(state, self.time, p, *parameters)

            likelihood = self._getLikelihood(output-prediction, R)+1e-99
            self.weights[k] *= likelihood


        prior = self.model.getState(self.state.mean, self.time, p, *parameters)

        self.normalizeWeights()
        self.resample(prior)

        mean, covariance = self.getHyperparameters(self.particles, self.weights)
        self.state.mean, self.state.cov = mean, covariance
        self.state.cov = (self.state.cov+self.state.cov)/2



class SPPF(PF):

    def __init__(self, model, N=None, likelihood='Normal'):
        
        super(SPPF, self).__init__(model, N, likelihood)

        self.filters = [UKF(model) for particle in range(self.N)]

        mean = np.zeros(self.model.stateSize)
        covariance = np.eye(self.model.stateSize)*1e-10

        for j in range(self.N):
            self.filters[j].setInitialState(mean, covariance)


    def setProcessNoise(self, covariance):

        """
        """
        
        super(SPPF, self).setProcessNoise(covariance)

        for item in self.filters:
            item.setProcessNoise(covariance)


    def setMeasurementNoise(self, covariance):
        
        super(SPPF, self).setMeasurementNoise(covariance)

        for item in self.filters:
            item.setMeasurementNoise(covariance)


    def predict(self, p=None, *parameters):

        """
        Predict the state and state estimation error covariance at the next
        time step.

        Parameters
        ----------
        p:ndarray
            The exogenous system input.
        """

        self.time = next(self._timer)
        print(self.state.cov)
        self.particles = self.getParticles(self.state.mean, self.state.cov)

        for k, state in enumerate(self.particles.T):
            self.filters[k].state.mean = state
            self.filters[k].state.cov = self.state.cov
            self.filters[k].predict(p, *parameters)
            self.particles[:, k] = self.filters[k].state.mean

        mean, covariance = self.noise.state.mean, self.noise.state.cov
        noiseParticles = self.getParticles(mean, covariance)
        particles = self.particles+noiseParticles

        mean, covariance = self.getHyperparameters(particles, self.weights)
        self.state.mean, self.state.cov = mean, covariance
        self.state.cov = (self.state.cov+self.state.cov)/2


    def correct(self, output, p=None, *parameters):

        """
        Description

        Parameters
        ----------
        output: ndarray
            ...
        p: ndarray
            ...
        """

        R = self.noise.output.cov

        for k, state in enumerate(self.particles.T):
            self.filters[k].correct(output, p, *parameters)

            # self.filters[k].state.cov = np.diag(np.abs(np.diag(self.filters[k].state.cov)))
            # print('Correction, covariance:', self.filters[k].state.cov)

            m, c = self.filters[k].state.mean, self.filters[k].state.cov
            difference = c.dot(np.random.rand(self.model.stateSize))
            m += difference
            # particle = np.random.multivariate_normal(m, (c+c.T)/2)

            # 1. Calculate transition prior
            prior = 1

            prediction = self.model.getOutput(state, self.time, p)
            likelihood = self._getLikelihood(output-prediction, R)+1e-99

            # 2. Calculate proposal
            proposal = 1

            # 3. Update weights
            
            self.weights[k] *= likelihood*prior/proposal


        self.normalizeWeights()

        mean, covariance = self.getHyperparameters(self.particles, self.weights)
        self.state.mean, self.state.cov = mean, covariance
        self.state.cov = (self.state.cov+self.state.cov)/2

        self.resample()


        ii, jj = np.where(self.state.cov > 1e20)
        self.state.cov[ii, jj] = 1e20


class RBPF(PF):

    """
    Parameters
    ----------
    filters: ...
        ...
    delta: float, (0, 1]
        ...
    m: int, positive
        The number of marginalized states
    """

    def __init__(self, filters, delta, m):
        
        if not isinstance(filters, list):
            raise TypeError('filters must be a list instance.')

        self.__types = (KF, UKF, PF, MPF)

        for filt in filters:

            if type(filt) not in self.__types:
                error = 'invalid type of filter.'
                raise TypeError(error)

            if filt.model.outputSize != filters[0].model.outputSize:
                error = 'inconsistent output dimensions among filters.'
                raise ValueError(error)

        s = filters[0].model.stateSize
        o = filters[0].model.outputSize

        self.N = len(filters)
        self.filters = filters
        self.alpha = (3*delta-1)/(2*delta)

        self._timer = it.count(0, self.filters[0].model.dt)
        self.time = next(self._timer)

        self.weights = np.repeat(1/self.N, self.N)
        mean, covariance = np.zeros(m), 1e-5*np.eye(m)
        self.state = Base(mean=mean, cov=covariance)
        self.particles = self.getParticles(mean, covariance)

        self.marginalized = Base(mean=np.zeros(s), cov=1e-2*np.eye(s))
        #self.state = Base(mean=np.zeros(s), cov=1e-2*np.eye(s))
        self.prediction = np.zeros(o)
        self.innovation = Base(mean=np.zeros(o), cov=1e-2*np.eye(o))

        self.likelihood = 'Normal'
        self.setLikelihood(self.likelihood)

        self.setResamplingStrategy('default')
        self.setResamplingThreshold(2)



    def setLatentState(self, mean, covariance):

        self.state.mean = mean
        self.state.cov = covariance

        self.particles = self.getParticles(mean, covariance)



    def predict(self, p=None):

        """
        """

        # self.state.mean = np.zeros_like(self.state.mean)
        # self.state.cov = np.zeros_like(self.state.cov)
        # self.prediction = np.zeros_like(self.prediction)

        self.marginalized.mean = np.zeros_like(self.marginalized.mean)
        self.marginalized.cov = np.zeros_like(self.marginalized.cov)
        self.prediction = np.zeros_like(self.prediction)

        for k, filt in enumerate(self.filters):
            filt.predict(p, self.particles[:, k])

            # self.state.mean += weight*filt.state.mean
            # self.state.cov += weight*filt.state.cov
            # self.prediction += weight*filt.prediction

            self.marginalized.mean += self.weights[k]*filt.state.mean
            self.marginalized.cov += self.weights[k]*filt.state.cov
            self.prediction += self.weights[k]*filt.prediction



    def correct(self, output, p=None):

        """
        """

        R = self.filters[0].noise.output.cov
        particle_mean = np.mean(self.particles, 1)
        particle_variance = (1-self.alpha**2)*self.state.cov#ariance

        for k, filt in enumerate(self.filters):

            self.particles[:, k] *= self.alpha
            self.particles[:, k] += (1-self.alpha)*particle_mean


            prediction = filt.model.getOutput(filt.state.mean, self.time, p, self.particles[:, k])
            likelihood = self._getLikelihood(output-prediction, R)+1e-99

            self.weights[k] *= likelihood

        self.normalizeWeights()


        for k, filt in enumerate(self.filters):
            filt.correct(output, p, self.particles[:, k])

        self.resample()

        for filt, weight in zip(self.filters, self.weights):

            self.marginalized.mean += weight*filt.state.mean
            self.marginalized.cov += weight*filt.state.cov

        self.marginalized.cov = (self.marginalized.cov+self.marginalized.cov.T)/2


        mean, cov = self.getHyperparameters(self.particles, self.weights)
        self.state.mean = mean
        self.state.cov = np.array([[cov]])



class DKFUKF(UKF):

    def __init__(self, model, D, alpha=1, beta=2, kappa=0):

        super(DKFUKF, self).__init__(model, )

        self.UKF = UKF(model, alpha, beta, kappa)
        self.D = D

        mean = np.zeros(model.inputSize)
        covariance = np.eye(model.inputSize)
        self.input = Base(mean=mean, cov=covariance)

        covariance = 1e-5*np.eye(model.inputSize)
        self.noise.input = Base(mean=mean, cov=covariance)

        covariance = np.zeros((model.inputSize, model.inputSize))
        self.correction.input = Base(mean=mean, cov=covariance)


    def setInitialInput(self, mean, covariance):

        error = 'Invalid shape of initial input {}.'

        if mean.shape != self.input.mean.shape:
            raise TypeError(error.format('vector'))

        if covariance.shape != self.input.cov.shape:
            raise TypeError(error.format('covariance matrix'))

        self.input.mean = mean
        self.input.cov = covariance


    def setInputNoise(self, covariance):
        
        if covariance.shape != self.noise.input.cov.shape:
            raise TypeError('Invalid shape of covariance matrix.')

        self.noise.input.cov = covariance
        

    def predict(self):
        
        self.time = next(self._timer)
        self.input.cov += self.noise.input.cov

        super(DKFUKF, self).predict(self.input.mean)


    def correct(self, output):
        
        leftGain = self.input.cov.dot(self.D.T)
        leftRightGain = self.D.dot(self.input.cov)
        rightGain = leftRightGain.dot(self.D.T)+self.noise.output.cov
        gain = leftGain.dot(np.linalg.inv(rightGain))

        prediction = self.model.getOutput(self.state.mean, self.time, self.input.mean)
        self.correction.input.mean = gain.dot(output-prediction)
        self.correction.input.cov = -gain.dot(leftRightGain)

        self.input.mean += self.correction.input.mean
        self.input.cov += self.correction.input.cov

        super(DKFUKF, self).correct(output, self.input.mean)


