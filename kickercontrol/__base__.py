"""
This module defines the base classes for different signal types used in the SignalGenerator GUI. It provides 
templates for a variety of signal shapes, each extending a common base class, BaseSignal. These signal classes
can be instantiated and used to generate data corresponding to their signal type.

The main logic of the class structure follows an object-oriented approach, where each signal type is encapsulated
in its own class that inherits from BaseSignal. BaseSignal defines the general structure and functionality that
all signals share, such as methods to provide variable mappings, default values, and LaTeX representations of 
the signal.

To add a new signal class:
1. Create a subclass that inherits from BaseSignal.
2. Define the variables used by the signal, followiWng the dictionary format in `self.variables`.
3. Implement the `generate()` method to produce the desired signal using input parameters.
4. Update the dict self.oscillators in the __init__ of singal.SignalGenerator
5. Optionally `get_latex_expression()` to return a LaTeX representation of the signal equation.

Each derived signal class should provide:
- Variable definitions (`self.variables`) to describe the parameters.
- A `generate()` method to compute the signal for a given time domain.
- A `get_latex_expression()` method to provide a LaTeX representation for gui purposes.
"""

import numpy as np
from scipy import signal
from scipy import special

from kickercontrol.timing import get_bunch_pattern
class BaseSignal:
    """
    Base class for all signal types. Provides methods to get variable mapping,
    default values, and LaTeX representation of the signal.
    """
    def __init__(self):
        self.variables = {}
        self.default_values = {}

    def get_variable_mapping(self):
        """
        Returns the mapping of variable names to their descriptions.
        """
        return self.variables

    def get_default_values(self):
        """
        Returns the default values for all variables.
        """
        return self.default_values

    def get_latex_expression(self):
        """
        Returns the LaTeX representation of the signal.
        """
        return ""
    

    @property
    def V1(self):
        return get_bunch_pattern()
    
class LineSignal(BaseSignal):
    """
    Generates a constant line signal within a specified time domain.
    """
    def __init__(self):
        super().__init__()
        self.variables = {"V0": "start", "V1": "end", "V2": "offset"}
        self.default_values = {"V0": 795, "V1": 1200, "V2": 0}

    def generate(self, t, V0, V1, V2,**kwargs):
        """
        Generates a line signal that is active between V0 and V1 and equal to V2 otherwise.
        """
        return np.where((t >= V0) & (t <= V1), V2, 0)

    def get_latex_expression(self):
        return "V_2 \\text{ for } t \\in [V_0, V_1]"

class SinSignal(BaseSignal):
    """
    Generates a sinusoidal signal within a specified time domain.
    """
    def __init__(self):
        super().__init__()
        self.variables = {"V0": "start", "V1": "end", "V2": "offset", "V3": "amplitude", "V4": "frequency", "V5": "phase"}
        self.default_values = {"V0": 795, "V1": 1200, "V2":  0, "V3": 1, "V4": 1/405, "V5": 0}


    def generate(self, t, V0, V1, V2, V3, V4, V5,**kwargs):
        """
        Generates a sinusoidal signal within the specified domain.
        """
        signal_values = V3 * np.sin(2 * np.pi * V4 * -(V0-t)-V5*t) + V2
        return np.where((t >= V0) & (t <= V1), signal_values, V2)

    def get_latex_expression(self):
        return "V_3 \\sin(2 \\pi V_4 t) + V_2, \\text{for } t \\in [V_0, V_1]"

class CosSignal(BaseSignal):
    """
    Generates a cosine signal within a specified time domain.
    """
    def __init__(self):
        super().__init__()
        self.variables = {"V0": "start", "V1": "end", "V2": "offset", "V3": "amplitude", "V4": "frequency", "V5": "phase"}
        self.default_values = {"V0": 795, "V1": 1200, "V2": 0, "V3": 1, "V4": 1/405, "V5": 0}

    def generate(self, t, V0, V1, V2, V3, V4,V5,**kwargs):
        """
        Generates a cosine signal within the specified domain.
        """
        signal_values = V3 * np.cos(2 * np.pi * V4 * -(V0-t)-V5*t) + V2
        return np.where((t >= V0) & (t <= V1), signal_values, V2)

    def get_latex_expression(self):
        return "V_3 \\cos(2 \\pi V_4 t) + V_2, \\text{for } t \\in [V_0, V_1]"

class SquareSignal(BaseSignal):
    """
    Generates a square wave signal within a specified time domain.
    """
    def __init__(self):
        super().__init__()
        self.variables = {"V0": "start", "V1": "end", "V2": "offset", "V3": "amplitude", "V4": "frequency", "V5": "duty"}
        self.default_values = {"V0": 795, "V1": 1200, "V2": 0, "V3": 1, "V4": 1, "V5": 0.5}

    def generate(self, t, V0, V1, V2, V3, V4, V5,**kwargs):
        """
        Generates a square wave signal within the specified domain.
        """
        signal_values = V3 * signal.square(2 * np.pi * V4 * -(V0-t), duty=V5) + V2
        return np.where((t >= V0) & (t <= V1), signal_values, V2)

    def get_latex_expression(self):
        return "V_3 \\text{square}(2 \\pi V_4 t, \\text{duty}=V_5) + V_2, \\text{for } t \\in [V_0, V_1]"

class TriangleSignal(BaseSignal):
    """
    Generates a triangle wave signal within a specified time domain.
    """
    def __init__(self):
        super().__init__()
        self.variables = {"V0": "start", "V1": "end", "V2": "offset", "V3": "amplitude", "V4": "frequency"}
        self.default_values = {"V0": 795, "V1": 1200, "V2": 0, "V3": 1, "V4": 1}

    def generate(self, t, V0, V1, V2, V3, V4,**kwargs):
        """
        Generates a triangle wave signal within the specified domain.
        """
        signal_values = V3 * signal.sawtooth(2 * np.pi * V4 * -(V0-t), width=0.5) + V2
        return np.where((t >= V0) & (t <= V1), signal_values, V2)

    def get_latex_expression(self):
        return "V_3 \\text{triangle}(2 \\pi V_4 t) + V_2, \\text{for } t \\in [V_0, V_1]"

class RampSignal(BaseSignal):
    """
    Generates a ramp (linear) signal within a specified time domain.
    """
    def __init__(self):
        super().__init__()
        self.variables = {"V0": "start", "V1": "end", "V2": "offset", "V3": "start_value","V4": "end_value"}
        self.default_values = {"V0": 795, "V1": 1200, "V2": 0, "V3": 0, "V4": 1}

    def generate(self, t, V0, V1, V2, V3, V4, **kwargs):
        """
        Generates a ramp signal within the specified domain.
        """
        ind_min = np.argmin(abs(t-V0))
        ind_max = np.argmin(abs(t-V1))

        ramp_signal = np.linspace(V3, V4, ind_max-ind_min)
        signal = np.ones_like(t)*V2
        signal[ind_min:ind_max] = ramp_signal
        return signal

    def get_latex_expression(self):
        return "\\text{linspace}(V_2, V_2 + (V_1 - V_0), t) + V_2"

class GaussianSignal(BaseSignal):
    """
    Generates a Gaussian (bell curve) signal within a specified time domain.
    """
    def __init__(self):
        super().__init__()
        self.variables = {"V0": "start", "V1": "end", "V2": "offset", "V3": "amplitude", "V4": "mean", "V5": "std_dev"}
        self.default_values = {"V0": 795, "V1": 1200, "V2": 0, "V3": 1, "V4": 1000, "V5": 1}

    def generate(self, t, V0, V1, V2, V3, V4, V5,**kwargs):
        """
        Generates a Gaussian signal within the specified domain.
        """
        signal_values = V3 * np.exp(-((-(V0-t) - V4) ** 2) / (2 * V5 ** 2)) + V2
        return np.where((t >= V0) & (t <= V1), signal_values, V2)

    def get_latex_expression(self):
        return "V_3 e^{-(t - V_4)^2 / (2 V_5^2)} + V_2, \\text{for } t \\in [V_0, V_1]"

class ExponentialDecaySignal(BaseSignal):
    """
    Generates an exponentially decaying signal within a specified time domain.
    """
    def __init__(self):
        super().__init__()
        self.variables = {"V0": "start", "V1": "end", "V2": "offset", "V3": "amplitude", "V4": "decay_rate"}
        self.default_values = {"V0": 795, "V1": 1200, "V2": 0, "V3": 1, "V4": 0.5}

    def generate(self, t, V0, V1, V2, V3, V4,**kwargs):
        """
        Generates an exponentially decaying signal within the specified domain.
        """
        signal_values = V3 * np.exp(-V4 * -(V0-t)) + V2
        return np.where((t >= V0) & (t <= V1), signal_values, V2)

    def get_latex_expression(self):
        return "V_3 e^{-V_4 t} + V_2, \\text{for } t \\in [V_0, V_1]"

class StepWithDecaySignal(BaseSignal):
    """
    Generates a step function with exponential decay within a specified time domain.
    """
    def __init__(self):
        super().__init__()
        self.variables = {"V0": "start", "V1": "end", "V2": "offset", "V3": "amplitude", "V4": "step_time", "V5": "decay_rate"}
        self.default_values = {"V0": 795, "V1": 1200, "V2":  0, "V3": 1, "V4": 2, "V5": 0.5}

    def generate(self, t, V0, V1, V2, V3, V4, V5,**kwargs):
        """
        Generates a step function with exponential decay within the specified domain.
        """
        step_signal = np.heaviside(-(V0-t) - V4, 1)
        decay_signal = np.exp(-V5 * (-(V0-t) - V4)) * step_signal
        signal_values = V3 * decay_signal + V2
        return np.where((t >= V0) & (t <= V1), signal_values, V2)

    def get_latex_expression(self):
        return "V_3 \\text{Heaviside}(t - V_4) e^{-V_5 (t - V_4)} + V_2, \\text{for } t \\in [V_0, V_1]"

class SpiralScanCosSignal(BaseSignal):
    """
    Generates the cosine component of a spiral scan signal within a specified time domain.
    """
    def __init__(self):
        super().__init__()
        self.variables = {"V0": "start", "V1": "end", "V2": "offset", "V3": "amplitude", "V4": "frequency", "V5": "radius_growth_rate"}
        self.default_values = {"V0": 795, "V1": 1200, "V2":  0, "V3": 1, "V4": 0.2, "V5": 0.1}

    def generate(self, t, V0, V1, V2, V3, V4, V5,**kwargs):
        """
        Generates the cosine component of a spiral scan within the specified domain.
        """
        signal_values = (V3 + V5 * -(V0-t)) * np.cos(2 * np.pi * V4 * -(V0-t)) + V2
        return np.where((t >= V0) & (t <= V1), signal_values, V2)

    def get_latex_expression(self):
        return "(V_3 + V_5 t) \\cos(2 \\pi V_4 t) + V_2, \\text{for } t \\in [V_0, V_1]"

class SpiralScanSinSignal(BaseSignal):
    """
    Generates the sine component of a spiral scan signal within a specified time domain.
    """
    def __init__(self):
        super().__init__()
        self.variables = {"V0": "start", "V1": "end", "V2": "offset", "V3": "amplitude", "V4": "frequency", "V5": "radius_growth_rate"}
        self.default_values = {"V0": 795, "V1": 1200, "V2": 0, "V3": 1, "V4": 0.002, "V5": 0.1}

    def generate(self, t, V0, V1, V2, V3, V4, V5,**kwargs):
        """
        Generates the sine component of a spiral scan within the specified domain.
        """
        signal_values = (V3 + V5 * -(V0-t)) * np.sin(2 * np.pi * V4 * -(V0-t)) + V2
        return np.where((t >= V0) & (t <= V1), signal_values, V2)

    def get_latex_expression(self):
        return "(V_3 + V_5 t) \\sin(2 \\pi V_4 t) + V_2, \\text{for } t \\in [V_0, V_1]"

class CustomExpressionSignal(BaseSignal):
    """
    Generates a custom signal based on a user-provided NumPy expression within a specified time domain.
    """
    def __init__(self, expression):
        super().__init__()
        self.expression = expression
        self.variables = {"V0": "start", "V1": "end", "V2": "offset", "V3": "custom_param1", "V4": "custom_param2"}
        self.default_values = {"V0": 795, "V1": 1200, "V2": 0, "V3": 1, "V4": 1}

    def generate(self, t, **kwargs):
        """
        Generates a custom signal based on the provided expression within the specified domain.
        """
        expr = self.expression
        for key, value in kwargs.items():
            expr = expr.replace(key, str(value))
        evaluated_expr = eval(expr, {"t": t, "np": np})
        return np.where((t >= kwargs.get("V0", 0)) & (t <= kwargs.get("V1", np.inf)), evaluated_expr, kwargs.get("V2", 0))

    def get_latex_expression(self):
        return self.expression
