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
2. Define the variables used by the signal, following the dictionary format in `self.variables`.
3. Implement the `generate()` method to produce the desired signal using input parameters.
4. Optionally override `get_latex_expression()` to return a LaTeX representation of the signal equation.

Each derived signal class should provide:
- Variable definitions (`self.variables`) to describe the parameters.
- A `generate()` method to compute the signal for a given time domain.
- A `get_latex_expression()` method to provide a LaTeX representation for documentation purposes.
"""

import numpy as np
from scipy import signal
from scipy import special

class BaseSignal:
    """
    Base class for all signal types. Provides methods to get variable mapping,
    default values, and LaTeX representation of the signal.
    """
    def __init__(self):
        self.variables = {}

    def get_variable_mapping(self):
        """
        Returns the mapping of variable names to their descriptions.
        """
        return self.variables

    def get_default_values(self):
        """
        Returns the default values for all variables.
        """
        return {k: (None if v is None else v) for k, v in self.variables.items()}

    def get_latex_expression(self):
        """
        Returns the LaTeX representation of the signal.
        """
        return ""

class LineSignal(BaseSignal):
    """
    Generates a constant line signal within a specified time domain.

    Parameters:
    - V0 : float
        Start time of the signal's active domain.
    - V1 : float
        End time of the signal's active domain.
    - V2 : float
        Offset value outside the specified domain.

    The signal value within [V0, V1] is V2, while outside this range, it equals the offset.
    """
    def __init__(self):
        super().__init__()
        self.variables = {"V0": "start", "V1": "end", "V2": "offset"}

    def generate(self, t, V0, V1, V2):
        """
        Generates a line signal that is active between V0 and V1 and equal to V2 otherwise.

        Parameters:
        t : array-like
            Time array.
        V0 : float
            Start time of the signal's active domain.
        V1 : float
            End time of the signal's active domain.
        V2 : float
            Offset value.

        Returns:
        array-like
            Line signal values.
        """
        return np.where((t >= V0) & (t <= V1), V2, V2)

    def get_latex_expression(self):
        return "V_2 \\text{ for } t \\in [V_0, V_1]"

class SinSignal(BaseSignal):
    """
    Generates a sinusoidal signal within a specified time domain.

    Parameters:
    - V0 : float
        Start time of the signal's active domain.
    - V1 : float
        End time of the signal's active domain.
    - V2 : float
        Offset value outside the specified domain.
    - V3 : float
        Amplitude of the sine wave.
    - V4 : float
        Frequency of the sine wave.

    The signal value within [V0, V1] follows the sine wave expression, while outside this range, it equals the offset.
    """
    def __init__(self):
        super().__init__()
        self.variables = {"V0": "start", "V1": "end", "V2": "offset", "V3": "amplitude", "V4": "frequency"}

    def generate(self, t, V0, V1, V2, V3, V4):
        """
        Generates a sinusoidal signal within the specified domain.

        Parameters:
        t : array-like
            Time array.
        V0 : float
            Start time of the signal's active domain.
        V1 : float
            End time of the signal's active domain.
        V2 : float
            Offset value.
        V3 : float
            Amplitude of the sine wave.
        V4 : float
            Frequency of the sine wave.

        Returns:
        array-like
            Sinusoidal signal values.
        """
        signal_values = V3 * np.sin(2 * np.pi * V4 * t) + V2
        return np.where((t >= V0) & (t <= V1), signal_values, V2)

    def get_latex_expression(self):
        return "V_3 \\sin(2 \\pi V_4 t) + V_2, \\text{for } t \\in [V_0, V_1]"

class CosSignal(BaseSignal):
    """
    Generates a cosine signal within a specified time domain.

    Parameters:
    - V0 : float
        Start time of the signal's active domain.
    - V1 : float
        End time of the signal's active domain.
    - V2 : float
        Offset value outside the specified domain.
    - V3 : float
        Amplitude of the cosine wave.
    - V4 : float
        Frequency of the cosine wave.

    The signal value within [V0, V1] follows the cosine wave expression, while outside this range, it equals the offset.
    """
    def __init__(self):
        super().__init__()
        self.variables = {"V0": "start", "V1": "end", "V2": "offset", "V3": "amplitude", "V4": "frequency"}

    def generate(self, t, V0, V1, V2, V3, V4):
        """
        Generates a cosine signal within the specified domain.

        Parameters:
        t : array-like
            Time array.
        V0 : float
            Start time of the signal's active domain.
        V1 : float
            End time of the signal's active domain.
        V2 : float
            Offset value.
        V3 : float
            Amplitude of the cosine wave.
        V4 : float
            Frequency of the cosine wave.

        Returns:
        array-like
            Cosine signal values.
        """
        signal_values = V3 * np.cos(2 * np.pi * V4 * t) + V2
        return np.where((t >= V0) & (t <= V1), signal_values, V2)

    def get_latex_expression(self):
        return "V_3 \\cos(2 \\pi V_4 t) + V_2, \\text{for } t \\in [V_0, V_1]"

class SquareSignal(BaseSignal):
    """
    Generates a square wave signal within a specified time domain.

    Parameters:
    - V0 : float
        Start time of the signal's active domain.
    - V1 : float
        End time of the signal's active domain.
    - V2 : float
        Offset value outside the specified domain.
    - V3 : float
        Amplitude of the square wave.
    - V4 : float
        Frequency of the square wave.
    - V5 : float
        Duty cycle of the square wave.

    The signal value within [V0, V1] follows the square wave expression, while outside this range, it equals the offset.
    """
    def __init__(self):
        super().__init__()
        self.variables = {"V0": "start", "V1": "end", "V2": "offset", "V3": "amplitude", "V4": "frequency", "V5": "duty"}

    def generate(self, t, V0, V1, V2, V3, V4, V5):
        """
        Generates a square wave signal within the specified domain.

        Parameters:
        t : array-like
            Time array.
        V0 : float
            Start time of the signal's active domain.
        V1 : float
            End time of the signal's active domain.
        V2 : float
            Offset value.
        V3 : float
            Amplitude of the square wave.
        V4 : float
            Frequency of the square wave.
        V5 : float
            Duty cycle of the square wave.

        Returns:
        array-like
            Square wave signal values.
        """
        signal_values = V3 * signal.square(2 * np.pi * V4 * t, duty=V5) + V2
        return np.where((t >= V0) & (t <= V1), signal_values, V2)

    def get_latex_expression(self):
        return "V_3 \\text{square}(2 \\pi V_4 t, \\text{duty}=V_5) + V_2, \\text{for } t \\in [V_0, V_1]"

class TriangleSignal(BaseSignal):
    """
    Generates a triangle wave signal within a specified time domain.

    Parameters:
    - V0 : float
        Start time of the signal's active domain.
    - V1 : float
        End time of the signal's active domain.
    - V2 : float
        Offset value outside the specified domain.
    - V3 : float
        Amplitude of the triangle wave.
    - V4 : float
        Frequency of the triangle wave.

    The signal value within [V0, V1] follows the triangle wave expression, while outside this range, it equals the offset.
    """
    def __init__(self):
        super().__init__()
        self.variables = {"V0": "start", "V1": "end", "V2": "offset", "V3": "amplitude", "V4": "frequency"}

    def generate(self, t, V0, V1, V2, V3, V4):
        """
        Generates a triangle wave signal within the specified domain.

        Parameters:
        t : array-like
            Time array.
        V0 : float
            Start time of the signal's active domain.
        V1 : float
            End time of the signal's active domain.
        V2 : float
            Offset value.
        V3 : float
            Amplitude of the triangle wave.
        V4 : float
            Frequency of the triangle wave.

        Returns:
        array-like
            Triangle wave signal values.
        """
        signal_values = V3 * signal.sawtooth(2 * np.pi * V4 * t, width=0.5) + V2
        return np.where((t >= V0) & (t <= V1), signal_values, V2)

    def get_latex_expression(self):
        return "V_3 \\text{triangle}(2 \\pi V_4 t) + V_2, \\text{for } t \\in [V_0, V_1]"

class RampSignal(BaseSignal):
    """
    Generates a ramp (linear) signal within a specified time domain.

    Parameters:
    - V0 : float
        Start time of the signal's active domain.
    - V1 : float
        End time of the signal's active domain.
    - V2 : float
        Offset value outside the specified domain.

    Within [V0, V1], the signal linearly increases from V2 to (V2 + (V1 - V0)). Outside this range, it equals the offset.
    """
    def __init__(self):
        super().__init__()
        self.variables = {"V0": "start", "V1": "end", "V2": "offset"}

    def generate(self, t, V0, V1, V2):
        """
        Generates a ramp signal within the specified domain.

        Parameters:
        t : array-like
            Time array.
        V0 : float
            Start time of the signal's active domain.
        V1 : float
            End time of the signal's active domain.
        V2 : float
            Offset value.

        Returns:
        array-like
            Ramp signal values.
        """
        ramp_signal = np.linspace(V2, V2 + (V1 - V0), len(t))
        return np.where((t >= V0) & (t <= V1), ramp_signal, V2)

    def get_latex_expression(self):
        return "\\text{linspace}(V_2, V_2 + (V_1 - V_0), t) + V_2"

class GaussianSignal(BaseSignal):
    """
    Generates a Gaussian (bell curve) signal within a specified time domain.

    Parameters:
    - V0 : float
        Start time of the signal's active domain.
    - V1 : float
        End time of the signal's active domain.
    - V2 : float
        Offset value outside the specified domain.
    - V3 : float
        Amplitude of the Gaussian.
    - V4 : float
        Mean of the Gaussian.
    - V5 : float
        Standard deviation of the Gaussian.

    Within [V0, V1], the signal follows the Gaussian expression. Outside this range, it equals the offset.
    """
    def __init__(self):
        super().__init__()
        self.variables = {"V0": "start", "V1": "end", "V2": "offset", "V3": "amplitude", "V4": "mean", "V5": "std_dev"}

    def generate(self, t, V0, V1, V2, V3, V4, V5):
        """
        Generates a Gaussian signal within the specified domain.

        Parameters:
        t : array-like
            Time array.
        V0 : float
            Start time of the signal's active domain.
        V1 : float
            End time of the signal's active domain.
        V2 : float
            Offset value.
        V3 : float
            Amplitude of the Gaussian.
        V4 : float
            Mean of the Gaussian.
        V5 : float
            Standard deviation of the Gaussian.

        Returns:
        array-like
            Gaussian signal values.
        """
        signal_values = V3 * np.exp(-((t - V4) ** 2) / (2 * V5 ** 2)) + V2
        return np.where((t >= V0) & (t <= V1), signal_values, V2)

    def get_latex_expression(self):
        return "V_3 e^{-(t - V_4)^2 / (2 V_5^2)} + V_2, \\text{for } t \\in [V_0, V_1]"

class ExponentialDecaySignal(BaseSignal):
    """
    Generates an exponentially decaying signal within a specified time domain.

    Parameters:
    - V0 : float
        Start time of the signal's active domain.
    - V1 : float
        End time of the signal's active domain.
    - V2 : float
        Offset value outside the specified domain.
    - V3 : float
        Amplitude of the decay.
    - V4 : float
        Decay rate.

    Within [V0, V1], the signal follows the exponential decay expression. Outside this range, it equals the offset.
    """
    def __init__(self):
        super().__init__()
        self.variables = {"V0": "start", "V1": "end", "V2": "offset", "V3": "amplitude", "V4": "decay_rate"}

    def generate(self, t, V0, V1, V2, V3, V4):
        """
        Generates an exponentially decaying signal within the specified domain.

        Parameters:
        t : array-like
            Time array.
        V0 : float
            Start time of the signal's active domain.
        V1 : float
            End time of the signal's active domain.
        V2 : float
            Offset value.
        V3 : float
            Amplitude of the decay.
        V4 : float
            Decay rate.

        Returns:
        array-like
            Exponentially decaying signal values.
        """
        signal_values = V3 * np.exp(-V4 * t) + V2
        return np.where((t >= V0) & (t <= V1), signal_values, V2)

    def get_latex_expression(self):
        return "V_3 e^{-V_4 t} + V_2, \\text{for } t \\in [V_0, V_1]"

class StepWithDecaySignal(BaseSignal):
    """
    Generates a step function with exponential decay within a specified time domain.

    Parameters:
    - V0 : float
        Start time of the signal's active domain.
    - V1 : float
        End time of the signal's active domain.
    - V2 : float
        Offset value outside the specified domain.
    - V3 : float
        Amplitude of the step.
    - V4 : float
        Time at which the step occurs.
    - V5 : float
        Decay rate after the step.

    Within [V0, V1], the signal follows the step-decay expression. Outside this range, it equals the offset.
    """
    def __init__(self):
        super().__init__()
        self.variables = {"V0": "start", "V1": "end", "V2": "offset", "V3": "amplitude", "V4": "step_time", "V5": "decay_rate"}

    def generate(self, t, V0, V1, V2, V3, V4, V5):
        """
        Generates a step function with exponential decay within the specified domain.

        Parameters:
        t : array-like
            Time array.
        V0 : float
            Start time of the signal's active domain.
        V1 : float
            End time of the signal's active domain.
        V2 : float
            Offset value.
        V3 : float
            Amplitude of the step.
        V4 : float
            Time at which the step occurs.
        V5 : float
            Decay rate after the step.

        Returns:
        array-like
            Step with decay signal values.
        """
        step_signal = np.heaviside(t - V4, 1)
        decay_signal = np.exp(-V5 * (t - V4)) * step_signal
        signal_values = V3 * decay_signal + V2
        return np.where((t >= V0) & (t <= V1), signal_values, V2)

    def get_latex_expression(self):
        return "V_3 \\text{Heaviside}(t - V_4) e^{-V_5 (t - V_4)} + V_2, \\text{for } t \\in [V_0, V_1]"

class SpiralScanCosSignal(BaseSignal):
    """
    Generates the cosine component of a spiral scan signal within a specified time domain.

    Parameters:
    - V0 : float
        Start time of the signal's active domain.
    - V1 : float
        End time of the signal's active domain.
    - V2 : float
        Offset value outside the specified domain.
    - V3 : float
        Amplitude of the spiral.
    - V4 : float
        Frequency of the spiral rotation.
    - V5 : float
        Radius growth rate.

    Within [V0, V1], the signal follows the spiral cosine expression. Outside this range, it equals the offset.
    """
    def __init__(self):
        super().__init__()
        self.variables = {"V0": "start", "V1": "end", "V2": "offset", "V3": "amplitude", "V4": "frequency", "V5": "radius_growth_rate"}

    def generate(self, t, V0, V1, V2, V3, V4, V5):
        """
        Generates the cosine component of a spiral scan within the specified domain.

        Parameters:
        t : array-like
            Time array.
        V0 : float
            Start time of the signal's active domain.
        V1 : float
            End time of the signal's active domain.
        V2 : float
            Offset value.
        V3 : float
            Amplitude of the spiral.
        V4 : float
            Frequency of the spiral rotation.
        V5 : float
            Radius growth rate.

        Returns:
        array-like
            Cosine component of the spiral scan.
        """
        signal_values = (V3 + V5 * t) * np.cos(2 * np.pi * V4 * t) + V2
        return np.where((t >= V0) & (t <= V1), signal_values, V2)

    def get_latex_expression(self):
        return "(V_3 + V_5 t) \\cos(2 \\pi V_4 t) + V_2, \\text{for } t \\in [V_0, V_1]"

class SpiralScanSinSignal(BaseSignal):
    """
    Generates the sine component of a spiral scan signal within a specified time domain.

    Parameters:
    - V0 : float
        Start time of the signal's active domain.
    - V1 : float
        End time of the signal's active domain.
    - V2 : float
        Offset value outside the specified domain.
    - V3 : float
        Amplitude of the spiral.
    - V4 : float
        Frequency of the spiral rotation.
    - V5 : float
        Radius growth rate.

    Within [V0, V1], the signal follows the spiral sine expression. Outside this range, it equals the offset.
    """
    def __init__(self):
        super().__init__()
        self.variables = {"V0": "start", "V1": "end", "V2": "offset", "V3": "amplitude", "V4": "frequency", "V5": "radius_growth_rate"}

    def generate(self, t, V0, V1, V2, V3, V4, V5):
        """
        Generates the sine component of a spiral scan within the specified domain.

        Parameters:
        t : array-like
            Time array.
        V0 : float
            Start time of the signal's active domain.
        V1 : float
            End time of the signal's active domain.
        V2 : float
            Offset value.
        V3 : float
            Amplitude of the spiral.
        V4 : float
            Frequency of the spiral rotation.
        V5 : float
            Radius growth rate.

        Returns:
        array-like
            Sine component of the spiral scan.
        """
        signal_values = (V3 + V5 * t) * np.sin(2 * np.pi * V4 * t) + V2
        return np.where((t >= V0) & (t <= V1), signal_values, V2)

    def get_latex_expression(self):
        return "(V_3 + V_5 t) \\sin(2 \\pi V_4 t) + V_2, \\text{for } t \\in [V_0, V_1]"

class CustomExpressionSignal(BaseSignal):
    """
    Generates a custom signal based on a user-provided NumPy expression within a specified time domain.

    Parameters:
    - V0 : float
        Start time of the signal's active domain.
    - V1 : float
        End time of the signal's active domain.
    - V2 : float
        Offset value outside the specified domain.
    - Additional custom variables as needed by the expression.

    Within [V0, V1], the signal follows the user-provided expression. Outside this range, it equals the offset.
    """
    def __init__(self, expression):
        super().__init__()
        self.expression = expression
        # Customize variable mapping based on expression needs
        self.variables = {"V0": "start", "V1": "end", "V2": "offset",
                          "V3": "custom_param1", "V4": "custom_param2", 
                          "V5": "custom_param3", "V6": "custom_param4"}

    def generate(self, t, **kwargs):
        """
        Generates a custom signal based on the provided expression within the specified domain.

        Parameters:
        t : array-like
            Time array.
        kwargs : dict
            Dictionary of variable names and their values (including V0, V1, V2, and any additional parameters required by the expression).

        Returns:
        array-like
            Custom signal values.
        """
        expr = self.expression
        for key, value in kwargs.items():
            expr = expr.replace(key, str(value))
        evaluated_expr = eval(expr, {"t": t, "np": np})
        return np.where((t >= kwargs.get("V0", 0)) & (t <= kwargs.get("V1", np.inf)), evaluated_expr, kwargs.get("V2", 0))

    def get_latex_expression(self):
        """
        Returns the LaTeX representation of the custom expression signal.
        """
        return self.expression
