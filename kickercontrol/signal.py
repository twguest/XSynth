import numpy as np
import xarray as xr
from kickercontrol.__base__ import (
    LineSignal, SinSignal, CosSignal, SquareSignal, TriangleSignal, RampSignal,
    GaussianSignal, ExponentialDecaySignal, StepWithDecaySignal, SpiralScanCosSignal, SpiralScanSinSignal, CustomExpressionSignal
)

class SignalGenerator:
    def __init__(self, t, unit='bunches', dt=None, oscillator=None, **kwargs):
        """
        Initialize the SignalGenerator class to create arbitrary signals.
        
        Parameters:
        t (int or float): time-axis
        unit (str): Unit for the signal duration, either 'bunches' (integer) or 'time' (seconds).
        dt (float): Time interval between points, required if unit is 'time'.
        oscillator (str): Type of signal to generate.
        kwargs: Additional arguments for each signal type.
        """

        self.oscillators = {
            'line': LineSignal,
            'sin': SinSignal,
            'cos': CosSignal,
            'square': SquareSignal,
            'triangle': TriangleSignal,
            'ramp': RampSignal,
            'gaussian': GaussianSignal,
            'exponential_decay': ExponentialDecaySignal,
            'step_with_decay': StepWithDecaySignal,
            'spiral_cos': SpiralScanCosSignal,
            'spiral_sin': SpiralScanSinSignal,
            'custom': CustomExpressionSignal
        }

        self.unit = unit
        
        if self.unit == 'bunches':
            if not isinstance(self.t, int):
                raise ValueError("duration must be an integer when unit is 'bunches'")

        elif self.unit not in ['bunches', 'time']:
            raise ValueError("Unit must be either 'bunches' or 'time'")


        self.t = t
        self.oscillator = None
        self.variables = None
        self.signal_params = None
        self.generated_signal = None

        if oscillator is not None:
            self.set_oscillator(oscillator, **kwargs)

    def set_oscillator(self, oscillator, **kwargs):
        """
        Set the oscillator.
        
        Parameters:
        oscillator (str): Type of signal to generate.
        kwargs: Additional arguments for each signal type.
        """

        if oscillator not in self.oscillators:
            raise ValueError(f"Unsupported signal type. Choose from {list(self.oscillators.keys())}")

        oscillator_class = self.oscillators[oscillator]
        if oscillator == 'custom':
            if 'expression' not in kwargs:
                raise ValueError("For custom signal, 'expression' parameter must be provided")
            self.oscillator = oscillator_class(kwargs['expression'])
        else:
            self.oscillator = oscillator_class()
        
        self.variables = self.oscillator.get_variable_mapping()
        self.default_values = self.oscillator.default_values ### Note: variables should also be defined this way
        # Set default values if not provided in kwargs
        for var in self.default_values.keys():
            if var not in kwargs:
                kwargs[var] = self.default_values[var]  # Default value for all other parameters is 1
        

        self.signal_params = kwargs

        
    def generate_signal(self):
        """
        Generate the signal using the configured oscillator.
        
        Returns:
        xarray.DataArray: Generated signal values.
        """
        if self.oscillator is None:
            raise RuntimeError("Oscillator has not been set. Use 'set_oscillator' to define the signal type.")
        
        signal_values = self.oscillator.generate(self.t, **self.signal_params)
        return xr.DataArray(signal_values, dims=['time'], coords={'time': self.t}, attrs={'unit': 'us'})

    def update_signal(self):
        """
        Update the signal after changing the oscillator or parameters.
        """
        if self.oscillator is None:
            raise RuntimeError("Oscillator has not been set. Use 'set_oscillator' to define the signal type.")
        
        self.generated_signal = self.generate_signal()

    def update_variable(self, **kwargs):
        """
        Update the value of one or more signal parameters and regenerate the signal.
        
        Parameters:
        kwargs: Dictionary of variable names and their new values.
        """
        if self.signal_params is None:
            raise RuntimeError("Signal parameters have not been set. Use 'set_oscillator' to define the signal type and parameters.")
        
        for key, value in kwargs.items():
            if key in self.signal_params:
                self.signal_params[key] = value
            else:
                pass #raise ValueError(f"Parameter '{key}' is not a valid parameter for the current oscillator.")
        if self.oscillator is not None:
            self.update_signal()

class DACSignalGenerator(SignalGenerator):
    def __init__(self, kicker_device, oscillator = None, **kwargs):
        """
        Initialize the DACSignalGenerator class, inheriting from SignalGenerator, for generating DAC signals.
        
        Parameters:
        kicker (KickerControl): Instance of a KickerControl class to write the generated signals. The kicker will provide time interval and signal duration.
        """
        self.kicker = kicker_device

        super().__init__(kicker_device.t, unit='time', oscillator = oscillator, **kwargs)
    
    def write_dac_signal(self):
        """
        Generate a specified type of signal and write it to the DAC device via the kicker.
        
        Parameters:
        oscillator (str): Type of signal to generate.
        kwargs: Additional arguments for each signal type.
        
        Returns:
        None
        """
        if self.kicker is None:
            raise ValueError("A kicker must be specified to write DAC signal.")

        signal_data = self.generated_signal
        # Here you would write the signal data to the DAC device through the kicker
        # This is a placeholder for the actual writing logic
        print(f"Writing signal to kicker {self.kicker.device_location}...")
        
        self.kicker.write_dac(signal_data.values)
        