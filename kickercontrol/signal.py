import warnings

import numpy as np
import xarray as xr

from kickercontrol.__base__ import (
    LineSignal, SinSignal, CosSignal, SquareSignal, TriangleSignal, RampSignal,
    GaussianSignal, ExponentialDecaySignal, StepWithDecaySignal, SpiralScanCosSignal, SpiralScanSinSignal, CustomExpressionSignal
)
from kickercontrol.utils import float_to_16bit_int
from kickercontrol.timing import get_region_bounds

class SignalGenerator:
    def __init__(self, t, unit='bunches', dt=None, oscillator=None, relative_scan = False, **kwargs):
        """
        Initialize the SignalGenerator class to create arbitrary signals.
        
        Parameters:
        t (int or float): time-axis
        unit (str): Unit for the signal duration, either 'bunches' (integer) or 'time' (seconds).
        dt (float): Time interval between points, required if unit is 'time'.
        oscillator (str): Type of signal to generate.
        kwargs: Additional arguments for each signal type.
        """

        self.__base__ = {
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
        self.relative_scan = relative_scan

        if oscillator is not None:
            self.set_oscillator(oscillator, **kwargs)

    def set_oscillator(self, oscillator, **kwargs):
        """
        Set the oscillator.
        
        Parameters:
        oscillator (str): Type of signal to generate.
        kwargs: Additional arguments for each signal type.
        """

        if oscillator not in self.__base__:
            raise ValueError(f"Unsupported signal type. Choose from {list(self.__base__.keys())}")

        oscillator_class = self.__base__[oscillator]
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
    """
    DACSignalGenerator

    A subclass of SignalGenerator for generating and writing signals to a DAC (Digital to Analog Converter) 
    device through a kicker control. This class specifically integrates with beamlines and kickers to generate 
    signals suitable for controlling devices in an accelerator or similar environment.

    Attributes:
    -----------
    kicker : KickerControl
        An instance of the KickerControl class that controls the DAC device.Provides time interval and signal duration.
    beamline : str or None
        The identifier of the beamline to target for signal generation. Allowed values are: [None, "D", "1", "2", "3", "13", "4"].
    
    Methods:
    --------
    __init__(self, kicker_device, beamline=None, oscillator=None, **kwargs)
        Initializes the DACSignalGenerator object. Sets up kicker and beamline, and initializes the base SignalGenerator.
        
    generate_signal(self)
        Generates the DAC-compatible signal based on the set oscillator and signal parameters. Ensures that the signal is 
        within the specified beamline bounds.
        
    write_dac_signal(self)
        Writes the generated signal to the DAC device via the kicker.
    """

    def __init__(self, kicker_device, beamline = None, oscillator = None, **kwargs):
        """
        Initialize the DACSignalGenerator class, inheriting from SignalGenerator, for generating DAC signals.
        
        Parameters:
        ----------
        kicker_device (KickerControl): 
            Instance of a KickerControl class to write the generated signals. The kicker provides time interval and signal duration.
        beamline (str, optional): 
            Specification of SASE beamline. Allowed values are: [None, "D", "1", "2", "3", "13", "4"].
        oscillator (str, optional): 
            Type of signal to generate. Must be one of the supported signal types defined in the SignalGenerator.
        kwargs: 
            Additional arguments for each signal type, e.g., amplitude or frequency.
        """
        beamlines = [None, "D", "1", "2", "3", "13", "4"]
        assert beamline in beamlines, f"Specified Beamline does not exist. beamline should be in {beamlines}"

        self.beamline = beamline

        assert type(kicker_device) != type, "Instantiate the kicker device (e.g., KickerDevice() ) before passing to DACSignalGenerator"
        self.kicker = kicker_device

        super().__init__(kicker_device.t, unit='time', oscillator = oscillator, **kwargs)

    def generate_signal(self):
        """
        Generate the signal using the configured oscillator.
        
        Returns:
        xarray.DataArray: Generated signal values.
        """
        if self.oscillator is None:
            raise RuntimeError("Oscillator has not been set. Use 'set_oscillator' to define the signal type.")
        
        ### hard limit for which beam regions can be written

        if self.beamline is not None:
            ti, tf = get_region_bounds(self.beamline)

            if self.signal_params["V0"] < ti:
                warnings.warn(f"Cannot write before beam region {self.beamline} starting @ {ti} us", UserWarning)
                self.signal_params["V0"] = ti
            if self.signal_params["V1"] > tf:
                warnings.warn(f"Cannot write after beam region {self.beamline} ending @ {tf} us", UserWarning)
                self.signal_params["V1"] = tf

        signal_values = float_to_16bit_int(self.oscillator.generate(self.t, **self.signal_params))

        if self.beamline is not None:
            time, current_signal = self.kicker.read_dac().T
            signal_values[time < ti] = current_signal[time < ti]
            signal_values[time > tf] = current_signal[time > tf]
        
        return xr.DataArray(signal_values, dims=['time'], coords={'time': self.t}, attrs={'unit': 'us'})


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
        try:
            self.kicker.write_dac(signal_data.values, self.relative_scan)
        except Exception as e:
            print("Could not write to device")
            print(e)