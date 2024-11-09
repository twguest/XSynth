# Python script to control KL stripline kickers at the European XFEL using pyDOOCS

import pydoocs
import numpy as np

class KickerDevice:
    def __init__(self, device_location):
        """
        Initialize the KickerControl class with the given device path.
        
        Parameters:
        device_location (str): The full path of the kicker device to control, including channel.
        """
        self.device_location = device_location
        self.t, self.initial_signal = pydoocs.read(self.device_location)['data'].T
        
        ### this could be problematic and may have to depend on which kicker is used
        self.initial_signal-=32767

    def read_dac(self):
        return pydoocs.read(self.device_location)['data']

    def write_dac(self, pulse_values, relative_scan = False):
        """
        Write the DAC values using a vector with time intervals.
        
        Parameters:
        pulse_values (numpy.ndarray): The pulse values to write to the DAC.
        read (Bool): Scan relative to the intial signla
        """
        
        ### may not strictly be true
        assert np.max(abs(pulse_values)) <= 32767, "Kicker Strengths must be in the domain (-32767, 32767)"
        
        if relative_scan:
            pulse_values+=self.initial_signal
        try:
            pydoocs.write(self.device_location, pulse_values.tolist())
            print("Succesful Write")
        except Exception as e:
            print(f"Failed to write DAC: {e}")

    @property
    def __name__(self):
        return ""

    @property
    def get_signal_interval(self):
        """
        Return the signal length for the kicker
        """

        return np.ptp(self.t)/len(self.t)
        
    def get_signal_length(self):
        """
        Determine the signal length for the kicker.
        
        Returns:
        int: The signal length as an integer.
        """
        return len(pydoocs.read(self.device_location)['data'][:,0])

    def __enter__(self):
        """
        Context management entry method.
        """
        print(f"Entering context for device: {self.device_location}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Context management exit method.
        """
        print(f"Exiting context for device: {self.device_location}")


class KL2005(KickerDevice):

    def __init__(self):
        
        super().__init__(device_location="XFEL.DIAG/SIS8300DMA/DI2001TL.2/DAC_CH0.TD")

    @property
    def __name__(self):
        return "KL2005"

class KMX1938(KickerDevice):

    def __init__(self):
        
        super().__init__(device_location="XFEL.DIAG/SIS8300DMA/DI1950TL.3/DAC_CH0.TD")

    @property
    def __name__(self):
        return "KMX1938"

class KNY1938(KickerDevice):

    def __init__(self):
        
        super().__init__(device_location="XFEL.DIAG/SIS8300DMA/DI1950TL.3/DAC_CH1.TD")

    @property
    def __name__(self):
        return "KNY1938"
    
class KMX1965(KickerDevice):

    def __init__(self):
        
        super().__init__(device_location="XFEL.DIAG/SIS8300DMA/DI1950TL.4/DAC_CH0.TD")

    @property
    def __name__(self):
        return "KMX1965"
    
class KNY1965(KickerDevice):

    def __init__(self):
        
        super().__init__(device_location="XFEL.DIAG/SIS8300DMA/DI1950TL.4/DAC_CH1.TD")

    @property
    def __name__(self):
        return "KNY1965"

if __name__ == "__main__":
    # Example usage with context management
    with KickerDevice("XFEL.DIAG/SIS8300DMA/DI1950TL.3/DAC_CH0.TD") as kicker:
        # Create a time vector and pulse values (example: sine wave)
 
        time_vector = np.arange(0, 10 * time_increment, time_increment)  # Generate time points
        pulse_values = np.sin(2 * np.pi * time_vector)  # Example pulse shape (sine wave)
        
        # Write pulse values to DAC
        kicker.write_dac(pulse_values)
