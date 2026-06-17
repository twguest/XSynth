# Python script to control KL stripline kickers at the European XFEL using pyDOOCS

import pydoocs
import numpy as np

    
#### Legacy for writing directly to the kickers
#----------------------------------------------

class KickerDevice:
    """
    
    """
    def __init__(self, device_location):
        """
        Initialize the xsynth class with the given device path.
        
        Parameters:
        device_location (str): The full path of the kicker device to control, including channel.
        """
        self.device_type = "DAC"
        self.device_location = device_location
        self.t, self.initial_signal = pydoocs.read(self.device_location)['data'].T
        
        ### this could be problematic and may have to depend on which kicker is used

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
        #assert np.max(abs(pulse_values)) <= 32767, "Kicker Strengths must be in the domain (-32767, 32767)"
        
        if relative_scan:
            pulse_values+=self.initial_signal

        try:
            pydoocs.write(self.device_location, pulse_values.tolist())
            #print("Succesful Write")
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
        
        super().__init__(device_location="XFEL.DIAG/SIS8300DMA/DI2001TL.2/DAC_CH.TD")

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
    

#### ADAPTATION SERVER AND MIDDLE LAYER
#----------------------------------------


class ADAPTSERVER(KickerDevice):
    """
    Adaptation Server 
    """
    def __init__(self, device_location, beam_region):
        """
        Initialize the xsynth class with the given device path.
        
        Parameters:
        device_location (str): The full path of the kicker device to control, including channel.
        """
        self.device_type = "ADAPT"
        self.device_location = device_location

        self.beam_region = beam_region
        assert self.beam_region in ["ALL", "SA1", "SA2", "SA3", "SA4"]
        
        self.ramp_location = device_location + "BPM.OFFSET.RAMP." + self.beam_region
                
        self.initial_signal = pydoocs.read(self.ramp_location)['data']

    @property
    def __name__(self):
        return "ADAPTATION_SERVER"

    @property
    def pulseId(self):
        return pydoocs.read(f"XFEL.FEEDBACK/KICKERDC_BETA/CONTROL/BPM.PULSES.{self.beam_region}.INDX")['data']
    
    def read(self):
        return pydoocs.read(self.ramp_location)['data']


    def write(self, signal, pulseId):
        """
        Write values
        
        Parameters:
        signal (numpy.ndarray): The signal values to the ramp location
        relative (Bool): Scan relative to the intial signla
        """
        ramp_signal = self.read()
        ramp_signal[pulseId] = signal

        try:
            pydoocs.write(self.ramp_location, ramp_signal)
            pydoocs.write(self.device_location + "ADAPT", 1)
            #print("Succesful Write")
        except Exception as e:
            print(f"Failed to write to Adaptation Server: {e}")


class ADAPTIONMIDDLELAYER(KickerDevice):
    """
    Adaptation Server 
    """
    def __init__(self, device_location, beam_region):
        """
        Initialize the xsynth class with the given device path.
        
        Parameters:
        device_location (str): The full path of the kicker device to control, including channel.
        """
        self.device_type = "ADAPT_MLS"
        self.device_location = device_location

        self.beam_region = beam_region
        assert self.beam_region in ["SA2"]
        
        self.ramp_location = device_location + "USER.BPM.OFFSET.RAMP." + self.beam_region
                
        self.initial_signal = pydoocs.read(self.ramp_location)['data']

    @property
    def __name__(self):
        return "ADAPTATION_MLS"

    @property
    def pulseId(self):
        return pydoocs.read(f"XFEL.FEEDBACK/KICKERDC_BETA/CONTROL/BPM.PULSES.{self.beam_region}.INDX")['data']
    
    def read(self):
        return pydoocs.read(self.ramp_location)['data']


    def write(self, signal, pulseId):
        """
        Write values
        
        Parameters:
        signal (numpy.ndarray): The signal values to the ramp location
        relative (Bool): Scan relative to the intial signla
        """
        ramp_signal = self.read()
        ramp_signal[pulseId] = signal

        try:
            pydoocs.write(self.ramp_location, ramp_signal)
            pydoocs.write(self.device_location + "USER.SET.RAMP.SA2", 1)
            pydoocs.write(self.device_location + "USER.ADAPT.SA2", 1)
            
            #print("Succesful Write")
        except Exception as e:
            print(f"Failed to write to Adaptation MLS: {e}")
            raise


#### ADAPTATION SERVER AND MIDDLE LAYER
#----------------------------------------


class IBFB_Server(KickerDevice):
    """
    T-IBFB Server 


    ### Need to check timing of shadow tables
    """
    def __init__(self, device_location, beam_region):
        """
        Initialize the xsynth class with the given device path.
        
        Parameters:
        device_location (str): The full path of the kicker device to control, including channel.
        """
        self.device_type = "IBFB"
        self.device_location = device_location

        self.beam_region = beam_region
        assert self.beam_region in ["ALL", "SA1", "SA2", "SA3", "SA4"]
      
        self.initial_signal = pydoocs.read(self.device_location)['data'][:,1]

    @property
    def __name__(self):
        return "T-IBFB Server"

### ONLY TRUE IF T-IBFB has same time structure as ADAPTATION SERVER, which may not be the case. TO-DO CHECK THIS
    @property
    def pulseId(self):
        return pydoocs.read(f"XFEL.FEEDBACK/KICKERDC_BETA/CONTROL/BPM.PULSES.{self.beam_region}.INDX")['data']
    
    def read(self):
        return pydoocs.read(self.device_location)['data'][:,1]

    
    ### TO-DO: 
    def write(self, signal, pulseId):
        """
        Write values
        
        Parameters:
        signal (numpy.ndarray): The signal values to the ramp location
        relative (Bool): Scan relative to the intial signal
        """
        init_signal = self.read()
        init_signal[pulseId] = signal

        try:
            pydoocs.write(self.device_location, init_signal)
            pydoocs.write(self.apply_location, 1)
            #print("Succesful Write")
        except Exception as e:
            print(f"Failed to write to T-IBFB Server: {e}")


class IBFB_X(IBFB_Server):

    def __init__(self, beam_region):
        
        super().__init__(device_location="XFEL.DIAG/DAMC2IBFB/DI1914TL.0_CTRL/SHADOW_TABLE_X_1",
                         beam_region = beam_region)
        
        self.apply_location = "XFEL.DIAG/DAMC2IBFB/DI1914TL.0_CTRL/FF_COMMAND_X"

    @property
    def __name__(self):
        return "IBFB_X"
    

class IBFB_Y(IBFB_Server):

    def __init__(self, beam_region):
        
        super().__init__(device_location="XFEL.DIAG/DAMC2IBFB/DI1914TL.0_CTRL/SHADOW_TABLE_Y_1",
                         beam_region = beam_region)
        
        self.apply_location = "XFEL.DIAG/DAMC2IBFB/DI1914TL.0_CTRL/FF_COMMAND_Y"

    @property
    def __name__(self):
        return "IBFB_Y"
    

class ADAPTX(ADAPTSERVER):

    def __init__(self, beam_region):
        
        super().__init__(device_location="XFEL.FEEDBACK/KICKERDC_BETA/ADAPT.EXTRACTION.X/",
                         beam_region = beam_region)

    @property
    def __name__(self):
        return "ADAPTX"
    

class ADAPT_MLS(ADAPTIONMIDDLELAYER):

    def __init__(self, beam_region):
        
        super().__init__(device_location="XFEL.FEL/KICKER.ADAPTION.ML/SA2/",
                         beam_region = beam_region)

    @property
    def __name__(self):
        return "ADAPT_MLS"
    
    


if __name__ == "__main__":
    # Example usage with context management
    with KickerDevice("XFEL.DIAG/SIS8300DMA/DI1950TL.3/DAC_CH0.TD") as kicker:
        # Create a time vector and pulse values (example: sine wave)
 
        time_vector = np.arange(0, 10 * time_increment, time_increment)  # Generate time points
        pulse_values = np.sin(2 * np.pi * time_vector)  # Example pulse shape (sine wave)
        
        # Write pulse values to DAC
        kicker.write_dac(pulse_values)
