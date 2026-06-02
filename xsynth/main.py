from xsynth.signal import DACSignalGenerator, ADAPTSignalGenerator
from xsynth.scan import MeshScan
from xsynth.timing import get_region_bounds

def Scan(devices,
              scan_vectors,
              oscillators,
              oscillator_variables,
              scan_variables,
              wait_time=0.1,
              write = False,
              all_messages = True,
              display = False,
              beam_region = None,
              relative_scan = False,
              restore = True,
                **kwargs):
            """
            Conducts a scanning routine for the specified kicker devices using a mesh scan.

            Parameters:
            - devices: list
                List of kicker device objects to be scanned.
            - scan_vectors: list
                List of scan vectors, one for each kicker device, defining the trajectory or parameter space for each device scan.
            - oscillators: list
                List of oscillators controlling each kicker device.
            - oscillator_variables: list of dict
                List of dictionaries, each specifying configuration variables for corresponding oscillators, e.g., "V0", "V1", "V4".
            - scan_variables: list
                Variables to be scanned for each kicker device, typically defined as a list of strings (e.g., "V7").
            - wait_time: float, optional (default=0.1)
                Time in seconds to wait between each scan iteration.
            - write: bool, optional (default=False)
                Flag to indicate if DAC values should be written after the scan completes.
            - all_messages: bool, optional (default=True)
                If True, prints status and debug messages throughout the scan.
            - display: bool, optional (default=False)
                If True, displays scan plots if applicable.
            - beam_region: str or None, optional (default=None)
                Specifies the beam_region; if provided, it sets up region bounds for oscillators using `get_region_bounds`.
            - kwargs: dict, optional
                Additional keyword arguments passed to DACSignalGenerator or MeshScan.

            Returns:
            - MeshScan object representing the completed scan.

            Raises:
            - AssertionError if lengths of kicker_devices, oscillators, and oscillator_variables are not equal.

            Example:
            ```
            scan_result = Scan(kicker_devices, scan_vectors, oscillators, oscillator_variables, scan_variables)
            ```

            Notes:
            This routine saves initial DAC conditions of each kicker, initializes DAC generators for each kicker with the
            appropriate oscillator, and then executes a mesh scan over specified variables.
            """

            assert len(devices) == len(oscillator_variables) == len(oscillators)
            initial_kicker_conditions = []
        
            generators = []

            for itr, device in enumerate(devices):
                
                os = oscillators[itr]
                ov = oscillator_variables[itr]

                Device = device()

                if Device.device_type == 'DAC':
                     Generator = DACSignalGenerator
                elif Device.device_type == 'ADAPT':
                     Generator = ADAPTSignalGenerator 
                
                generators.append(Generator(Device,
                                            oscillator = os,
                                            beam_region = beam_region,
                                            relative_scan=relative_scan,
                                            **ov,
                                            **kwargs))

            
            M = MeshScan(generators,
                         scan_variables,
                         scan_vectors,
                         wait_time,
                         all_messages = all_messages,
                         plot_display = display)
            """"""
            M.execute_scan(write)

            #M['initial_kicker_conditions'] = initial_kicker_conditions

            if restore:
                pass


            return M


def SignalGenerator(kicker_devices,
              oscillators,
              oscillator_variables,
              wait_time=0,
              write = False,
              all_messages = True,
              beam_region = '2',
              display = True,
              relative_scan = False,
                **kwargs):
    """
    Wrapper function for `Scan` to initialize a signal generator configuration.

    Parameters:
    - kicker_devices: list
        List of kicker device objects to control.
    - oscillators: list
        List of oscillators controlling each kicker device.
    - oscillator_variables: list of dict
        Configuration variables for each oscillator.
    - wait_time: float, optional (default=0)
        Time to wait between scan iterations.
    - write: bool, optional (default=False)
        If True, restores DAC conditions after execution.
    - all_messages: bool, optional (default=True)
        Enables message display during execution.
    - beam_region: str, optional (default='2')
        Specifies the beam_region used for `Scan`.
    - kwargs: dict, optional
        Additional arguments passed to `Scan`.

    Returns:
    - MeshScan object generated by `Scan`.

    Example:
    ```
    signal_gen = SignalGenerator(kicker_devices, oscillators, oscillator_variables)
    ```
    """
    
    return Scan(kicker_devices=kicker_devices,
                scan_vectors=[None for k in kicker_devices],
                oscillators = oscillators,
                oscillator_variables=oscillator_variables,
                scan_variables=["V7" for k in kicker_devices],
                write = write,
                all_messages = all_messages,
                display = display,
                beam_region = beam_region,
                wait_time= wait_time,
                **kwargs)




def MacroScan(kicker_devices,
              scan_vectors,
              write = False,
              all_messages = False,
              display = False,
              beam_region = None,
              wait_time = 1,
              relative_scan = False):
    
    N = len(kicker_devices)
    scan_output = Scan(kicker_devices,
              scan_vectors = scan_vectors,
              oscillators = ['line' for n in range(N)],
              scan_variables=["V2" for n in range(N)],
              oscillator_variables=[{} for n in range(N)],
              write = write,
              display = display,
              all_messages=all_messages,
              beam_region = beam_region,
              wait_time = wait_time,
              relative_scan=relative_scan)
    
    return scan_output


def SetKicker(kicker_device,
              value,
              beam_region = '2',
              all_messages = True,
              write = True,
              display = False,
              start_time = None,
              end_time = None,
              **kwargs
              ):
    return SignalGenerator(kicker_devices=[kicker_device],
                        oscillators = ['line'],
                        oscillator_variables=[{"V0": start_time, "V1":end_time,'V2': value}],
                        wait_time=0,
                        write=write,
                        all_messages=all_messages,
                        beam_region = beam_region,
                        restore = False,
                        display = display,
                        **kwargs
                        )


from xsynth.timing import get_region_bounds


def SinScan(kicker_device,
     scan_vector,
     scan_variable,
     wait_time = 0,
     start_time = None,
     end_time = None,
     offset = 0,
     amplitude = 1,
     periods = 1,
     phase = 0,
     beam_region = '2',
     all_messages = False,
     display = True,
     relative_scan = False,
     restore = True,
     write = True
     ):

     ti, tf = get_region_bounds(beam_region)

     if start_time is None:
          start_time = ti
     if end_time is None:
          end_time = tf
     

     return Scan(kicker_devices=[kicker_device],
          scan_vectors=[scan_vector],
          oscillators = ['sin'],
          oscillator_variables=[{"V0": start_time,
                                 "V1": end_time,
                              "V2": offset,
                              "V3":amplitude,
                              "V4": periods/(tf-ti),
                              "V5": phase}],
          scan_variables=[scan_variable],
          wait_time=wait_time,
          write = write,
          display = display,
          beam_region = beam_region,
          all_messages=all_messages,
          relative_scan=relative_scan,
          restore=restore)

def RampScan(kicker_device,
            scan_vector,
            scan_variable,
            wait_time = 0,
            start_time = None,
            end_time = None,
            offset = 0,
            start_value = 0,
            end_value = 1,
            beam_region = '2',
            all_messages = False,
            display = True,
            relative_scan = False,
            restore = True,
            write = True
            ):
    ti, tf = get_region_bounds(beam_region)

    if start_time is None:
        start_time = ti
    if end_time is None:
        end_time = tf

    return Scan(kicker_devices=[kicker_device],
        scan_vectors=[scan_vector],
        oscillators = ['ramp'],
        oscillator_variables=[{"V0": start_time,
                                "V1": end_time,
                            "V2": offset,
                            "V3":start_value,
                            "V4": end_value}],
        scan_variables=[scan_variable],
        wait_time=wait_time,
        write = write,
        display = display,
        beam_region = beam_region,
        all_messages=all_messages,
        relative_scan=relative_scan,
        restore=restore)


def SquareScan(kicker_device,
                scan_vector,
                scan_variable,
                wait_time = 0,
                start_time = None,
                end_time = None,
                offset = 0,
                amplitude = 1,
                n_frequency = 1,
                duty = 1,
                beam_region = '2',
                all_messages = False,
                display = True,
                relative_scan = False,
                restore = True,
                write = True
                ):
    ti, tf = get_region_bounds(beam_region)

    if start_time is None:
        start_time = ti
    if end_time is None:
        end_time = tf
    

    return Scan(kicker_devices=[kicker_device],
        scan_vectors=[scan_vector],
        oscillators = ['square'],
        oscillator_variables=[{"V0": start_time,
                                "V1": end_time,
                            "V2": offset,
                            "V3":amplitude,
                            "V4": n_frequency/((tf-ti)),
                            "V5": duty}],
        scan_variables=[scan_variable],
        wait_time=wait_time,
        write = write,
        display = display,
        beam_region = beam_region,
        all_messages=all_messages,
        relative_scan=relative_scan,
        restore=restore)