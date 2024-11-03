from kickercontrol.signal import DACSignalGenerator
from kickercontrol.scan import MeshScan
from kickercontrol.timing import get_region_bounds

def Scan(kicker_devices,
              scan_vectors,
              oscillators,
              oscillator_variables,
              scan_variables,
              wait_time=0.1,
              write_dac = False,
              all_messages = True,
              display = False,
              beamline = None,
                **kwargs):
            """
            Conducts a scanning routine for the specified kicker devices using a mesh scan.

            Parameters:
            - kicker_devices: list
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
            - write_dac: bool, optional (default=False)
                Flag to indicate if DAC values should be written after the scan completes.
            - all_messages: bool, optional (default=True)
                If True, prints status and debug messages throughout the scan.
            - display: bool, optional (default=False)
                If True, displays scan plots if applicable.
            - beamline: str or None, optional (default=None)
                Specifies the beamline; if provided, it sets up region bounds for oscillators using `get_region_bounds`.
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

            assert len(kicker_devices) == len(oscillator_variables) == len(oscillators)
            initial_kicker_conditions = []
        
            dac_generators = []

            for itr, kicker in enumerate(kicker_devices):
                
                os = oscillators[itr]
                ov = oscillator_variables[itr]

                initial_kicker_conditions.append(kicker().read_dac()[:,1]) 
                
                if beamline is not None:
                    ti, tf = get_region_bounds(beamline)

                    if "V0" not in ov:
                          ov["V0"] = ti
                    if "V1" not in ov:
                          ov["V1"] = tf
                    if "V4" not in ov:
                          ov["V4"] = 1/(tf-ti)

                if all_messages:

                    print("initial kicker conditions saved")

                dac_generators.append(DACSignalGenerator(kicker(),
                                                        oscillator = os,
                                                        beamline = beamline,
                                                        **ov))
            
            
            M = MeshScan(dac_generators,
                         scan_variables,
                         scan_vectors,
                         wait_time,
                         all_messages = all_messages,
                         plot_display = display)
            """"""
            M.execute_scan(write_dac)


            if write_dac:


                for itr, kicker in kicker_devices:

                    kicker.write_dac(initial_kicker_conditions[itr])
            return M


def SignalGenerator(kicker_devices,
              oscillators,
              oscillator_variables,
              wait_time=0,
              write_dac = False,
              all_messages = True,
              beamline = '2',
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
    - write_dac: bool, optional (default=False)
        If True, restores DAC conditions after execution.
    - all_messages: bool, optional (default=True)
        Enables message display during execution.
    - beamline: str, optional (default='2')
        Specifies the beamline used for `Scan`.
    - kwargs: dict, optional
        Additional arguments passed to `Scan`.

    Returns:
    - MeshScan object generated by `Scan`.

    Example:
    ```
    signal_gen = SignalGenerator(kicker_devices, oscillators, oscillator_variables)
    ```
    """
    
    return Scan(kicker_devices,
                scan_vectors=[None for k in kicker_devices],
                oscillators = oscillators,
                oscillator_variables=oscillator_variables,
                scan_variables=["V7" for k in kicker_devices],
                write_dac = write_dac,
                all_messages = all_messages,
                display = False,
                beamline = beamline,
                wait_time= wait_time,
                *kwargs)