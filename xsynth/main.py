from xsynth.signal import DACSignalGenerator, ADAPTSignalGenerator
from xsynth.scan import MeshScan
from xsynth.device import ADAPT_MLS
import inspect
import numpy as np


def _instantiate_scan_device(device, beam_region):
    """Instantiate a scan device, passing beam_region only if accepted."""

    if not callable(device):
        return device

    signature = inspect.signature(device)
    beam_region_param = signature.parameters.get("beam_region")
    if beam_region_param is not None:
        if beam_region is not None:
            return device(beam_region=beam_region)

        if beam_region_param.default is inspect.Parameter.empty:
            raise ValueError(
                f"{getattr(device, '__name__', device)!r} requires beam_region."
            )

        return device()

    return device()


def _as_sequence(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _expand_to_length(value, length, name):
    values = _as_sequence(value)
    if len(values) == 1:
        return values * length
    if len(values) != length:
        raise ValueError(f"{name} must have length 1 or match the number of devices.")
    return values


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
                Beam region used when constructing ADAPT/IBFB device servers.
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

                Device = _instantiate_scan_device(device, beam_region)

                if Device.device_type == 'DAC':
                     generator = DACSignalGenerator(
                         Device,
                         oscillator=os,
                         relative_scan=relative_scan,
                         **ov,
                         **kwargs,
                     )
                elif Device.device_type == 'ADAPT':
                     generator = ADAPTSignalGenerator(
                         Device,
                         oscillator=os,
                         **ov,
                         **kwargs,
                     )
                elif Device.device_type == 'ADAPT_MLS':
                     generator = ADAPTSignalGenerator(
                         Device,
                         oscillator=os,
                         **ov,
                         **kwargs,
                     )
                elif Device.device_type == 'IBFB':
                     generator = ADAPTSignalGenerator(
                         Device,
                         oscillator=os,
                         **ov,
                         **kwargs,
                     )
                else:
                     raise ValueError(f"Unsupported device type: {Device.device_type}")

                generators.append(generator)

            
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


def SetDevice(
    devices,
    value,
    beam_region=None,
    wait_time=0,
    write=True,
    all_messages=True,
    display=False,
    relative_scan=False,
    **kwargs,
):
    """Set one or more ADAPT/IBFB devices to a constant value.

    The beam region is configured on the device/server. The signal generator
    uses a line oscillator over that device-owned beam region, and the scan has
    a single point per device.
    """

    devices = _as_sequence(devices)
    values = _expand_to_length(value, len(devices), "value")

    return Scan(
        devices=devices,
        scan_vectors=[np.asarray([v], dtype=float) for v in values],
        oscillators=["line" for _ in devices],
        oscillator_variables=[{} for _ in devices],
        scan_variables=["V2" for _ in devices],
        wait_time=wait_time,
        write=write,
        all_messages=all_messages,
        display=display,
        beam_region=beam_region,
        relative_scan=relative_scan,
        **kwargs,
    )


def SetADAPT(
    device=None,
    value=None,
    beam_region=None,
    server=None,
    **kwargs,
):
    """Compatibility wrapper for setting an ADAPT-like device constant."""

    if device is None:
        device = server
    if device is None:
        raise ValueError("SetADAPT requires a device or server.")
    if value is None:
        raise ValueError("SetADAPT requires a value.")
    if beam_region is None:
        beam_region = getattr(device, "beam_region", None)

    return SetDevice(
        devices=device,
        value=value,
        beam_region=beam_region,
        **kwargs,
    )


def SetADAPTMLS(
    value,
    beam_region="SA2",
    devices=ADAPT_MLS,
    **kwargs,
):
    """Set the ADAPT middle-layer device to a constant value."""

    return SetDevice(
        devices=devices,
        value=value,
        beam_region=beam_region,
        **kwargs,
    )
