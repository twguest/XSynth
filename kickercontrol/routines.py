from kickercontrol.signal import DACSignalGenerator
from kickercontrol.scan import MeshScan
from kickercontrol.timing import get_region_bounds

def Scan(kicker_devices,
              scan_vectors,
              oscillator,
              oscillator_variables,
              scan_variables,
              wait_time=0.1,
              write_dac = False,
              all_messages = True,
              display = False,
              beamline = None,
                **kwargs):
            """
            Scan Routine 
            """
    
            initial_kicker_conditions = []
        
            dac_generators = []

            for kicker in kicker_devices:
                
                initial_kicker_conditions.append(kicker().read_dac()[:,1]) 
                
                if beamline is not None:
                    ti, tf = get_region_bounds(beamline)

                    if "V0" not in oscillator_variables:
                          oscillator_variables["V0"] = ti
                    if "V1" not in oscillator_variables:
                          oscillator_variables["V1"] = tf
                    if "V4" not in oscillator_variables:
                          oscillator_variables["V4"] = 1/(tf-ti)

                    

                if all_messages:

                    print("initial kicker conditions saved")

                dac_generators.append(DACSignalGenerator(kicker(),
                                                        oscillator = oscillator,
                                                        beamline = beamline,
                                                        **oscillator_variables))
            
            
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
