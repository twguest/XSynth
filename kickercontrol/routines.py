from kickercontrol.signal import DACSignalGenerator
from kickercontrol.scan import MeshScan


def MicroScan(kicker_devices,
              scan_vectors,
              oscillator,
              oscillator_variables,
              scan_variables,
              wait_time=0.1,
              write_dac = True,
              all_messages = True,
              display = False,
                **kwargs):
            """
            Scan Routine 
            """
    
            initial_kicker_conditions = []
        
            dac_generators = []

            for kicker in kicker_devices:
                
                initial_kicker_conditions.append(kicker().read_dac()[:,1]) 
                
                if all_messages:
                    print("initial kicker conditions saved")
                
                dac_generators.append(DACSignalGenerator(kicker(),
                                                        oscillator = oscillator,
                                                        **oscillator_variables))
            
            
            M = MeshScan(dac_generators, scan_variables, scan_vectors, wait_time, all_messages = all_messages, plot_display = display)
            
            M.execute_scan(write_dac)

            if write_dac:

                for itr, kicker in kicker_devices:
                    kicker.write_dac(initial_kicker_conditions[itr])

def MacroScan(kicker_devices,
              scan_vectors,
              wait_time=0.1,
              write_dac = True,
              all_messages = True,
              display = False,
              **kwargs):
            """
            Scan 
            """
    
            initial_kicker_conditions = []
        
            dac_generators = []

            for kicker in kicker_devices:
                
                initial_kicker_conditions.append(kicker().read_dac()[:,1]) 
                
                if all_messages:
                    print("initial kicker conditions saved")
                
                dac_generators.append(DACSignalGenerator(kicker(),
                                                        oscillator = 'line',
                                                        **kwargs))
            
            scan_variables = ["V2" for kicker in kicker_devices]
            
            M = MeshScan(dac_generators, scan_variables, scan_vectors, wait_time, all_messages = all_messages, plot_display = display)
            
            M.execute_scan(write_dac)

            if write_dac:

                for itr, kicker in kicker_devices:
                    kicker.write_dac(initial_kicker_conditions[itr])