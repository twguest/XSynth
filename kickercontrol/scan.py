import time
import warnings

from tqdm import tqdm
from datetime import datetime
import numpy as np
from copy import copy
from matplotlib import pyplot as plt
from kickercontrol.utils import in_notebook
from kickercontrol.timing import get_beam_regions
from IPython.display import display, clear_output

from scipy.spatial import cKDTree



class MiniScan:
    def __init__(self, dac_generators, scan_variables, scan_properties=None, wait_time=0.1, write_dac = False, plot_display = False, all_messages = False):
        """
        Initialize the MiniScan tool.

        Parameters:
        dac_generators (list of DACSignalGenerator): A list of DACSignalGenerator objects to scan.
        scan_variables (list of str): A list of variable names to update for each DACSignalGenerator.
        scan_properties (dict): A dictionary of scan properties (e.g., radii, center points, etc.).
        wait_time (float): Time in seconds to sleep between each scan point.
        """
        if len(dac_generators) == 0:
            raise ValueError("At least one DACSignalGenerator is required.")
        if len(dac_generators) != len(scan_variables):
            raise ValueError("The number of DACSignalGenerators must match the number of variable names.")
        
        self.all_messages = all_messages
        self.dac_generators = dac_generators
        self.scan_variables = scan_variables
        self.scan_properties = scan_properties or {}
        self.wait_time = wait_time
        self.plot_display = plot_display
   


    def generate_scan_points(self):
        """
        Generate the scan points based on the specific scan type and properties.
        Must be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement 'generate_scan_points' method.")
        self.scan_points = None

    def execute_scan(self, write_dac = False, plot_display = False):
        """
        Execute the scan by iterating through all scan points and writing to the DAC devices.
        """
        if self.scan_points is None:
            raise RuntimeError("Scan points have not been generated. Call 'generate_scan_points' first.")
        
        
        self.timestamps = []
        self.signals = []

        ### initialise signal gen
        for i, dac_generator in enumerate(self.dac_generators):
            init_point = self.scan_points[0]
            variable_name = self.scan_variables[i]
            dac_generator.update_variable(**{variable_name: init_point[i]})

        print(f"# Scan Points: {len(self.scan_points)}")
        
        t_start = datetime.now()
        
        for k, point in tqdm(enumerate(self.scan_points)):
            
            
            S = [] ## Signal Container

            for i, dac_generator in enumerate(self.dac_generators):
                
                
                # Update signal parameter for each DAC generator based on the scan point
                variable_name = self.scan_variables[i]
                dac_generator.update_variable(**{variable_name: point[i]})

                S.append(dac_generator.generated_signal)

                if self.plot_display:

                    if k == 0 and i == 0:

                        if self.all_messages:
                            print("Setting Up Display")

                        assert in_notebook() == True, "Display is only possible in Jupyter Notebooks"

                        lines = {}
                        ins_lines = {}

                        fig, ax = plt.subplots(figsize = (8,6))

                        def plot_beam_regions(ax):

                            regions, region_timing = get_beam_regions()
                            for itr, r in enumerate(regions):

                                if r == 'D':
                                    color = 'grey'
                                if r == '1':
                                    color = 'blue'
                                if r == '2':
                                    color = 'orange'
                                if r == '3':
                                    color = 'green'
                                if "13" in r:
                                    color = 'magenta'
                                
                                try:

                                    ax.axvspan(region_timing[itr], region_timing[itr+1], color = color, alpha = 0.2)

                                except Exception as e:
                                    break
                            
                            return regions, region_timing
                        
                        ins_ax = ax.inset_axes([.55, .65, 0.4, 0.3])
                        ins_ax.patch.set_alpha(0.5)

                        for a in [ax, ins_ax]:
                            
                            regions, region_timing = plot_beam_regions(a)
                            
                            a.set_xlabel("Time (us)")
                            
                        #ins_ax.set_ylim(-32767,32767)
                        ins_ax.set_ylim(32767-32767*0.125,32767+32767*0.125)
                        ins_ax.set_yticks([32767-32767*0.125,32767+32767*0.125])
                        
                        ins_ax.set_xlim(700, region_timing[-1]+75)
        
                        ax.set_xlim(dac_generator.signal_params["V0"]-75,
                                    dac_generator.signal_params["V1"]+75)                        

                        ax.set_ylabel("DAC Signal (16-bit Signed Integer)")

                        
                        tax = ax.twinx()
                        
                        tax.set_ylabel("DAC Signal (Volts.)")

                        vmax = 1 ### horrible hard coded voltage max for main display
                        ax.set_ylim(0*vmax, 2*32767*vmax)
                        #ax.set_ylim(-32767*vmax, 32767*vmax)
                        tax.set_ylim(-vmax, vmax)

                        ### init plot
                        for itr, dac_generator in enumerate(self.dac_generators):
                            ins_lines[str(itr)], = ins_ax.plot(dac_generator.t, dac_generator.generated_signal)
                            lines[str(itr)], = ax.plot(dac_generator.t, dac_generator.generated_signal, label = dac_generator.kicker.__name__)

                            clear_output(wait=True)
                            display(fig)

                        lines = [lines[str(itr)] for itr in range(len(self.dac_generators))]
                        print(lines)
                        ins_lines = [ins_lines[str(itr)] for itr in range(len(self.dac_generators))]    
                        ax.legend(loc = 'lower right')
                    
                    lines[i].set_ydata(dac_generator.generated_signal)
                    ins_lines[i].set_ydata(dac_generator.generated_signal)
                    
                    
                    clear_output(wait=True)
                    display(fig)

                if i == len(self.dac_generators)-1:    
                    itr_start = datetime.now()
                if write_dac:
                    try:
                        dac_generator.write_dac_signal()
                    except Exception as e:
                        print(e)
                        break
            
            itr_end = datetime.now()
            itr_delta = (itr_end-itr_start).seconds
            # Sleep for the specified amount of time between each scan point
            try:
                time.sleep(self.wait_time-itr_delta-1) ## not sure why -1
            except ValueError:
                break


            t_update = datetime.now()
            
            self.timestamps.append(t_update-t_start)
            self.signals.append(S)

        if self.plot_display:
            clear_output(wait=True)

    @property
    def scan_points(self):
        return self.generate_scan_points()

class SpiralScan(MiniScan):
    def __init__(self, dac_generators, scan_variables, x_center, y_center, x_radius, y_radius, num_turns=4, N=1000, wait_time=0.1):
        scan_properties = {
            'x_center': x_center,
            'y_center': y_center,
            'x_radius': x_radius,
            'y_radius': y_radius,
            'num_turns': num_turns,
            'N': N
        }
        super().__init__(dac_generators, scan_variables, scan_properties, wait_time)

    def generate_scan_points(self):
        """
        Generate the spiral scan points based on the specified properties.
        """
        N = self.scan_properties['N']
        x_center = self.scan_properties['x_center']
        y_center = self.scan_properties['y_center']
        x_radius = self.scan_properties['x_radius']
        y_radius = self.scan_properties['y_radius']
        num_turns = self.scan_properties['num_turns']

        if len(self.dac_generators) != 2:
            raise ValueError("Spiral scan requires exactly 2 DACSignalGenerators.")

        # Generate theta values for the spiral
        theta = np.linspace(0, num_turns * 2 * np.pi, N)

        # Generate radius values that grow linearly with theta
        r = np.linspace(0, 1, N)

        # Generate the spiral points in polar coordinates and convert to Cartesian coordinates
        x = x_center + r * x_radius * np.cos(theta)
        y = y_center + r * y_radius * np.sin(theta)

        # Combine x and y coordinates into an Nx2 array
        scan_points = np.vstack((x, y)).T

        # Optimize order of the scan points with respect to minimum distance between points
        tree = cKDTree(self.scan_points)
        _, optimal_order = tree.query(self.scan_points[0], k=N)
        return scan_points[optimal_order]


class MeshScan(MiniScan):

    def __init__(self, dac_generators, scan_variables, scan_vectors, wait_time=0.1, **kwargs):
        scan_properties = {'scan_vectors': scan_vectors}
        super().__init__(dac_generators, scan_variables, scan_properties, wait_time, **kwargs)

    def generate_scan_points(self):
        """
        Generate the mesh scan points based on the specified scan_vectors.
        """
        M = len(self.dac_generators)
        if len(self.scan_properties['scan_vectors']) != M:
            raise ValueError("The number of scan_vectors must match the number of DACSignalGenerators.")

        scan_vectors = self.scan_properties['scan_vectors']

        # Generate the meshgrid from the input scan_vectors
        mesh = np.meshgrid(*scan_vectors, indexing='ij')

        # Flatten the meshgrid and combine coordinates into an array of shape (M, len(vectors))
        return np.vstack([m.flatten() for m in mesh]).T


if __name__ == '__main__':
    
    scan_output = Scan([KL2005],
            scan_vectors=[np.arange(25)/25],
            oscillator = 'sin',
            oscillator_variables={"V2": 0},
            scan_variables=["V3"],
            write_dac = False,
            all_messages=False,
            display = True,
            beamline = '2',
            wait_time=0.1)
