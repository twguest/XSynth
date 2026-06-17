import time
import warnings

from tqdm import tqdm
from datetime import datetime
import numpy as np
from copy import copy
from matplotlib import pyplot as plt
from xsynth.utils import in_notebook
from xsynth.timing import get_beam_regions
from IPython.display import display, clear_output

from scipy.spatial import cKDTree



class MiniScan:
    def __init__(self, generators, scan_variables, scan_properties=None, fix_beam_region = False, wait_time=0.1, write = False, plot_display = False, all_messages = False):
        """
        Initialize the MiniScan tool.

        Parameters:
        generators (list of DACSignalGenerator): A list of DACSignalGenerator objects to scan.
        scan_variables (list of str): A list of variable names to update for each DACSignalGenerator.
        scan_properties (dict): A dictionary of scan properties (e.g., radii, center points, etc.).
        wait_time (float): Time in seconds to sleep between each scan point.
        """
        if len(generators) == 0:
            raise ValueError("At least one DACSignalGenerator is required.")
        if len(generators) != len(scan_variables):
            raise ValueError("The number of DACSignalGenerators must match the number of variable names.")
        
        self.all_messages = all_messages
        self.generators = generators
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

    def execute_scan(self, write = False, plot_display = False):
        """
        Execute the scan by iterating through all scan points and writing to the DAC devices.
        """
        if plot_display:
            self.plot_display = True

        scan_points = np.asarray(self.generate_scan_points(), dtype=float)
        if scan_points.size == 0:
            raise RuntimeError("Scan points have not been generated. Call 'generate_scan_points' first.")
        
        
        self.timestamps = []
        self.signals = []
        self.scan_points_executed = scan_points

        ### initialise signal gen
        for i, generator in enumerate(self.generators):
            init_point = scan_points[0]
            variable_name = self.scan_variables[i]
            generator.update_variable(**{variable_name: init_point[i]})

        print(f"# Scan Points: {len(scan_points)}")
        
        t_start = datetime.now()
        
        for k, point in tqdm(enumerate(scan_points), total=len(scan_points)):
            point_start = time.time()
            
            
            S = [] ## Signal Container

            for i, generator in enumerate(self.generators):
                
                
                # Update signal parameter for each DAC generator based on the scan point
                variable_name = self.scan_variables[i]
                generator.update_variable(**{variable_name: point[i]})

                signal_values = np.asarray(generator.signal).copy()
                S.append(signal_values)

                if self.plot_display:

                    if k == 0 and i == 0:

                        if self.all_messages:
                            print("Setting Up Display")

                        assert in_notebook() == True, "Display is only possible in Jupyter Notebooks"

                        lines = {}
                        ins_lines = {}

                        fig, ax = plt.subplots(figsize = (8,6))

                            
                        ax.set_xlabel(self.generators[0].unit)
                            
                        #ins_ax.set_ylim(-32767,32767)
                        
                        
                        #ins_ax.set_xlim(700, region_timing[-1]+75)
        
                        # ax.set_xlim(generator.signal_params["V0"]-75,
                        #             generator.signal_params["V1"]+75)                        

                        ax.set_ylabel("Signal")

                        
                        #tax = ax.twinx()
                        
                        #tax.set_ylabel("DAC Signal (Volts.)")

                        #vmax = 1 ### horrible hard coded voltage max for main display
                        #tax.set_ylim(-vmax, vmax)

                        ### init plot
                        for itr, generator in enumerate(self.generators):
                            server = getattr(generator, "server", getattr(generator, "kicker", None))
                            label = getattr(server, "__name__", f"Generator {itr + 1}")
                            
                            lines[str(itr)], = ax.plot(
                                np.asarray(generator.t),
                                np.asarray(generator.signal),
                                label=label,
                            )
                            clear_output(wait=True)
                            display(fig)

                        lines = [lines[str(itr)] for itr in range(len(self.generators))]
                        
                        ax.legend(loc = 'lower right')
                    
                    lines[i].set_data(np.asarray(generator.t), np.asarray(generator.signal))
                    ax.relim()
                    ax.autoscale(enable=True, axis="both")
                    ax.autoscale_view()
                                        
                    clear_output(wait=True)
                    display(fig)

                if write:
                    try:
                        generator.write()
                    except Exception as e:
                        print(e)
                        raise
            
            # Sleep for the specified amount of time between each scan point
            sleep_time = float(self.wait_time) - (time.time() - point_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


            t_update = datetime.now()
            
            self.timestamps.append(t_update-t_start)
            self.signals.append(S)

        if self.plot_display:
            clear_output(wait=True)

    @property
    def scan_points(self):
        return self.generate_scan_points()

class SpiralScan(MiniScan):
    def __init__(self, generators, scan_variables, x_center, y_center, x_radius, y_radius, num_turns=4, N=1000, wait_time=0.1):
        scan_properties = {
            'x_center': x_center,
            'y_center': y_center,
            'x_radius': x_radius,
            'y_radius': y_radius,
            'num_turns': num_turns,
            'N': N
        }
        super().__init__(generators, scan_variables, scan_properties, wait_time)

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

        if len(self.generators) != 2:
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

    def __init__(self, generators, scan_variables, scan_vectors, wait_time=0.1, **kwargs):
        scan_properties = {'scan_vectors': scan_vectors}
        super().__init__(
            generators,
            scan_variables,
            scan_properties,
            wait_time=wait_time,
            **kwargs,
        )

    def generate_scan_points(self):
        """
        Generate the mesh scan points based on the specified scan_vectors.
        """
        M = len(self.generators)
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
            write = False,
            all_messages=False,
            display = True,
            beamline = '2',
            wait_time=0.1)
