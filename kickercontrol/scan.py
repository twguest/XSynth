import time

import numpy as np

from matplotlib import pyplot as plt
from kickercontrol.utils import in_notebook
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


            
        plotted_signal = []

        ### initialise signal gen
        for i, dac_generator in enumerate(self.dac_generators):
            init_point = self.scan_points[0]
            variable_name = self.scan_variables[i]
            dac_generator.update_variable(**{variable_name: init_point[i]})

        for k, point in enumerate(self.scan_points):
            for i, dac_generator in enumerate(self.dac_generators):
                
                # Update signal parameter for each DAC generator based on the scan point
                variable_name = self.scan_variables[i]
                dac_generator.update_variable(**{variable_name: point[i]})

                if self.plot_display:
                    
                    if k == 0:
                
                        assert in_notebook() == True, "Display is only possible in Jupyter Notebooks"

                        lines = []

                        fig, ax = plt.subplots()
                        ax.set_ylim(-1,1)

                        for dac_generator in self.dac_generators:
                            l, = ax.plot(dac_generator.t, dac_generator.generated_signal)
                            lines.append(l)
                        
                        display(fig)
                
                    lines[i].set_ydata(dac_generator.generated_signal.values)
                    clear_output(wait=True)
                    display(fig)




                if write_dac:
                    dac_generator.write_dac_signal()

            # Sleep for the specified amount of time between each scan point
            time.sleep(self.wait_time)
        
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



