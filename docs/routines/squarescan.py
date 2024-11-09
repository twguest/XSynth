
import numpy as np

from kickercontrol.main import SquareScan
from kickercontrol.device import (KL2005, KMX1938, KMX1965, KNY1938, KNY1965)

from kickercontrol.timing import get_region_bounds

ti, tf = get_region_bounds('2')

SquareScan(kicker_device = KL2005,
           scan_vector = np.linspace(0,10,10)/((tf-ti)),
           scan_variable="V4",
           amplitude = 0.1,
           duty = 0.5,
           n_frequency=2,
           wait_time = 1)
