
import numpy as np

from kickercontrol.main import MacroScan
from kickercontrol.device import (KL2005, KMX1938, KMX1965, KNY1938, KNY1965)

MacroScan(kicker_devices=[KL2005],
          scan_vectors=[np.linspace(-.1,.1,25)],
                        write_dac=True,
                        display = True,
                        beamline = '2',
                        wait_time=1)
