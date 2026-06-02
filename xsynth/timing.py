import re

import pydoocs
from matplotlib import pyplot as plt

def get_bunch_pattern():
    """
    gets the bunch pattern as specified by the bunch pattern builder control server
    """
    pattern_str = pydoocs.read("XFEL.UTIL/BUNCH_PATTERN/PATTERN_BUILDER/PATTERN_FINAL")['data']
    pattern_freq = pydoocs.read("XFEL.UTIL/BUNCH_PATTERN/PATTERN_BUILDER/PATTERN_FREQUENCY")['data']/1e3

    return pattern_str, pattern_freq

def pulse_index(beamline):
    """
    returns the index for all pulses in specified beam region
    """
    assert beamline in ["D", "1", "2", "3", "13", "4"]

    pattern_str, pattern_freq = get_bunch_pattern()
    
    index_= [match.start() for match in re.finditer(beamline, pattern_str)]


def pulse_time(beamline):
    """
    returns time-value for all pulses in specified beam region
    """
    _, pattern_freq = get_bunch_pattern()
    index = pulse_index(beamline)

    return index*(1/pattern_freq)+800

def plot_beam_regions():
    """
    sanity-check plot of beam regions
    """

    pattern_str, pattern_freq = get_bunch_pattern()
    
    beamlines = ["TLD", "SA1", "SA2", "SA3"]
    bc = ['grey', 'blue', 'orange', 'green']
    index_TLD= [match.start() for match in re.finditer("D", pattern_str)]
    index_SA1= [match.start() for match in re.finditer("1", pattern_str)]
    index_SA2= [match.start() for match in re.finditer("2", pattern_str)]
    index_SA3= [match.start() for match in re.finditer("3", pattern_str)]

    fig, ax = plt.subplots(1,1,figsize=(8, 2))

    for itr, item in enumerate([index_TLD, index_SA1, index_SA2, index_SA3]):

        ax.bar([800 + i*1/pattern_freq for i in item],[1]*len(item),
                width=0.1,
                align='center',
                alpha = 0.8,
                label = beamlines[itr],
                color = bc[itr])
        
    ax.set_xlim(800,)
    ax.set_xlabel("Time ($\mu$s)")
    ax.set_yticks([0,1], ["Off", "On"])
    ax.set_title("Beam Regions")
    ax.legend()

    plt.show()

def get_beam_regions():
    """
    return all beam regiohs and their start times
    """
    region_type = []
    region_start = []
    
    for itr in range(10):
        try:
            type_ = pydoocs.read(f"XFEL.UTIL/BUNCH_PATTERN/PATTERN_BUILDER/SUBPATTERN_{itr}/")['data']
            region_type.append(type_)
            start_ = pydoocs.read(f"XFEL.UTIL/BUNCH_PATTERN/PATTERN_BUILDER/SUBPATTERN_{itr}_START_TIME/")['data']
            region_start.append(start_)

        except Exception as e:
            pass
    
    return region_type, region_start

def get_region_bounds(beamline):
    """
    get the boundaries of a beam region (in microsecond) corresponding to specific type

    thus far only supports the first beam region of each type
    """
    
    assert beamline in ["D", "1", "2", "3", "13", "4"]

    region_type, region_start = get_beam_regions()
    region_type.append("2")
    idx = region_type.index(beamline)

    ti = region_start[idx]
    tf = region_start[idx+1]
    return ti, tf

