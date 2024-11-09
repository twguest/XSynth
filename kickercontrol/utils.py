import numpy as np

def in_notebook():
    """
    checks if snippet is being run in a .ipynb
    """
    try:
        shell = get_ipython().__class__.__name__
        return shell == 'ZMQInteractiveShell'
    except NameError:
        return False
    


def float_to_16bit_int(array):
    """
    Convert an array of float values in the range [-1, 1] to 16-bit signed integers.

    Parameters:
    array (numpy.ndarray): Array of float values in the range [-1, 1].

    Returns:
    numpy.ndarray: Array of 16-bit signed integers.
    """
    # Ensure the values are within the range [-1, 1]
    array = np.clip(array, -1, 1)
    # Scale and map to the range of 16-bit integers
    int_array = np.round(array * 32767).astype(np.int16)
    return int_array


def float_to_16bit_unsigned_int(array):
    """
    Convert an array of float values in the range [-1, 1] to 16-bit unsigned integers in the range [0, 65536].
    
    Parameters:
    array (numpy.ndarray): Array of float values in the range [-1, 1].
    
    Returns:
    numpy.ndarray: Array of unsigned 16-bit integers in the range [0, 65536].
    """
    # Ensure the values are within the range [-1, 1]
    array = np.clip(array, -1, 1)
    # Scale and map to the range of 16-bit unsigned integers [0, 65536]
    int_array = np.round((array + 1) * 32768).astype(np.uint16)
    return int_array
