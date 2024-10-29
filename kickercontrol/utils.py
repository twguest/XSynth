
def in_notebook():
    """
    checks if snippet is being run in a .ipynb
    """
    try:
        shell = get_ipython().__class__.__name__
        return shell == 'ZMQInteractiveShell'
    except NameError:
        return False