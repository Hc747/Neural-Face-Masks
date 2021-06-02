from enum import Enum

"""
A module exporting valid states for various components of the application to be in.
"""


class State(Enum.int):
    """
    The different state representations that a component may be in.
    """
    UNINITIALISED = 0  # valid when the component is uninitialised
    INTERMEDIATE = 1  # valid when the component is transitioning from uninitialised to running
    RUNNING = 2  # valid when the component is fully initialised
