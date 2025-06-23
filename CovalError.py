"""
Stub implementation of CovalError for local testing.
"""

STATUS_COMPLETED = "completed"


class SimulationInitialSetupError(Exception):
    """Error raised during initial setup of simulation"""
    pass


class SimulationAPIError(Exception):
    """Error raised during API calls in simulation"""
    pass


class SimulationDatabaseError(Exception):
    """Error raised during database operations in simulation"""
    pass 