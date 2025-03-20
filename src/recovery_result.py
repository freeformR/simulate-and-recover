from typing import Tuple
from .ez_diffusion import validate_observed_stats
from .recovery import recover_parameters

class RecoveryResult: 
    """Encapsulates recovered parameters and observed data"""
    """The initial structure of this file was suggested by copilot"""
    
    def __init__(self, data: Tuple[float, float, float]):
        self.data = data  # Triggers validation via setter

    @property
    def data(self) -> Tuple[float, float, float]:
        """Observed statistics (R_obs, M_obs, V_obs)"""
        return self._data

    @data.setter  # Now properly attached to the property. Property Decorator patterns suggested by copilot 
    def data(self, new_data: Tuple[float, float, float]):
        """Set new data with validation"""
        try:
            # Allow V_obs=0 for N=1 cases
            validate_observed_stats(*new_data, allow_zero_v=True)
        except ValueError as e:
            if "V_obs" not in str(e):  # Only re-raise non-V_obs errors
                raise
        self._data = new_data
        self._recover_parameters()
        
    #Exception handeling suggested by copilot 
    def _recover_parameters(self): 
        """Recover parameters and validate bounds"""
        try:
            self._nu, self._a, self._t = recover_parameters(*self._data)
            self._validate_parameters()
        except Exception as e:
            raise RuntimeError(f"Recovery failed: {str(e)}")
    
    def _validate_parameters(self):
        """Ensure parameters stay within valid ranges"""
        if not (0.5 <= self._a <= 2):
            raise ValueError(f"Invalid boundary separation: {self._a}")
        if not (0.5 <= self._nu <= 2):
            raise ValueError(f"Invalid drift rate: {self._nu}")
        if not (0.1 <= self._t <= 0.5):
            raise ValueError(f"Invalid non-decision time: {self._t}")
    
    @property
    def drift_rate(self) -> float:
        """Estimated drift rate (ν)"""
        return self._nu
    
    @property
    def boundary_separation(self) -> float:
        """Estimated boundary separation (a)"""
        return self._a
    
    @property
    def non_decision_time(self) -> float:
        """Estimated non-decision time (τ)"""
        return self._t