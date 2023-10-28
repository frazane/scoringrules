from .registry import BackendsRegistry

backends = BackendsRegistry()

try:
    import numba 
    _NUMBA_IMPORTED = True
except:
    _NUMBA_IMPORTED = False