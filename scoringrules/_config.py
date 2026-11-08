"""Global configuration: ``set_config``, ``get_config`` and ``config_context``.

There is one process-wide configuration and an optional override per context
(a ``contextvars.ContextVar``), set by ``config_context``. An asyncio task
starts with a copy of the override of the code that created it. A new thread
starts without it (unless the interpreter makes threads inherit the context,
``sys.flags.thread_inherit_context``, the default on free-threaded builds), so
work sent to other threads from inside a ``config_context`` block (a
``ThreadPoolExecutor``, dask's threaded scheduler) sees the process-wide values:
use ``set_config`` for those. Other processes only read ``SCORINGRULES_NUMBA``.
As with ``decimal.localcontext``, a context opened in a generator that yields
inside the block applies to the generator's caller until the generator resumes
and exits it.

The only option is ``numba``, whose initial process-wide value is read from the
``SCORINGRULES_NUMBA`` environment variable when scoringrules is imported."""

import contextlib
import os
import warnings
from contextvars import ContextVar

_NUMBA_VALUES = ("auto", True, False)
_NUMBA_ENV = "SCORINGRULES_NUMBA"
_NUMBA_ENV_VALUES = {
    "": "auto",
    "auto": "auto",
    "1": True,
    "true": True,
    "0": False,
    "false": False,
}


def _numba_from_env():
    """The ``numba`` value given by ``SCORINGRULES_NUMBA`` (``"auto"`` if unset).

    Case and surrounding whitespace are ignored. An unknown value warns and
    gives ``"auto"``: the import must not fail because of the environment."""
    raw = os.environ.get(_NUMBA_ENV)
    if raw is None:
        return "auto"
    value = _NUMBA_ENV_VALUES.get(raw.strip().lower())
    if value is None:
        warnings.warn(
            f"Ignoring {_NUMBA_ENV}={raw!r}: expected one of 0, 1, auto, false or "
            "true. Using numba='auto'.",
            UserWarning,
            stacklevel=2,
        )
        return "auto"
    return value


# the process-wide values, changed by ``set_config`` outside a ``config_context``
_global_config = {"numba": _numba_from_env()}
# the overridden options of the current context (``None``: no override)
_local_config = ContextVar("scoringrules_config", default=None)


def _numba_mode():
    """The current ``numba`` value (``"auto"``, ``True`` or ``False``)."""
    local = _local_config.get()
    if local is not None and "numba" in local:
        return local["numba"]
    return _global_config["numba"]


def _validated(numba):
    """The options passed to ``set_config``/``config_context``, validated."""
    options = {}
    if numba is not None:
        # ``is`` for the booleans: 1, 0 and np.bool_ are refused
        if not (
            numba is True or numba is False or (type(numba) is str and numba == "auto")
        ):
            raise ValueError(f"numba must be 'auto', True or False, got {numba!r}.")
        options["numba"] = numba
    return options


def get_config():
    """The current configuration of scoringrules.

    Returns
    -------
    config : dict
        A new dict with the value of every option, taking the innermost active
        ``config_context`` into account. Changing it has no effect; use
        ``set_config`` or ``config_context``.

    See Also
    --------
    set_config : Change the configuration.
    config_context : Change the configuration within a ``with`` block.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.get_config()["numba"]
    'auto'
    """
    local = _local_config.get()
    return {**_global_config, **(local or {})}


def set_config(*, numba=None):
    """Change the configuration of scoringrules.

    Outside a ``config_context`` this sets the process-wide value, seen by every
    thread and task without a ``config_context`` of its own. Inside a
    ``config_context`` it changes the value of that context only, and the
    change is undone when the context exits.

    Parameters
    ----------
    numba : {"auto", True, False}, optional
        When the numba gufuncs compute the scores that have one:

        - ``"auto"`` (default): for numpy inputs when numba is installed, except
          for a few score and input combinations where the array path is
          faster or more accurate;
        - ``True``: for numpy inputs, always. A score call that would use a
          gufunc raises ``ImportError`` if numba cannot be imported;
        - ``False``: never.

        Inputs other than plain numpy data (jax and torch arrays, masked
        arrays, float16, ...) always take the array path. ``None`` leaves the
        value unchanged. The initial value is read from the environment
        variable ``SCORINGRULES_NUMBA`` (``0``, ``1``, ``auto``, ``false`` or
        ``true``) when scoringrules is imported.

    Raises
    ------
    ValueError
        If ``numba`` is not one of the values above.

    See Also
    --------
    get_config : The current configuration.
    config_context : Change the configuration within a ``with`` block.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.set_config(numba=False)
    >>> sr.get_config()["numba"]
    False
    >>> sr.set_config(numba="auto")
    """
    options = _validated(numba)
    local = _local_config.get()
    if local is None:
        _global_config.update(options)
    else:
        _local_config.set({**local, **options})


@contextlib.contextmanager
def config_context(*, numba=None):
    """Change the configuration of scoringrules within a ``with`` block.

    The values apply to the code in the block, including the asyncio tasks it
    creates, and are restored when the block exits, also on an exception.
    Other threads and tasks keep their own configuration. Contexts can be
    nested, and ``set_config`` inside a context changes only that context.
    Also usable as a function decorator.

    Work sent to other threads from inside the block (a ``ThreadPoolExecutor``,
    dask's threaded scheduler) does not see the context, except on interpreters
    whose threads inherit it (``sys.flags.thread_inherit_context``): use
    ``set_config`` for such work. Spawned processes only read the
    ``SCORINGRULES_NUMBA`` environment variable. A context opened in a
    generator that yields inside the block leaks to the generator's caller, as
    with ``decimal.localcontext``.

    Parameters
    ----------
    numba : {"auto", True, False}, optional
        See ``set_config``. ``None`` keeps the current value.

    Raises
    ------
    ValueError
        If ``numba`` is not one of the accepted values.

    See Also
    --------
    get_config : The current configuration.
    set_config : Change the configuration.

    Examples
    --------
    >>> import numpy as np
    >>> import scoringrules as sr
    >>> obs, fct = np.zeros(3), np.ones((3, 5))
    >>> with sr.config_context(numba=False):  # the array path
    ...     sr.crps_ensemble(obs, fct)
    array([1., 1., 1.])
    """
    options = _validated(numba)
    local = _local_config.get()
    token = _local_config.set({**(local or {}), **options})
    try:
        yield
    finally:
        _local_config.reset(token)
