"""Version-aware start-up, tensor binding and shutdown helpers for Isaac Sim.

Isaac and Omniverse modules are imported inside the functions, so this file can be
imported (and tested) without Isaac.  The helpers follow the lifecycle used by the
other RoboCompiler Isaac workflows: PhysX selected before the ``World`` exists, the
scene built on the stage of the USD context, ``World.reset()``, then a CPU/NumPy
tensor view bound to the explicit stage id.  Nothing here steps the simulation.
"""
from __future__ import annotations

import inspect
import sys
from numbers import Integral


def select_physx_engine():
    """Select PhysX before the World is created (Isaac versions with several physics engines)."""
    from isaacsim.core.simulation_manager import SimulationManager

    get_engine = getattr(SimulationManager, 'get_active_physics_engine', None)
    if get_engine is None:
        return dict(engine='physx', engine_selection='single-engine runtime')
    previous = get_engine()
    label = str(getattr(previous, 'value', previous)).lower()
    if 'physx' not in label:
        switch = getattr(SimulationManager, 'switch_physics_engine', None)
        if switch is None:
            raise RuntimeError(f'PhysX is required, but the active engine is {previous!r}')
        switch('physx')
    active = get_engine()
    if 'physx' not in str(getattr(active, 'value', active)).lower():
        raise RuntimeError(f'could not select PhysX; the active engine is {active!r}')
    return dict(engine=str(active), engine_selection='SimulationManager', previous_engine=str(previous))


def current_stage_id(expected_stage=None):
    """The id of the stage of the USD context (never the tensor API default of -1)."""
    import omni.usd

    context = omni.usd.get_context()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError('no USD stage is open; build the scene before binding tensors')
    if expected_stage is not None and stage != expected_stage:
        raise RuntimeError('the active USD stage changed during physics initialization')
    stage_id = context.get_stage_id()
    if isinstance(stage_id, bool) or not isinstance(stage_id, Integral) or stage_id <= 0:
        raise RuntimeError(f'invalid USD stage id {stage_id!r}')
    return int(stage_id)


def create_numpy_view(stage_id, world=None):
    """CPU/NumPy PhysX tensor view of the stage initialized by ``World.reset()``.

    First choice: a view created for the explicit stage id.  If that is refused, the view the
    World already owns is used, provided it is valid.  Returns ``(view, route)``.
    """
    import omni.physics.tensors as tensors

    if isinstance(stage_id, bool) or not isinstance(stage_id, Integral) or stage_id <= 0:
        raise ValueError('a positive, explicit USD stage id is required')
    kwargs = dict(stage_id=int(stage_id))
    try:
        parameters = inspect.signature(tensors.create_simulation_view).parameters
    except (TypeError, ValueError):
        parameters = {}
    if 'backend' in parameters:
        kwargs['backend'] = 'physx'
    first_error = None
    try:
        view = tensors.create_simulation_view('numpy', **kwargs)
        if view is not None:
            return view, 'omni.physics.tensors.create_simulation_view(numpy, explicit stage id)'
        first_error = 'the tensor API returned no view'
    except Exception as exc:                                         # noqa: BLE001 (reported below)
        first_error = f'{type(exc).__name__}: {exc}'
    candidates = []
    if world is not None:
        candidates.append(('World.physics_sim_view', getattr(world, 'physics_sim_view', None)))
    try:
        from isaacsim.core.simulation_manager import SimulationManager
        for name in ('get_physics_simulation_view', 'get_physics_sim_view'):
            getter = getattr(SimulationManager, name, None)
            if callable(getter):
                candidates.append(('SimulationManager.' + name, getter()))
    except Exception:                                                # noqa: BLE001
        pass
    for route, view in candidates:
        if view is None:
            continue
        check = getattr(view, 'check', None)
        if callable(check) and not bool(check()):
            continue
        return view, route + f' (explicit-id view failed: {first_error})'
    raise RuntimeError(f'could not bind a NumPy PhysX tensor view to USD stage {stage_id}: {first_error}. '
                       'The stage must contain the physics scene and must have been initialized with World.reset().')


def to_numpy(value):
    """NumPy array of a tensor returned by the PhysX tensor API (NumPy, Warp or Torch front end, CPU)."""
    import numpy as np

    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    if hasattr(value, 'numpy'):
        return np.asarray(value.numpy())
    return np.asarray(value)


def shutdown_configuration(app_type, mode='auto'):
    """Fast shutdown only where the application preserves an explicit exit code."""
    if mode not in ('auto', 'fast', 'graceful'):
        raise ValueError(f'unknown shutdown mode {mode!r}')
    try:
        supports_exit_code = 'exit_code' in inspect.signature(app_type.close).parameters
    except (TypeError, ValueError, AttributeError):
        supports_exit_code = False
    if mode == 'fast' and not supports_exit_code:
        raise RuntimeError('fast shutdown needs SimulationApp.close(exit_code=...); use --shutdown-mode auto or graceful')
    fast = mode != 'graceful' and supports_exit_code
    return dict(requested=mode, selected='fast' if fast else 'graceful', fast_shutdown=fast, supports_exit_code=supports_exit_code)


def close_simulation_app(app, exit_code, settle_updates=0):
    """Close the application.  ``settle_updates``: application updates before a graceful close, which the known
    issues of Isaac Sim 6.1.0 give as the workaround for threads that are not cleaned up on Windows."""
    for _ in range(int(settle_updates)):
        try:
            app.update()
        except Exception:                                            # noqa: BLE001 (closing goes on)
            break
    try:
        parameters = inspect.signature(app.close).parameters
    except (TypeError, ValueError):
        parameters = {}
    if 'exit_code' in parameters:
        app.close(exit_code=int(exit_code))
    else:
        app.close()


def cleanup_actions(actions):
    """Attempt every named cleanup even when an earlier one fails; returns what happened."""
    steps, errors = [], []
    for name, action in actions:
        try:
            action()
        except Exception as exc:                                     # noqa: BLE001
            message = f'{name}: {type(exc).__name__}: {exc}'
            errors.append(message)
            steps.append(dict(name=name, passed=False, error=message))
            print(f'[cleanup] {message}', file=sys.stderr, flush=True)
        else:
            steps.append(dict(name=name, passed=True))
    return dict(passed=not errors, steps=steps, errors=errors)


def version_info():
    """Versions of the runtime (best effort; never fatal)."""
    import importlib.metadata
    import platform

    info = dict(python=sys.version.split()[0], platform=platform.platform())
    for package in ('isaacsim', 'isaacsim-kernel', 'numpy', 'scipy'):
        try:
            info[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            info[package] = None
    try:
        from isaacsim.core.version import get_version
        info['isaac_version'] = list(get_version())
    except Exception:                                                # noqa: BLE001
        info['isaac_version'] = None
    try:
        import omni.kit.app
        app = omni.kit.app.get_app()
        info['kit_version'] = app.get_build_version()
        manager = app.get_extension_manager()
        info['physx_extension'] = manager.get_enabled_extension_id('omni.physx')
        info['tensors_extension'] = manager.get_enabled_extension_id('omni.physics.tensors')
        info['physx_tensors_extension'] = manager.get_enabled_extension_id('omni.physx.tensors')
    except Exception as exc:                                         # noqa: BLE001
        info['extension_version_error'] = str(exc)
    return info
