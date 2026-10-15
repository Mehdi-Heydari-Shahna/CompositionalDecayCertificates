"""What the workflow needs of an Isaac Sim installation, looked up without building a scene.

The other stages meet a missing or renamed call when they reach it, one call per run.
This stage lists in one pass the modules, functions and classes the adapter uses and
whether each is there, with the versions of the runtime.  It simulates nothing, so it
says nothing about how the calls behave: the ``preflight`` stage does that.

Isaac and Omniverse modules are imported inside the functions, so this file can be
imported (and tested) without Isaac.
"""
from __future__ import annotations

import importlib
import inspect

# (module, name): the workflow cannot run without these.  ``name`` None: the module itself.
REQUIRED = (
    ('isaacsim', 'SimulationApp'),
    ('isaacsim.core.api', 'World'),
    ('isaacsim.core.simulation_manager', 'SimulationManager'),
    ('omni.usd', 'get_context'),
    ('omni.physics.tensors', 'create_simulation_view'),
    ('omni.physx', 'get_physx_simulation_interface'),
    ('pxr', 'Usd'), ('pxr', 'UsdGeom'), ('pxr', 'UsdPhysics'), ('pxr', 'UsdShade'), ('pxr', 'UsdLux'),
    ('pxr', 'Gf'), ('pxr', 'Sdf'), ('pxr', 'Vt'), ('pxr', 'PhysicsSchemaTools'),
    ('numpy', None), ('scipy', None),
    # SciPy, as the compiler, the controller, the audit and the scene generator import it
    ('scipy.linalg', 'qr'), ('scipy.linalg', 'lu_factor'), ('scipy.linalg', 'lu_solve'), ('scipy.linalg', 'expm'),
    ('scipy.spatial', 'ConvexHull'),
    ('scipy.spatial.transform', 'Rotation'), ('scipy.interpolate', 'CubicSpline'), ('scipy.interpolate', 'PchipInterpolator'),
    ('scipy.ndimage', 'gaussian_filter1d'), ('scipy.optimize', 'minimize'), ('scipy.signal', 'savgol_filter'),
)
# The workflow runs without these; what is lost is noted beside each.
OPTIONAL = (
    ('omni.physx', 'get_physx_interface', 'PhysX error events are not collected'),
    ('omni.physx', 'get_physx_cooking_interface', 'the preflight cannot compare the cooked roller hulls'),
    ('omni.kit.app', 'get_app', 'Kit and extension versions are not recorded'),
    ('isaacsim.core.version', 'get_version', 'the Isaac Sim version is not recorded'),
    ('isaacsim.core.utils.viewports', 'set_camera_view', 'no camera placement in a run with a window'),
    ('pxr', 'PhysxSchema', 'PhysX API schemas are applied as their tokens instead of through the bindings'),
    ('matplotlib', None, 'no figures in the Isaac environment (tools/make_report.py draws them elsewhere)'),
)
# Attributes the adapter uses of objects that exist before a scene does.  ``True``: required.
WORLD = dict(get_physics_context=True, reset=True, step=True, is_playing=True, stop=True, clear_instance=True,
             current_time=True, current_time_step_index=True, physics_sim_view=False)
SIMULATION_MANAGER = dict(get_active_physics_engine=False, switch_physics_engine=False, get_physics_simulation_view=False,
                          get_physics_sim_view=False)
SIMULATION_VIEW = dict(set_subspace_roots=True, create_articulation_view=True, create_rigid_body_view=True,
                       update_articulations_kinematic=True, get_gravity=False)
ARTICULATION_VIEW = dict(
    get_root_transforms=True, get_root_velocities=True, set_root_transforms=True, set_root_velocities=True,
    get_dof_positions=True, get_dof_velocities=True, set_dof_positions=True, set_dof_velocities=True,
    set_dof_actuation_forces=True, apply_forces_and_torques_at_position=True, get_link_transforms=True, get_coms=True,
    get_masses=True, get_inertias=True, get_dof_armatures=False, get_dof_stiffnesses=False, get_dof_dampings=False,
    get_dof_limits=False, get_dof_friction_properties=False, get_dof_friction_coefficients=False,
    get_link_incoming_joint_force=False)
RIGID_BODY_VIEW = dict(get_transforms=True, get_velocities=True, set_transforms=True, set_velocities=True,
                       apply_forces_and_torques_at_position=True, get_coms=True, get_masses=True, get_inertias=True,
                       set_kinematic_targets=False)
PHYSX_SCHEMA_APIS = ('PhysxSceneAPI', 'PhysxArticulationAPI', 'PhysxRigidBodyAPI', 'PhysxJointAPI', 'PhysxCollisionAPI',
                     'PhysxConvexHullCollisionAPI', 'PhysxMaterialAPI', 'PhysxContactReportAPI')   # backends/usd_export.PHYSX_APIS
TENSOR_CLASS_MODULES = ('omni.physics.tensors', 'omni.physics.tensors.impl.api', 'omni.physics.tensors.api')


def _lookup(module, name):
    """What ``from module import name`` gives (an attribute, or a submodule that import loads), or the reason it fails."""
    try:
        mod = importlib.import_module(module)
    except Exception as exc:                                         # noqa: BLE001 (an import can fail in many ways)
        return None, f'{type(exc).__name__}: {exc}'
    if name is None:
        return mod, None
    if hasattr(mod, name):
        return getattr(mod, name), None
    try:
        return importlib.import_module(f'{module}.{name}'), None
    except Exception as exc:                                         # noqa: BLE001
        return None, f'{module} has no {name} ({type(exc).__name__}: {exc})'


def _attributes(obj, wanted):
    return {name: bool(hasattr(obj, name)) for name in wanted}


def _missing(label, table, wanted):
    return [f'{label}.{name}' for name, required in wanted.items() if required and not table.get(name)]


def probe():
    """Look everything up.  Returns the report; ``passed`` is False when something required is missing."""
    report = dict(required=[], optional=[], attributes={}, notes=[])
    missing, missing_optional = [], []
    for module, name in REQUIRED:
        obj, error = _lookup(module, name)
        report['required'].append(dict(module=module, name=name, available=error is None, error=error))
        if error is not None:
            missing.append(module if name is None else f'{module}.{name}')
    for module, name, lost in OPTIONAL:
        obj, error = _lookup(module, name)
        report['optional'].append(dict(module=module, name=name, available=error is None, error=error, without_it=lost))
        if error is not None:
            missing_optional.append((module if name is None else f'{module}.{name}') + f' ({lost})')

    world, error = _lookup('isaacsim.core.api', 'World')
    if error is None:
        report['attributes']['World'] = table = _attributes(world, WORLD)
        missing += _missing('World', table, WORLD)
    manager, error = _lookup('isaacsim.core.simulation_manager', 'SimulationManager')
    if error is None:
        report['attributes']['SimulationManager'] = _attributes(manager, SIMULATION_MANAGER)

    # the classes of the tensor views, where the installation lets them be found without a scene
    found = {}
    for module in TENSOR_CLASS_MODULES:
        for name in ('SimulationView', 'ArticulationView', 'RigidBodyView'):
            if name not in found:
                obj, error = _lookup(module, name)
                if error is None and inspect.isclass(obj):
                    found[name] = (module, obj)
    for name, wanted in (('SimulationView', SIMULATION_VIEW), ('ArticulationView', ARTICULATION_VIEW), ('RigidBodyView', RIGID_BODY_VIEW)):
        if name in found:
            module, cls = found[name]
            report['attributes'][name] = table = dict(_attributes(cls, wanted), found_in=module)
            missing += _missing(name, table, wanted)
        else:
            report['attributes'][name] = None
            report['notes'].append(f'{name}: class not found without a scene; its methods are met in the preflight stage')
    art = report['attributes'].get('ArticulationView')
    if art and not (art.get('get_dof_friction_properties') or art.get('get_dof_friction_coefficients')):
        missing_optional.append('ArticulationView joint friction readback (the preflight skips that check)')

    create, error = _lookup('omni.physics.tensors', 'create_simulation_view')
    if error is None:
        try:
            report['create_simulation_view_parameters'] = list(inspect.signature(create).parameters)
        except (TypeError, ValueError) as exc:
            report['create_simulation_view_parameters'] = f'not inspectable: {exc}'

    getter, error = _lookup('omni.physx', 'get_physx_simulation_interface')
    if error is None:
        try:
            interface = getter()
            table = dict(subscribe_full_contact_report_events=callable(getattr(interface, 'subscribe_full_contact_report_events', None)),
                         subscribe_contact_report_events=callable(getattr(interface, 'subscribe_contact_report_events', None)))
            report['attributes']['physx simulation interface'] = table
            if not any(table.values()):
                missing.append('omni.physx simulation interface: no contact report subscription')
            elif not table['subscribe_full_contact_report_events']:
                missing_optional.append('full contact report (wheel traction of mecanum wheels is not compared with PhysX)')
        except Exception as exc:                                      # noqa: BLE001
            report['notes'].append(f'physx simulation interface: {type(exc).__name__}: {exc}')

    tools, error = _lookup('pxr', 'PhysicsSchemaTools')
    if error is None:
        report['attributes']['PhysicsSchemaTools'] = table = _attributes(tools, dict(sdfPathToInt=True, intToSdfPath=True))
        missing += _missing('PhysicsSchemaTools', table, dict(sdfPathToInt=True, intToSdfPath=True))
    schema, error = _lookup('pxr', 'PhysxSchema')
    if error is None:
        report['attributes']['PhysxSchema'] = table = _attributes(schema, PHYSX_SCHEMA_APIS)
        # backends/usd_export applies a schema the bindings lack as its token; whether PhysX reads it shows in the preflight
        missing_optional += [f'PhysxSchema.{name} (the schema is applied as its token)' for name, there in table.items() if not there]

    try:
        import omni.usd
        context = omni.usd.get_context()
        report['usd_context'] = dict(get_stage=callable(getattr(context, 'get_stage', None)),
                                     get_stage_id=callable(getattr(context, 'get_stage_id', None)))
        if not all(report['usd_context'].values()):
            missing.append('omni.usd context: get_stage / get_stage_id')
    except Exception as exc:                                          # noqa: BLE001
        report['notes'].append(f'omni.usd context: {type(exc).__name__}: {exc}')

    report.update(missing_required=missing, missing_optional=missing_optional, passed=not missing)
    return report
