"""The OpenGL backend of MuJoCo is chosen before ``mujoco`` is imported, and never against the user's own choice."""
import ctypes.util
import os
import sys

import pytest

from backends import gl


@pytest.fixture()
def clean(monkeypatch):
    """No MUJOCO_GL, no display, ``mujoco`` not imported; the environment is restored afterwards."""
    saved = {k: os.environ.get(k) for k in ('MUJOCO_GL', 'DISPLAY', 'WAYLAND_DISPLAY')}
    for k in saved:
        os.environ.pop(k, None)
    monkeypatch.delitem(sys.modules, 'mujoco', raising=False)
    monkeypatch.setattr(sys, 'platform', 'linux')
    yield monkeypatch
    for k, v in saved.items():
        os.environ.pop(k, None)
        if v is not None:
            os.environ[k] = v


def test_no_rendering_switches_opengl_off(clean):
    assert gl.prepare_mujoco_gl(render=False) == 'disable' and os.environ['MUJOCO_GL'] == 'disable'


@pytest.mark.parametrize('library, expected', [('libOSMesa.so.8', 'osmesa'), (None, 'egl')])
def test_headless_rendering_takes_osmesa_if_installed_else_egl(clean, library, expected):
    clean.setattr(ctypes.util, 'find_library', lambda name: library if name == 'OSMesa' else None)
    assert gl.prepare_mujoco_gl(render=True) == expected and os.environ['MUJOCO_GL'] == expected


def test_a_display_keeps_the_default_backend(clean):
    os.environ['DISPLAY'] = ':0'
    assert gl.prepare_mujoco_gl(render=True) == 'default' and 'MUJOCO_GL' not in os.environ


def test_the_users_choice_and_an_imported_mujoco_are_left_alone(clean):
    os.environ['MUJOCO_GL'] = 'egl'
    assert gl.prepare_mujoco_gl(render=False) == 'egl' and os.environ['MUJOCO_GL'] == 'egl'
    del os.environ['MUJOCO_GL']
    clean.setitem(sys.modules, 'mujoco', object())                 # too late to choose: nothing is changed
    assert gl.prepare_mujoco_gl(render=True) == 'default' and 'MUJOCO_GL' not in os.environ
