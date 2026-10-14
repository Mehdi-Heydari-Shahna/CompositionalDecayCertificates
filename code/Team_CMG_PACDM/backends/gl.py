"""Choice of MuJoCo's OpenGL backend (no MuJoCo import here).

``mujoco`` reads the environment variable ``MUJOCO_GL`` once, when it is imported.  A
script that renders on a machine without a display therefore has to choose the backend
before that import; a script that does not render is safest with OpenGL switched off,
because then no OpenGL library is loaded at all.
"""
from __future__ import annotations

import ctypes.util
import os
import sys


def prepare_mujoco_gl(render):
    """Set ``MUJOCO_GL`` if it is unset and ``mujoco`` is not imported yet.  Returns the backend in effect.

    A value already in the environment is never changed.  Without one: no rendering -> ``disable``;
    rendering on Linux without a display -> ``osmesa`` if the OSMesa library is installed, else ``egl``;
    otherwise MuJoCo's default (a window-system context).
    """
    current = os.environ.get('MUJOCO_GL')
    if current or 'mujoco' in sys.modules:
        return current or 'default'
    if not render:
        os.environ['MUJOCO_GL'] = 'disable'
    elif sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        os.environ['MUJOCO_GL'] = 'osmesa' if ctypes.util.find_library('OSMesa') else 'egl'
    return os.environ.get('MUJOCO_GL', 'default')
