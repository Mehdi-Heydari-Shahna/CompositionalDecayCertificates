"""The shipped missions of the skid-steered teams: task data a base on conventional wheels can drive.

A mecanum base follows the payload with a fixed planar offset and a fixed heading, sideways where the
route says so.  A Husky cannot: its origin can only move along its heading, and sideways by turning
about a point of its centre line (``teamc/skid.py``).  These tests read the mission records alone.
"""
import numpy as np

from mission.route import compose


def steps(poses):
    """Per step of a pose path: displacement of the origin along and across the heading, and the heading change."""
    P = np.asarray(poses, float)
    d = np.diff(P[:, :2], axis=0)
    heading = .5 * (P[1:, 2] + P[:-1, 2])
    along = np.cos(heading) * d[:, 0] + np.sin(heading) * d[:, 1]
    across = -np.sin(heading) * d[:, 0] + np.cos(heading) * d[:, 1]
    return along, across, np.diff(P[:, 2])


def test_route_bases_keep_their_offset_from_the_payload_and_never_move_sideways(missions, skid_team):
    m = missions(skid_team)
    route = m['route']
    payload = np.asarray(route['samples'])
    assert set(route['bases']) == set(m['formation']['carry'])
    for rid, poses in route['bases'].items():
        poses = np.asarray(poses)
        offset = m['formation']['carry'][rid]
        origin = np.array([compose(p, [offset[0], offset[1], 0.])[:2] for p in payload])
        assert np.abs(poses[:, :2] - origin).max() < 1e-9                # the origin keeps its place in the formation
        along, across, turn = steps(poses)
        assert along.min() > 0.                                          # every base drives forwards all the way
        # sideways only by turning about a point of the centre line, within the range the skid model gave for the loads
        lo, hi = route['steering']['by_base'][rid]['turning_point_ahead_m']
        assert -.15 < lo < 0. < hi < .15
        defect = np.maximum(np.minimum(across + lo * turn, across + hi * turn), 0.) \
            - np.minimum(np.maximum(across + lo * turn, across + hi * turn), 0.)
        assert defect.max() < 2e-4                                       # m per step of about 5 cm
        assert np.abs(across).max() < 5e-3
        # the heading leaves that of the payload where the payload turns, and the base ends facing the supports
        relative = route['steering']['by_base'][rid]['relative_heading_rad']
        assert relative[0] < -.4 and abs(relative[1] - np.pi / 2) < 1e-6
        assert abs(poses[0, 2] - payload[0, 2]) < 1e-9                   # in line with the beam at the rack
        assert abs(poses[-1, 2] - payload[-1, 2] - m['formation']['place'][rid][2]) < 1e-6
    # a mecanum team of the same site has no base routes: its bases keep one offset and one heading
    assert 'bases' not in missions('ridgeback' + skid_team[-1])['route']


def test_bases_cross_the_ramp_edges_squarely(missions, compiled, skid_team):
    """No suspension: a base that met an edge of the ramp obliquely would stand on three wheels."""
    m = missions(skid_team)
    site = compiled(skid_team).scene['site']
    for rid, poses in m['route']['bases'].items():
        poses = np.asarray(poses)
        assert m['route']['steering']['by_base'][rid]['heading_at_ramp_ends_rad'] < .01
        for edge in (site['y_ramp0'], site['y_ramp1']):
            near = np.abs(poses[:, 1] - edge) < .3                       # both axles pass within this distance of the edge
            assert near.sum() > 5 and np.abs(poses[near, 2] - np.pi / 2).max() < .01


def test_approach_turns_on_the_spot_and_runs_straight_into_the_station(missions, skid_team):
    m = missions(skid_team)
    paths = m['base_paths']
    for rid, start in m['starts'].items():
        spin, way = (np.asarray(piece['samples']) for piece in paths['approach'][rid])
        dock = np.asarray(paths['dock'][rid][0]['samples'])
        x0 = paths['turning_centre_m'][rid]
        assert np.allclose(spin[0], start) and abs(x0) < .05
        pivot = spin[:, :2] + x0 * np.c_[np.cos(spin[:, 2]), np.sin(spin[:, 2])]
        assert np.abs(pivot - pivot[0]).max() < 1e-9                     # the turn on the spot is about the turning point
        assert np.abs(way[0] - spin[-1]).max() < 1e-6 and np.abs(dock[0] - way[-1]).max() < 1e-9
        for poses in (way, dock):
            along, across, turn = steps(poses)
            assert along.min() > 0. and np.abs(across + x0 * turn).max() < 2e-4
        # the last metre is a straight run along the rack, and it ends where the route starts
        assert np.ptp(dock[:, 1]) < 1e-12 and abs(dock[-1, 2]) < 1e-12
        assert abs(abs(dock[-1, 0] - dock[0, 0]) - paths['dock_distance_m'][rid]) < 1e-9
        assert np.allclose(dock[-1], m['route']['bases'][rid][0], atol=1e-9)


def test_bounds_and_controller_options_of_the_skid_steered_missions(missions, skid_team):
    m = missions(skid_team)
    assert m['bounds']['grasp_force_N'] == 85. and m['bounds']['grasp_moment_Nm'] == 10.     # the UR10e sensor, as for the Ridgeback
    assert 'resultant' in m['bounds']['note']                            # traction: net ground force, the wheels slide in every turn
    assert m['controller']['wheel_gain'] == 60. and m['controller']['wheel_integral'] == 2000.
    assert m['route']['base_yaw_rate'] == .15
    zones = np.asarray(m['route']['zones'])
    assert zones[:, 2].max() <= m['route']['cruise_speed'] and zones[-1, 2] == .15
