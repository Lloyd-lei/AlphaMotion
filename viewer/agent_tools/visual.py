"""Headless CPU views of the very same compiled meshes and FK as the web UI."""
from __future__ import annotations

import io
import numpy as np

VIEWS = {'front': (12, -90), 'side': (12, 0), 'top': (90, -90), 'iso': (22, -55)}


def world_points(scene, chunk):
    ids = np.asarray(scene['surface_geom'], dtype=int)
    return np.einsum('npij,pj->npi', chunk['geom_rot'][:, ids], scene['surface_points']) + chunk['geom_pos'][:, ids]


def metrics(scene, chunk, meta):
    world = world_points(scene, chunk)
    z = world[:, :, 2]
    report = {'frames': chunk['frame_indices'].tolist(), 'coordinate_system': meta['coordinate_system'],
              'source_sha256': meta['sha256'], 'mode': meta['mode'], 'labels_pose': 'corrected',
              'surface_points_per_frame': int(world.shape[1]),
              'ground_min_z_m': z.min(axis=1).tolist(),
              'points_below_ground_0_5mm': (z < -.0005).sum(axis=1).tolist(),
              'scope': 'Sampled surface geometry; not continuous mesh collision or dynamics validation',
              'warnings': meta.get('warnings', [])}
    if 'ui_object_distance' in chunk:
        report['object_min_distance_m'] = np.min(chunk['ui_object_distance'], axis=1).tolist()
        report['object_distance_method'] = meta.get('object_distance_description')
    if meta['mode'] == 'corrected':
        for kind in ('ground', 'object'):
            key = kind + '_point_contact'
            if key in chunk:
                contacts = np.unpackbits(chunk[key], axis=1, bitorder='big')[:, :world.shape[1]]
                report[kind + '_source_contact_points'] = contacts.sum(axis=1).tolist()
    return report


def mesh_for(geom, arrays):
    if geom['type'] == 'mesh':
        return arrays[geom['mesh'] + '_vertices'], arrays[geom['mesh'] + '_faces']
    import trimesh
    size = np.asarray(geom['size'])
    kind = geom['type']
    if kind == 'box':
        mesh = trimesh.creation.box(extents=2 * size)
    elif kind in ('sphere', 'ellipsoid'):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=1)
        mesh.vertices *= size[0] if kind == 'sphere' else size
    elif kind == 'cylinder':
        mesh = trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=24)
    elif kind == 'capsule':
        mesh = trimesh.creation.capsule(radius=size[0], height=2 * size[1], count=[12, 24])
        mesh.vertices -= mesh.bounds.mean(axis=0)
    else:
        raise ValueError('Unsupported source visual primitive: ' + kind)
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def render(panels, view='iso', surface=True, human=True):
    """Each panel is (scene_meta, scene_arrays, chunk_meta, chunk_arrays)."""
    if view not in VIEWS or not 1 <= len(panels) <= 6:
        raise ValueError('Choose a supported view and 1-6 panels')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    world = [world_points(s, c)[0] for _, s, _, c in panels]
    # Common world bounds across panels: no per-body realignment or normalization.
    anchors = [p for p in world]
    for _, _, _, c in panels:
        if human and 'human_points' in c:
            anchors.append(c['human_points'][0])
        if 'object_pos' in c:
            anchors.append(c['object_pos'][0:1])
    xyz = np.concatenate(anchors)
    low, high = xyz.min(axis=0), xyz.max(axis=0)
    low[2] = min(0., low[2]); center = (low + high) / 2
    radius = max(float((high - low).max()) * .58, .4)
    fig = plt.figure(figsize=(5 * min(len(panels), 3), 4.7 * ((len(panels) + 2) // 3)), dpi=100)
    reports = []
    try:
        for index, (sm, sa, cm, ca) in enumerate(panels):
            ax = fig.add_subplot((len(panels) + 2) // 3, min(len(panels), 3), index + 1, projection='3d')
            for geom in sm['geoms']:
                v, f = mesh_for(geom, sa)
                posed = v @ ca['geom_rot'][0, geom['id']].T + ca['geom_pos'][0, geom['id']]
                ax.add_collection3d(Poly3DCollection(posed[f], facecolors=geom['rgba'][:3], linewidths=0, alpha=1))
            for geom in sm.get('object_geoms', []):
                if 'object_pos' not in ca:
                    raise ValueError('Object mesh exists but no object transform')
                v, f = mesh_for(geom, sa)
                local = v @ np.asarray(geom['local_rot']).T + geom['local_pos']
                posed = local @ ca['object_rot'][0].T + ca['object_pos'][0]
                ax.add_collection3d(Poly3DCollection(posed[f], facecolors=geom['rgba'][:3], linewidths=0))
            if human and 'human_points' in ca:
                p = ca['human_points'][0]
                step = max(1, int(np.ceil(len(p) / 1500)))
                ax.scatter(*p[::step].T, s=1, c='#64748b', alpha=.25, depthshade=False)
            if surface:
                p = world[index]; colors = np.tile([.35, .5, .7], (len(p), 1))
                if cm['mode'] == 'corrected':
                    for kind, color in [('ground', [.05, .65, .3]), ('object', [1., .55, .05])]:
                        k = kind + '_point_contact'
                        if k in ca:
                            mask = np.unpackbits(ca[k][0], bitorder='big')[:len(p)].astype(bool)
                            colors[mask] = color
                colors[p[:, 2] < -.0005] = [.85, .1, .2]
                step = max(1, int(np.ceil(len(p) / 2500)))
                ax.scatter(*p[::step].T, c=colors[::step], s=2, depthshade=False)
            ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
                   zlim=(center[2]-radius, center[2]+radius), xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
            ax.set_box_aspect((1, 1, 1)); ax.view_init(*VIEWS[view])
            xx, yy = np.meshgrid([center[0]-radius, center[0]+radius], [center[1]-radius, center[1]+radius])
            ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=.12, color='#94a3b8')
            frame = int(ca['frame_indices'][0])
            ax.set_title(f"{sm.get('item_id', cm['key'])[-55:]}\n{cm['mode']} | frame {frame} | t={frame/cm['fps']:.3f}s", fontsize=9)
            reports.append(metrics(sa, ca, cm))
        fig.suptitle('Source meshes + FK | Kinematic replay | Not physics simulation', fontsize=12)
        fig.text(.02, .015, 'Shared world camera; Z-up metres. Green/orange: corrected source contact. Red: sampled points below ground >0.5 mm.\nHuman points and surface overlay may be subsampled for display; metrics use all points.', fontsize=8)
        fig.tight_layout(rect=(0, .065, 1, .94))
        buffer = io.BytesIO(); fig.savefig(buffer, format='png')
        return buffer.getvalue(), {'view': view, 'camera_center_m': center.tolist(), 'camera_radius_m': radius,
                                  'panels': reports, 'evidence_level': 'kinematic_replay'}
    finally:
        plt.close(fig)
