"""WP-ASSETS: UMR-density dense surface assets (surface_v2) for all 18 AM2 bodies.

Runtime API lives in surface_v2.py (pure torch, no sealed imports).  Build code (MuJoCo/trimesh/SMPL-X) lives in
the other modules and is only imported by build_surface_v2.py, render_assets.py and the tests.
"""
