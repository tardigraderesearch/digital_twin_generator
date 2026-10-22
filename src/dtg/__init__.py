"""Digital Twin Generator: built robot -> verified, sim-ready model.

Pipeline (see README):
  1. load scan            dtg.scan
  2. scan health check    dtg.health, dtg.preview
  3. tag placement        markers.json from aruco-tag-maker (dtg.markers)
  4. wiggle               dtg.observations + dtg.linkage  (camera only)
  5. ROM sweep            dtg.observations + dtg.kinematics (camera only)
  6. register tags->scan  dtg.registration
  7. cut scan by linkage  dtg.segment
  8. missing parts        dtg.parts, dtg.servo_library
  9. export               dtg.export
"""
__version__ = "0.1.0.dev0"
