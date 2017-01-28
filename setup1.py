from dronesim import *
import numpy as _np
import vpython as _vp

_size = 0.5

start_pad = _vp.box(pos=_vp.vector(0,-_size-0.1,0), length=2, height=0.22, width=2, color=_vp.color.yellow)
lift_pad = _vp.box(pos=_vp.vector(10,-_size-0.1,10), length=2, height=0.22, width=2, color=_vp.color.magenta)
end_pad = _vp.box(pos=_vp.vector(10,-_size-0.1,-10), length=2, height=0.22, width=2, color=_vp.color.cyan)

def updated_cb(drone):
    if drone.xyz.y < 1 and _np.abs(drone.xyz.x-lift_pad.pos.x) < 1 and _np.abs(drone.xyz.z-lift_pad.pos.z) < 1:
        drone.set_mass(1.5)
        drone.body.color = _vp.color.orange

drone.set_updated_callback(updated_cb)
