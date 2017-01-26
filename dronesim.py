import numpy as _np
import vpython as _vp
import transforms3d.euler as _euler

### settings

_dt              = 0.025
_update_dt       = 0.1
_follow_drone    = True
_size            = 0.5
_mass            = 1.0
_air_density     = 1.2
_gravity         = 9.8
_ground_friction = 0.8
_lin_drag_coef   = 0.5 * _air_density * _np.pi * 0.47
_rot_drag_coef   = _size**2
_power_coef      = 5.0
_wind            = 0.1

### initialize public variables

time = 0.0
canvas = _vp.canvas(background=_vp.color.cyan, range=10, forward=_vp.vector(1,-0.2,0), caption='')
ground = _vp.box(pos=_vp.vector(0,-_size-0.1,0), length=1000, height=0.2, width=1000, color=_vp.color.green, texture=_vp.textures.rough)

### takeoff/landing pads

start_pad = _vp.box(pos=_vp.vector(0,-_size-0.1,0), length=2, height=0.22, width=2, color=_vp.color.yellow)
lift_pad = _vp.box(pos=_vp.vector(10,-_size-0.1,10), length=2, height=0.22, width=2, color=_vp.color.magenta)
end_pad = _vp.box(pos=_vp.vector(10,-_size-0.1,-10), length=2, height=0.22, width=2, color=_vp.color.cyan)

### class definition

class Drone:

  def __init__(self):
    self.mass     = _mass
    self.size     = _size
    self.inertia  = 2.0/3.0 * self.mass * self.size**2
    self.cgpos    = -0.25 * _size
    self.energy   = 0.0
    self.thrust1  = 0.0
    self.thrust2  = 0.0
    self.thrust3  = 0.0
    self.thrust4  = 0.0
    self.body     = _vp.sphere(radius=1.0*_size, color=_vp.color.red)
    self.top      = _vp.sphere(radius=0.2*_size, color=_vp.color.blue)
    self.prop1    = _vp.ring(radius=0.3*_size, thickness=0.05*_size, color=_vp.color.orange)
    self.prop2    = _vp.ring(radius=0.3*_size, thickness=0.05*_size, color=_vp.color.blue)
    self.prop3    = _vp.ring(radius=0.3*_size, thickness=0.05*_size, color=_vp.color.blue)
    self.prop4    = _vp.ring(radius=0.3*_size, thickness=0.05*_size, color=_vp.color.blue)
    self.wind     = _wind * _vp.vector(_np.random.normal(), 0.0, _np.random.normal())
    self.reset()

  def reset(self):
    self.xyz      = _vp.vector(0,0,0)
    self.xyz_dot  = _vp.vector(0,0,0)
    self.pqr      = _vp.vector(0,0,0)
    self.pqr_dot  = _vp.vector(0,0,0)
    self.draw()

  def draw(self):
    axis, theta = _euler.euler2axangle(self.pqr.x, self.pqr.y, self.pqr.z)
    axis = _vp.vector(axis[0], axis[1], axis[2])
    up = _vp.rotate(_vp.vector(0,1,0), theta, axis)
    self.body.pos   = self.xyz
    self.top.pos    = self.xyz + up*_size
    self.prop1.pos  = self.xyz + _vp.rotate(_vp.vector(1.3*_size,0,0), theta, axis)
    self.prop2.pos  = self.xyz + _vp.rotate(_vp.vector(0,0,1.3*_size), theta, axis)
    self.prop3.pos  = self.xyz + _vp.rotate(_vp.vector(-1.3*_size,0,0), theta, axis)
    self.prop4.pos  = self.xyz + _vp.rotate(_vp.vector(0,0,-1.3*_size), theta, axis)
    self.prop1.axis = up
    self.prop2.axis = up
    self.prop3.axis = up
    self.prop4.axis = up
    if _follow_drone:
      canvas.center = self.xyz
    canvas.caption = 'time = %0.1f, pos = (%0.1f, %0.1f, %0.1f), mass = %0.1f, energy = %0.1f' % (time, self.xyz.x, self.xyz.y, self.xyz.z, self.mass, self.energy)

  def update(self, dt):
    # forces
    axis, theta = _euler.euler2axangle(self.pqr.x, self.pqr.y, self.pqr.z)
    axis = _vp.vector(axis[0], axis[1], axis[2])
    up = _vp.rotate(_vp.vector(0,1,0), theta, axis)
    a = _vp.vector(0, -_gravity, 0)
    a = a + (self.thrust1+self.thrust2+self.thrust3+self.thrust4)/self.mass * up + self.wind/self.mass
    a = a - (_lin_drag_coef * _vp.mag(self.xyz_dot)**2)/self.mass * self.xyz_dot
    self.xyz_dot = self.xyz_dot + a * dt
    # torques (ignoring propeller torques)
    cg = self.cgpos * up
    tpos1 = _vp.rotate(_vp.vector(1.3*_size,0,0), theta, axis)
    tpos2 = _vp.rotate(_vp.vector(0,0,1.3*_size), theta, axis)
    tpos3 = _vp.rotate(_vp.vector(-1.3*_size,0,0), theta, axis)
    tpos4 = _vp.rotate(_vp.vector(0,0,-1.3*_size), theta, axis)
    torque = _vp.cross(cg, _vp.vector(0, -_gravity, 0))
    torque = torque + _vp.cross(tpos1, self.thrust1 * up)
    torque = torque + _vp.cross(tpos2, self.thrust2 * up)
    torque = torque + _vp.cross(tpos3, self.thrust3 * up)
    torque = torque + _vp.cross(tpos4, self.thrust4 * up)
    torque = torque - _rot_drag_coef * self.pqr_dot
    aa = torque/self.inertia
    if _vp.mag(aa) > 0:
      aai, aaj, aak = _euler.axangle2euler((aa.x, aa.y, aa.z), _vp.mag(aa))
      aa = _vp.vector(aai, aaj, aak)
      self.pqr_dot = self.pqr_dot + aa * dt
    else:
      self.pqr_dot = _vp.vector(0,0,0)
    # ground interaction
    if self.xyz.y <= 0:
      self.xyz.y = 0
      if _np.abs(self.xyz.x-lift_pad.pos.x) < 1 and _np.abs(self.xyz.z-lift_pad.pos.z) < 1:
          self.mass = 1.5
          self.inertia = 2.0/3.0 * self.mass * self.size**2
          self.body.color = _vp.color.orange
      if self.xyz_dot.y <= 0:
        self.xyz_dot.x = self.xyz_dot.x * _ground_friction
        self.xyz_dot.y = 0
        self.xyz_dot.z = self.xyz_dot.z * _ground_friction
        self.pqr_dot = self.pqr_dot * _ground_friction
    # energy update
    self.energy += _power_coef * (self.thrust1**1.5 + self.thrust2**1.5 + self.thrust3**1.5 + self.thrust4**1.5) * dt
    # time update
    self.xyz += self.xyz_dot * dt
    self.pqr += self.pqr_dot * dt
    self.draw()

  def altitude(self):
    return self.xyz.y

  def roll(self):
    return self.pqr.x

  def yaw(self):
    return self.pqr.y

  def pitch(self):
    return self.pqr.z

  def x(self):
    return self.xyz.x

  def y(self):
    return self.xyz.y

  def z(self):
    return self.xyz.z


### initialize drone (public variable)

drone = Drone()

### utility functions (public)

def reset():
  global canvas, drone, time
  time = 0.0
  drone.reset()
  canvas.waitfor('redraw')

def delay(t):
  global canvas, _update_dt, time, _dt, drone
  t0 = time
  t1 = time + t
  while time < t1:
    drone.update(_dt)
    time += _dt
    if time-t0 > _update_dt:
      canvas.waitfor('redraw')
      t0 = time
  canvas.waitfor('redraw')

def thrust(t1, t2=None, t3=None, t4=None):
  global drone
  if t2 == None: t2 = t1
  if t3 == None: t3 = t1
  if t4 == None: t4 = t1
  drone.thrust1 = t1
  drone.thrust2 = t2
  drone.thrust3 = t3
  drone.thrust4 = t4
