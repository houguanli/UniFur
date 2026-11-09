"""Spatially coherent gusts shared by guide and terminal-strand dynamics."""
import numpy as np
from hairgs_hybrid.gs_dynamics_v1.rig_motion import RigWindDriver, _soft_limit


class GustWindDriver(RigWindDriver):
    def air_velocity(self, time):
        p = self.parameters
        roots = self._previous_roots
        phase = .2*self.phase if roots is None else roots@np.array([.8,1.8,.6])
        pulse = .5+.5*np.sin(2*np.pi*p['gust_hz']*time+.35*phase)
        speed = p['wind_strength']*((1-p['gust_strength'])+p['gust_strength']*pulse)
        az = np.deg2rad(p['wind_azimuth']+40*np.sin(2*np.pi*.32*time+.2*phase)
                       +10*np.sin(2*np.pi*.83*time+.35*phase))
        el = np.deg2rad(p['wind_elevation']+16*np.sin(2*np.pi*.49*time+.2*phase+1.2))
        return speed[:,None]*np.stack((np.cos(az)*np.cos(el),np.sin(az)*np.cos(el),np.sin(el)),axis=1)

    def _target(self, time):
        # Same drag/inertia, skin guard and force-to-angle convention as the
        # existing RigWindDriver; only the air field is replaced.
        p = self.parameters
        limit = np.deg2rad(p['max_angle_degrees'])
        world_air = self.air_velocity(time)-self.inertia_strength*.8*(self.root_velocity+self.frame_velocity)/self.body_scale
        inv = np.swapaxes(self.body_rotations,-1,-2)
        fur = self.fur_directions
        drag = _soft_limit(np.cross(fur,np.einsum('nij,nj->ni',inv,world_air))*limit,limit)
        omega = 2*np.pi*p['frequency_hz']*(1+.12*np.sin(self.phase))
        field = self.inertia_strength*(np.einsum('nij,nj->ni',inv,self.GRAVITY-self.root_acceleration)-self.GRAVITY)
        inertial = np.cross(fur,field)/(self.hair_length*omega**2)[:,None]
        inertial -= self.inertia_strength*np.einsum('nij,nj->ni',inv,self.angular_acceleration)/(omega**2)[:,None]
        return self._skin_guard(_soft_limit(drag+inertial,min(2*limit,np.deg2rad(75.))))
