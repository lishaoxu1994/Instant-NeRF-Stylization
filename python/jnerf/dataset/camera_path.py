import jittor as jt
import numpy as np

def pose_spherical(theta, phi, radius):
    trans_t = lambda t : jt.array(np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,t],
        [0,0,0,1]]).astype(np.float32))
    rot_phi = lambda phi : jt.array(np.array([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]]).astype(np.float32))
    rot_theta = lambda th : jt.array(np.array([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]]).astype(np.float32))
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = jt.array(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    c2w = c2w[:-1, :]
    return c2w

def path_spherical(nframe=4):
    #poses_t = [pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,nframe+1)[:-1]]
    poses = []
    pose_angle = []
    angle_x = -180
    angle_y = -90
    angle_z = 0
    for angle_x_add in np.linspace(0,0,1):
        for angle_y_add in np.linspace(0,0,1):
            #angle_z_add = 0
            for angle_z_add in np.linspace(0,0,1):
                poses.append(pose_spherical(angle_x+angle_x_add, angle_y+angle_y_add, angle_z+angle_z_add))
                pose_angle.append([angle_x_add, angle_y_add, angle_z_add])
    return poses, pose_angle