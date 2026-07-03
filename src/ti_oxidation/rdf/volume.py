import numpy as np

def get_sphere_volume(r):
    return 4/3*np.pi*(r**3)

def get_cap_volume(r, h):
    return 1/3*np.pi*(h**2)*(3*r-h)

def get_effective_volume(r, h, z):
    p_up = max(0, r-z)

    p_down = max(0, r-(h-z))

    V = get_sphere_volume(r)

    return V - get_cap_volume(r, p_up) - get_cap_volume(r, p_down)

def get_effective_shell_volume(r, dr, h, z):
    p_1_up = max(0, (r+dr)-z)
    p_2_up = max(0, r-z)

    p_1_down = max(0, (r+dr)-(h-z))
    p_2_down = max(0, r-(h-z))

    V_shell = get_sphere_volume(r+dr) - get_sphere_volume(r)
    V_scal1_up = get_cap_volume(r+dr, p_1_up) 
    V_scal2_up = get_cap_volume(r, p_2_up)

    V_scal1_down = get_cap_volume(r+dr, p_1_down) 
    V_scal2_down = get_cap_volume(r, p_2_down)

    return V_shell - (V_scal1_up - V_scal2_up) - (V_scal1_down - V_scal2_down)

