import pylhe
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count

"""
Previously, we sample the radiation l wrt the beam direction z, which inevitably
introduces a bias in the radiation direction. In this version, we sample the radiation l
in the rest frame of the parton, which is more accurate.

But the problem is the parton is massless in high energy limit, so we need to
introduce a small mass to the parton.

TO DO:
1. Implement an eta cut(-2.5 to 2.5) to both the splitting jet and the radiation l.
2. Implement a delta R cut (<0.35) to the splitting jet and the radiation l.
3. Normalize the asymmetry rate.

July 25, 2025
DUMB MISTAKE: wrong histogram. density=True is not what we want.
mjj channel 007 not 006 2000-2250GeV
"""

no = 1
z = [0,0,1] # beam direction

######## Define Functions ########
# rotation matrix
def rotation_matrix(p):
    """
    Determines the rotation matrix between the z-axis and the particle's (final) three-momentum
    """
    E0, px0, py0, pz0 = p[0], p[1], p[2], p[3]
    ThZ = np.arccos(pz0/np.sqrt(px0**2 + py0**2 + pz0**2))
    PhiZ = np.arctan2(py0, px0)
    return [[1, 0, 0, 0],
        [0, np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
        [0, np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
        [0, -np.sin(ThZ), 0, np.cos(ThZ)]]

def get_l_list(p):
    l_list = []
    for i in range(Ngrid):
        for j in range(Ngrid):
            # sample l randomly wrt to phi and eta
            phi = np.random.uniform(0.0, 2.0*np.pi)
            eta = np.random.uniform(-5.0, 5.0)
            l_cm = [np.cosh(eta), np.cos(phi), np.sin(phi), np.sinh(eta)]
            # rotate the particle from rest frame of the parton to lab frame
            l_rot = np.dot(rotation_matrix(p), l_cm)
            # calculate pseudorapidity
            eta_rot = np.arctanh(l_rot[3]/l_rot[0])
            if eta_rot < -2.5 or eta_rot > 2.5:
                continue
            # calculate delta R
            delta_phi = np.arctan2(np.sin(eta_phi(l_rot)[1] - eta_phi(p)[1]), np.cos(eta_phi(l_rot)[1] - eta_phi(p)[1]))
            delta_eta = eta_rot - eta_phi(p)[0]
            delta_r = np.sqrt(delta_phi**2 + delta_eta**2)
            #print(delta_phi, delta_eta, delta_r)
            if delta_r > 0.35:
                continue
            l_list.append(l_rot)
            #l_list.append([np.cosh(eta_list[j]), np.cos(phi_list[i]), np.sin(phi_list[i]), np.sinh(eta_list[j])])
    return l_list

# dot product of two 4-vectors
def dot_product(k1, k2):
    return k1[0]*k2[0] - k1[1]*k2[1] - k1[2]*k2[2] - k1[3]*k2[3]

def eta_phi(k):
    return np.arctanh(k[3]/k[0]), np.arctan2(k[2], k[1])
# calculate the radiation amplitude, assuming lT = 1
def radiation_amplitude(k1, k2, l):
    # this blows up when l and k1/k2 are collinear
    return dot_product(k1, k2)/(dot_product(l, k1)*dot_product(l, k2))

def asymmetry_angle(k, l, z):
    cross_product = np.cross(k[1:], l[1:])
    vector_sum = np.array(k[1:]) + np.array(l[1:])
    numerator = np.dot(cross_product, z) * np.linalg.norm(vector_sum)
    denominator = np.dot(np.cross(cross_product, z), vector_sum)
    fraction = np.abs(numerator/denominator)
    return np.arctan(fraction)

def asymmetry_variable(k, l, z):
    return np.cos(2*asymmetry_angle(k, l, z))

def calculate_amplitude(args):
    event_no = args
    p1 = np.array(particle_list[event_no*4])
    p2 = np.array(particle_list[event_no*4+1])
    k1 = np.array(particle_list[event_no*4+2])
    k2 = np.array(particle_list[event_no*4+3])
    # which parton to radiate from
    k = k1
    radiation_amplitude_pk = 0
    radiation_amplitude_kk = 0
    radiation_amplitude_pk_weighted = 0
    asym = 0
    # smaple radiation l around k1
    l_list = get_l_list(k)
    #print(f"Event {event_no}: {len(l_list)} radiation vectors sampled.")
    for i in range(len(l_list)):
        if len(l_list) == 0:
            continue
        l = l_list[i]
        radiation_amplitude_pk_weighted += (radiation_amplitude(p1, k, l)+radiation_amplitude(p2, k, l))*asymmetry_variable(k, l, z)
        radiation_amplitude_pk += radiation_amplitude(p1, k, l) + radiation_amplitude(p2, k, l)
        asym += asymmetry_variable(k, l, z)
        #radiation_amplitude_kk += radiation_amplitude(k1, k2, l)*(asymmetry_variable(k1, l, z)+asymmetry_variable(k2, l, z))
    #radiation_amplitude_combined = (radiation_amplitude_pk + radiation_amplitude_kk)/5
    if radiation_amplitude_pk == 0:
        return [0, 0, eta_phi(k)[0],0]
    else:
        normalized_amplitude = radiation_amplitude_pk_weighted/radiation_amplitude_pk
        return [normalized_amplitude, radiation_amplitude_pk, eta_phi(k)[0], asym]

#for CHN in range(1, 10):
CHN = 2
events = pylhe.read_lhe_with_attributes(f"events/500k_channels_eta2.5/run_0{CHN}/unweighted_events.lhe")
Nevents = int(5e5)
# events = pylhe.read_lhe_with_attributes(f"events/10k_channels/run_0{CHN}/unweighted_events.lhe")
# Nevents = int(1e4)
particle_list = []

# load particles from the LHE file
for event in events:
    for particle in event.particles:
        #if abs(particle.id) <= 6 or particle.id == 21:  # quarks and gluons
            # save the information of the particle
        particle_list.append([particle.e, particle.px, particle.py, particle.pz])
            #print(f"ID: {particle.id}, E: {particle.e}, px: {particle.px}, py: {particle.py}, pz: {particle.pz}")
    #print("--------------------------------------------------")


# parameter list of radiated l
Ngrid = 100
# phi_list = np.linspace(0, 2*np.pi, Ngrid)
# eta_list = np.linspace(-3, 3, Ngrid)
# l_list = []
# for i in range(Ngrid):
#     for j in range(Ngrid):
#         # sample l randomly wrt to cos(theta) and phi.
#         cos_theta = np.random.uniform(-1.0, 1.0)
#         sin_theta = np.sqrt(1 - cos_theta**2)
#         phi = np.random.uniform(0.0, 2.0*np.pi)
#         l_list.append([1, sin_theta*np.sin(phi), sin_theta*np.cos(phi), cos_theta])

######## Set up Parameter Scan ########

# Prepare arguments for multiprocessing
args = [i for i in range(1000)]
#args = [(i, j, phi_list, eta_list, particle_list, z) for i in range(len(phi_list)) for j in range(len(eta_list))]

# run scan in parallel
if __name__ == "__main__":
    # Use multiprocessing to calculate amplitudes
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(calculate_amplitude, args), total=len(args)))
    # Collect results
    amp_list_combined = np.array([result[0] for result in results])
    rad_list_combined = np.array([result[1] for result in results])
    k_list = np.array([result[2] for result in results])
    asym_list = np.array([result[3] for result in results])
    # save the data
    np.save(f'data/500k_rot_cut/amp_list.npy', amp_list_combined)
    np.save(f'data/500k_rot_cut/rad_list.npy', rad_list_combined)
    #np.save(f'data/500k_rot_cut/weighted_amp_list_real_{CHN}_test1.npy', weighted_amp_list_combined)
    np.save(f'data/500k_rot_cut/k_list.npy', k_list)
    np.save(f'data/500k_rot_cut/asym_list.npy', asym_list)

