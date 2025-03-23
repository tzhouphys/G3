import pylhe
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count

events = pylhe.read_lhe_with_attributes("events/run_01/unweighted_events.lhe")
Nevents = int(1e4)
counter = 0
particle_list = []

# load particles from the LHE file
for event in events:
    for particle in event.particles:
        if abs(particle.id) <= 6 or particle.id == 21:  # quarks and gluons
            counter += 1
            # save the information of the particle
            particle_list.append([particle.e, particle.px, particle.py, particle.pz])
            #print(f"ID: {particle.id}, E: {particle.e}, px: {particle.px}, py: {particle.py}, pz: {particle.pz}")
    #print("--------------------------------------------------")

######## Define Functions ########
# dot product of two 4-vectors
def dot_product(k1, k2):
    return k1[0]*k2[0] - k1[1]*k2[1] - k1[2]*k2[2] - k1[3]*k2[3]

# calculate the radiation amplitude, assuming lT = 1
def radiation_amplitude(k1, k2, l):
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
    i, j, phi_list, eta_list, particle_list, z = args
    radiation_amplitude_pk = 0
    radiation_amplitude_kk = 0
    l = [np.cosh(eta_list[j]), np.cos(phi_list[i]), np.sin(phi_list[i]), np.sinh(eta_list[j])]
    for event_no in range(Nevents):
        p1 = np.array(particle_list[event_no*4])
        p2 = np.array(particle_list[event_no*4+1])
        k1 = np.array(particle_list[event_no*4+2])
        k2 = np.array(particle_list[event_no*4+3])
        radiation_amplitude_pk += ((radiation_amplitude(p1, k1, l)+radiation_amplitude(p2, k1, l))*asymmetry_variable(k1, l, z)
                                   +(radiation_amplitude(p1, k2, l)+radiation_amplitude(p2, k2, l))*asymmetry_variable(k2, l, z))
        radiation_amplitude_kk += radiation_amplitude(k1, k2, l)*(asymmetry_variable(k1, l, z)+asymmetry_variable(k2, l, z))
    radiation_amplitude_combined = (radiation_amplitude_pk + radiation_amplitude_kk)/5
    return radiation_amplitude_combined/Nevents
######## Set up Parameter Scan ########
z = [0,0,1] # beam direction

# parameter list of radiated l
phi_list = np.linspace(0, 2*np.pi, 50)
eta_list = np.linspace(-3, 3, 50)
mesh_phi, mesh_eta = np.meshgrid(phi_list, eta_list)
#amp_list_k1k2 = [] # color connected final states
#amp_list_pk = [] # color connected initial and final states
amp_list_combined = [] # all 5 possible color connections

# Prepare arguments for multiprocessing
args = [(i, j, phi_list, eta_list, particle_list, z) for i in range(len(phi_list)) for j in range(len(eta_list))]

# run scan in parallel
if __name__ == "__main__":
    # Use multiprocessing to calculate amplitudes
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(calculate_amplitude, args), total=len(args)))
    # Collect results
    amp_list_combined = np.array(results).reshape(len(phi_list), len(eta_list)).T
    # save the data
    np.save(f'data/amp_list_test.npy', amp_list_combined)

# import matplotlib.pyplot as plte
# plt.rc('text', usetex=True)
# # 3 subplots horizontal
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={'width_ratios':[1,1,1.2],'wspace': 0.1}, sharey=True)

# for ax in ax1, ax2, ax3:
#     ax.set_xlabel(r"$\eta$")
# contour1 = ax1.contourf(mesh_eta, mesh_phi, amp_list_k1k2, levels=np.linspace(0, 10, 10))
# contour2 = ax2.contourf(mesh_eta, mesh_phi, amp_list_pk, levels=np.linspace(0, 10, 10))
# contour3 = ax3.contourf(mesh_eta, mesh_phi, amp_list_combined, levels=np.linspace(0, 10, 10))
# fig.colorbar(contour3, ax=ax3)
# ax1.set_ylabel(r"$\phi$")
# ax1.set_title(r"$l_T^2[k_1 k_2]$")
# ax2.set_title(r"$l_T^2[p\, k]$")
# ax3.set_title(r"Combined(averaged over color connections)")
# plt.show()