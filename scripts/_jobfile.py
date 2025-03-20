import os
import glob
import numpy as np
import subprocess

parent_dir = os.getcwd()
dat_path = parent_dir + "/data/final_data"


num_plrs = 1500

Rp, Sp, Tp, Pp = 1.0, -0.1, 1.1, 0.0
Wh = 0.7

kp = 4
kh = 2
wf = np.round(1/(kp+kh),3)
rho_pi = 0.5
rho_hi = 0.5

runs_num = 400

num_cores = 256


overlap_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
p_switch_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
Gh_list = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]

all_processes = []
all_parameters = []


for overlap in overlap_list:
    for p_switch in p_switch_list:
        for Gh in Gh_list:

            for this_ind in range(runs_num):
        
                all_processes.append(f'julia -t 1 --project=. ./main_bilayer.jl {overlap} {Gh} {rho_pi} {rho_hi} {p_switch} {this_ind} & ')
                all_parameters.append((overlap, Gh, p_switch, this_ind))


print(len(all_processes))


# run all processes in batches of num_cores
for i in range(0, len(all_processes), num_cores):
    commands = all_processes[i:i+num_cores]
    commands = ' '.join(commands) + ' wait'
    result = subprocess.call(commands, shell=True)

    print(f"Batch {i} done")


    parameters = all_parameters[i:i+num_cores]
    for this_par in parameters:

        overlap, Gh, p_switch, run_ID = this_par
        run_ID = run_ID[0]

        filename = f"RR-averaged_coop_N_{num_plrs}_kp_{kp}_kh_{kh}_overlap_{overlap}_Rp_{Rp}_Sp_{Sp}_Tp_{Tp}_Pp_{Pp}_Gh_{Gh}_Wh_{Wh}_wf_{wf}_rho_pi_{rho_pi}_rho_hi_{rho_hi}_p_switch_{p_switch}"


        dat = np.load(dat_path + filename + ".npy")

        data_p = np.loadtxt(dat_path + filename.replace('averaged_coop','temporal_coop_pairwise_single_run') + f"_runID_{run_ID}.txt")
        data_h = np.loadtxt(dat_path + filename.replace('averaged_coop','temporal_coop_higher_single_run') + f"_runID_{run_ID}.txt")

        dat[0, run_ID] = np.mean(data_p)
        dat[1, run_ID] = np.mean(data_h)
        dat[2, run_ID] = np.std(data_p)
        dat[3, run_ID] = np.std(data_h)

        np.save(dat_path + filename + ".npy", dat)

        os.remove(dat_path + filename.replace('averaged_coop','temporal_coop_pairwise_single_run') + f"_runID_{run_ID}.txt")
        os.remove(dat_path + filename.replace('averaged_coop','temporal_coop_higher_single_run') + f"_runID_{run_ID}.txt")


print("done")




























'''
try:
    temp = np.load(dat_path + f"RR-averaged_coop_N_{num_plrs}_kp_{kp}_kh_{kh}_overlap_{overlap}_Rp_{Rp}_Sp_{Sp}_Tp_{Tp}_Pp_{Pp}_Gh_{Gh}_Wh_{Wh}_wf_{wf}_rho_pi_{rho_pi}_rho_hi_{rho_hi}_p_switch_{p_switch}.npy")

except:

    data_set = np.full((4, runs_num), np.nan)

    filename = f"RR-temporal_coop_pairwise_single_run_N_{num_plrs}_kp_{kp}_kh_{kh}_overlap_{overlap}_Rp_{Rp}_Sp_{Sp}_Tp_{Tp}_Pp_{Pp}_Gh_{Gh}_Wh_{Wh}_wf_{wf}_rho_pi_{rho_pi}_rho_hi_{rho_hi}_p_switch_{p_switch}_runID_*"

    all_files = glob.glob(dat_path + filename)

    for this_file in all_files:
        data_p = np.loadtxt(this_file)
        data_h = np.loadtxt(this_file.replace('pairwise', 'higher'))

        id = this_file.replace(dat_path + f"RR-temporal_coop_pairwise_single_run_N_{num_plrs}_kp_{kp}_kh_{kh}_overlap_{overlap}_Rp_{Rp}_Sp_{Sp}_Tp_{Tp}_Pp_{Pp}_Gh_{Gh}_Wh_{Wh}_wf_{wf}_rho_pi_{rho_pi}_rho_hi_{rho_hi}_p_switch_{p_switch}_runID_", '').replace('.txt', '')

        id = int(id)

        data_set[0, id] = np.mean(data_p)
        data_set[1, id] = np.mean(data_h)
        data_set[2, id] = np.std(data_p)
        data_set[3, id] = np.std(data_h)

        os.remove(this_file)
        os.remove(this_file.replace('pairwise', 'higher'))

    np.save(dat_path + f"RR-averaged_coop_N_{num_plrs}_kp_{kp}_kh_{kh}_overlap_{overlap}_Rp_{Rp}_Sp_{Sp}_Tp_{Tp}_Pp_{Pp}_Gh_{Gh}_Wh_{Wh}_wf_{wf}_rho_pi_{rho_pi}_rho_hi_{rho_hi}_p_switch_{p_switch}.npy", data_set)


time_end = time.time()

print(f"Overlap: {overlap}, p_switch: {p_switch}, Gh: {Gh}, time: {time_end - time_start}")
'''