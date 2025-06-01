import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import h5py

def save(filename, *args):
    # Get global dictionary
    glob = globals()
    d = {}
    for v in args:
        # Copy over desired values
        d[v] = glob[v]
    with open(filename, 'wb') as f:
        # Put them in the file 
        pickle.dump(d, f)

def load(filename):
    # Get global dictionary
    glob = globals()
    with open(filename, 'rb') as f:
        for k, v in pickle.load(f).items():
            # Set each global variable to the value from the file
            glob[k] = v

v_drift = 0.16 #cm/us @ 500kV/cm
pitches = {'coarse_hits':50, 'pixel_hits':10} #taken from readout_config/default.yaml

input_file = '1-2GeVmuons_nonoise.h5'
input_file_DT = '1-2GeVmuons_DT'

''' 'coarse or fine' - choosing which to analyse '''
cf = 'coarse_hits'
cf = 'pixel_hits'
pitch = pitches[cf]

fh = h5py.File(input_file)
n_events = len(np.unique(np.array(fh[cf]['event id'])))
load(input_file_DT) #now we have drifted_tracks, an array of the drifted_track dict for all events in the input file

'''histogram of how many of the tiles see a particular amount of charge'''
# rows = int(np.ceil(n_events**0.5))
# columns = int(np.ceil(n_events/rows))
# plt.suptitle('histogram: amount of charge on one hit (e-) ({})'.format(cf))
# for i in range(n_events):
    # event_mask = fh[cf]['event id'] == i
    # plt.subplot(rows, columns, i+1)
    # plt.hist((fh[cf][event_mask])['hit charge'], bins=65)
    # meta_event_mask = fh['meta']['event id'] == i
    # plt.plot([], [], label='{} MeV'.format(int(fh['meta']['primary energy'][meta_event_mask])))
    # plt.plot([], [], label='θ={}°'.format(int(fh['meta']['phi'][meta_event_mask]*360/(2*3.14159))))
    # plt.plot([], [], label='φ={}°'.format(int(fh['meta']['theta'][meta_event_mask]*360/(2*3.14159))))
    # plt.plot([], [], label='hits={}'.format(len((fh[cf][event_mask])['hit charge'])))
    # plt.legend(handlelength=0, handletextpad=0, fontsize=7)
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# plt.tight_layout()
# plt.show()


'''cdf of how many of the tiles detect upto a certain amount of charge'''
# rows = int(np.ceil(n_events**0.5))
# columns = int(np.ceil(n_events/rows))
# plt.suptitle('y% of tiles detected x charge or less ({})'.format(cf))
# for i in range(n_events):
    # event_mask = fh[cf]['event id'] == i
    # meta_event_mask = fh['meta']['event id'] == i
    # n_hits = len((fh[cf][event_mask])['hit charge'])
    # counts, bins, patches = plt.hist((fh[cf][event_mask])['hit charge'], bins=int(n_hits/3), cumulative=True, density=True, alpha=0)
    # plt.stairs(counts, bins, baseline=None, color='C{}'.format(i), label='{} MeV'.format(int(fh['meta']['primary energy'][meta_event_mask])))
# plt.legend()
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# plt.yticks(np.arange(0,1,0.1))
# plt.tight_layout()
# plt.show()


'''total detected charge vs total deposited charge (with attenuation and detector volume cut applied)'''
# total_detected_charge = []
# total_deposited_charge = []
# for i in range(n_events):
    # event_mask = fh[cf]['event id'] == i
    # total_detected_charge.append(np.sum((fh[cf][event_mask])['hit charge']))
    # total_deposited_charge.append(torch.sum(drifted_tracks[i]['charge']))
    # meta_event_mask = fh['meta']['event id'] == i
    # plt.scatter(i, total_detected_charge[i]/total_deposited_charge[i], label='{} MeV'.format(int(fh['meta']['primary energy'][meta_event_mask])))
########### plt.axline(xy1=(0,0), slope=1, label='y=x')
# plt.legend()
# plt.xlabel('event id')
# plt.ylabel('total detected charge / total deposited charge ({})'.format(cf))
# plt.show()

'''if dynamic range was limited, how much of the total charge would be lost? (assume input was generated with nonoise=True)'''
# total_detected_charge = []
# total_deposited_charge = []
# d_sweep = np.linspace(2000, 200000, 10)
# for i in range(n_events):
    # event_mask = fh[cf]['event id'] == i
    # max_charge_detected = (np.sum((fh[cf][event_mask])['hit charge']))
    # total_deposited_charge.append(torch.sum(drifted_tracks[i]['charge']))
    # total_detected_charge.append([])
    # meta_event_mask = fh['meta']['event id'] == i
    # for d in d_sweep: #d=dynamic range upper limit, i.e. saturation point
        # total_detected_charge[i].append(np.sum(np.where( (fh[cf][event_mask])['hit charge']>d, d, (fh[cf][event_mask])['hit charge'] )))
    # plt.plot(d_sweep, np.array(total_detected_charge[i])/np.array(total_deposited_charge[i]), label='{} MeV'.format(int(fh['meta']['primary energy'][meta_event_mask])), marker='o')
# plt.legend()
# plt.xlabel('dynamic range (e-)')
# plt.ylabel('percentage of charge detected ({})'.format(cf))
# plt.show()

'''loss in detected charge vs drift length'''
'''hit z in readout_objects actually refers to arrival time! (in us)'''
# for i in range(n_events):
    # event_mask = fh[cf]['event id'] == i
    # meta_event_mask = fh['meta']['event id'] == i
    # hit_zs = v_drift*fh[cf][event_mask]['hit z'] #'hit z' field actually stores arrival time (us)
    # hit_qs = fh[cf][event_mask]['hit charge']
    # dz = v_drift*1 #drift speed * clock tick
    # true_qs = []
    # for hit in fh[cf][event_mask]:
        # hit_z = v_drift*hit['hit z']
        # hit_q = hit['hit charge']
        # hit_x = hit['tile x' if cf=='coarse_hits' else 'pixel x']
        # hit_y = hit['tile y' if cf=='coarse_hits' else 'pixel y']
        # sample_pos = np.array(drifted_tracks[i]['position'])
        # xy_bin_mask = sample_pos[:,0] >= hit_x-pitch/2
        # xy_bin_mask *= sample_pos[:,0] <= hit_x+pitch/2
        # xy_bin_mask *= sample_pos[:,1] >= hit_y-pitch/2
        # xy_bin_mask *= sample_pos[:,1] <= hit_y+pitch/2
        # z_bin_mask = sample_pos[:,2] >= hit_z
        # z_bin_mask *= sample_pos[:,2] <= hit_z+dz
        # overall_mask = z_bin_mask*xy_bin_mask
        # sample_qs_in_bin = np.array(drifted_tracks[i]['charge'][overall_mask])
        # true_q = np.sum(sample_qs_in_bin)
        # true_qs.append(true_q)
    # true_qs = np.array(true_qs)
    # plt.scatter(hit_zs, hit_qs/true_qs, marker='.', alpha=0.3, label='{} MeV'.format(int(fh['meta']['primary energy'][meta_event_mask])))
    ##### plt.scatter(true_qs, hit_qs, marker='.', alpha=0.3, label='{} MeV'.format(int(fh['meta']['primary energy'][meta_event_mask])))
    ##### plt.scatter(np.linspace(min(hit_zs), max(hit_zs), len(hit_zs)), hit_zs)
### plt.axline(xy1=(0,0), slope=1, label='y=x', color='black')
# plt.legend()
# plt.show()

'''number of hits'''
for i in range(n_events):
    event_mask = fh[cf]['event id'] == i
    meta_event_mask = fh['meta']['event id'] == i
    # hits = len(drifted_tracks[i]['coarse_tiles_samples' if cf=='coarse_hits' else 'pixel_samples'])
    hits =  len(fh[cf][event_mask]['hit charge'])
    theta = fh['meta']['phi'][meta_event_mask]*360/(2*3.14159)
    plt.scatter(theta, hits, label='{} MeV'.format(int(fh['meta']['primary energy'][meta_event_mask])))
plt.legend()
plt.xlabel('θ')
plt.ylabel('# of hits')
plt.show()


fh.close()