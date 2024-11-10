#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 14:46:20 2023

@author: sserkez
"""

import sys
sys.path.append('/local/lib')

savedir = '/System/Volumes/Data/home/xfeloper/user/guestt/FD_09-11-24/'

import random
#from skopt import gp_minimize
#from skopt.callbacks import CheckpointSaver, VerboseCallback, DeltaYStopper
#from skopt.utils import use_named_args
#from skopt.space import Real
import pydoocs
from statistics import mean
import time
import numpy as np

import matplotlib.pyplot as plt
plt.ion()
#cell_names = [f"XFEL.FEL/UNDULATOR.SASE2/CA{axis}.CELL{i}.SA2/FIELD.OFFSET"
 #             for axis in ['X', 'Y'] for i in range(24, n_Cell+1,1) if i != 9 and i != 18]

import argparse
parser = argparse.ArgumentParser("Contribution Per Cell scan. typing python gain_v2.py SASE2 10 32 will open cells 32 to 10 sequentially")
parser.add_argument("beamline", help="Undulator beamline to scan. Choose among [SASE1, SASE2, SASE3]", type=str)
parser.add_argument("cell_upstream", help="Upstream cell number of the scan range to be opened", type=int)
parser.add_argument("cell_downstream", help="Downstream cell number of the scan range to be opened", type=int)
parser.add_argument("savedir", help="Save Directory", type=str)

parser.add_argument("--Ntrains", default=100, help='Number of trains to process per cell measurement. default 100', type=int)
parser.add_argument("--subtract_last", default=0, help='subtract last measurement as background. 1=yes, 0=no (default)', type=int)
parser.add_argument("--I_address", default=None, help="override doocs address of intensity channel", type=str)
parser.add_argument("--taper_off", default=0.002, help="override taper to disable lasing", type=float)

args = parser.parse_args()

print(args)

machine = 'XFEL'
#machine = 'XFEL_SIM'


beamline = args.beamline
cell_range = (args.cell_upstream,args.cell_downstream)

N_trains_to_read = args.Ntrains
subtract_last_msrmnt_as_bkg = args.subtract_last
taper_off = args.taper_off
#beamline = 'SASE2'
#cell_range = (10,32) #cells to open range
K_to_open_enforce = None #enforced fixed K value to open undulator to

#print("N_trains_to_read=",N_trains_to_read)

#N_trains_to_read=100
#subtract_last_msrmnt_as_bkg = 1

if beamline == 'SASE1':
    address_xgm = machine+'.FEL/XGM/XGM.2643.T9/'
    address_xgm_mean_raw = address_xgm + 'INTENSITY.SA1.RAW.TRAIN'
    address_xgm_std = address_xgm + 'STAT.STDDEV.INTRA'
    address_Nbunches= 'XFEL.DIAG/CHARGE.ML/TORA.2462.T4/NUMBEROFBUNCHES.SA1'
    und_type='U40'
    und_appendix='SA1'
    und_loc_arr = [2244,2250,2256,2262,2269,2275,2281,2287,2293,2299,2305,2311,2317,2323,2330,2336,2342,2348,2354,2360,2366,2372,2378,2384,2391,2397,2403,2409,2415,2421,2427,2439,2445,2452,2458]
    und_cell_nr_arr = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,  34,35,36,37]
    K_to_open_offset=0.05 #relative offset of the last opened cell w.r.t the last chosen cell
    K_to_open_taper=taper_off #relative negative K taper per cell to prevent lasing
elif beamline == 'SASE2':
    address_xgm = machine+'.FEL/XGM/XGM.2595.T6/'
    address_xgm_mean_raw = address_xgm + 'INTENSITY.RAW.TRAIN'
    address_xgm_std = address_xgm + 'STAT.STDDEV.INTRA'
    address_Nbunches= 'XFEL.DIAG/CHARGE.ML/TORA.2744.T5/NUMBEROFBUNCHES.SA2'
    und_type='U40'
    und_appendix='SA2'
    und_loc_arr=[2200,2206,2212,2218,2224,2230,2237,2243, 2255,2261,2267,2273,2279,2285,2291,2297, 2310,2316,2322,2328,2334,2340,2346,2352,2358,2365,2371,2377,2383,2389,2395,2401,2407,2413,2419]
    und_cell_nr_arr=[1,2,3,4,5,6,7,8, 10,11,12,13,14,15,16,17, 19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]
    K_to_open_offset=0.05 #relative offset of the last opened cell w.r.t the last chosen cell
    K_to_open_taper=taper_off #relative negative K taper per cell to prevent lasing
elif beamline == 'SASE3':
    address_xgm = machine+'.FEL/XGM/XGM.3130.T10/'
    address_xgm_mean_raw = address_xgm + 'INTENSITY.SA3.RAW.TRAIN'
    address_xgm_std = address_xgm + 'STAT.STDDEV.INTRA'
    address_Nbunches= 'XFEL.DIAG/CHARGE.ML/TORA.2967.T4D/NUMBEROFBUNCHES.SA3'
    und_type='U68'
    und_appendix='SA3'
    und_loc_arr = [2809,2815,2821,2827,2834,2840,2846,2852,2858,2864,2870,2882,2888,2894,2901,2907,2913,2919,2925,2931,2937,2942,2946,2950,2953]
    und_cell_nr_arr = [2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
    K_to_open_offset=0.05 #relative offset of the last opened cell w.r.t the last chosen cell
    K_to_open_taper=taper_off #relative negative K taper per cell to prevent lasing

if args.I_address is None:
	I_address =address_xgm_mean_raw
else:
	print("I_address overrulled with " + args.I_address)
	I_address = args.I_address

#I_address='XFEL.UTIL/DYNPROP/MISC/HIREX_INTEG' #Pyspectrometer
#K_to_open_offset=0.05 #relative offset of the last opened cell w.r.t the last chosen cell
#K_to_open_taper=0.002 #relative negative K taper per cell to prevent lasing
    
#pydoocs.read('XFEL.FEL/XGM/XGM.2643.T9/STAT.STDDEV.INTRA')['data']


    
#intensity = pydoocs.read(address_xgm_mean_raw)['data']
#print(beamline,intensity)


def path_cell(machine, beamline, und_type, und_loc, und_appendix):
    if und_appendix=='SA3' and und_loc in [2942,2946,2950,2953]: #bugfix to accommodate appleX
        und_type_ = 'UE90'#bugfix to accommodate appleX
    else:
        und_type_ = und_type
    string = '{machine}.FEL/UNDULATOR.{beamline}/{und_type}.{und_loc}.{und_appendix}'.format(machine=machine, beamline=beamline, und_type=und_type_, und_loc=und_loc, und_appendix=und_appendix)
    #print(string)
    return string
    
#path_cell_K_rbv = path_cell(machine, beamline, und_type, und_loc, und_appendix) + '/K'
#path_cell_K_set = path_cell(machine, beamline, und_type, und_loc, und_appendix) + '/K.SET'
#command_use_K = path_cell(machine, beamline, und_type, und_loc, und_appendix) + '/CONTROL.USE_K'
#command_move_undulator  = path_cell(machine, beamline, und_type, und_loc, und_appendix) + '/CONTROL.START'

def stderr(data):
    return np.std(data, ddof=1) / np.sqrt(np.size(data))


def set_cell_K(machine, beamline, und_type, und_loc, und_appendix, K_set_val):
    path_cell_str = path_cell(machine=machine, beamline=beamline, und_type=und_type, und_loc=und_loc, und_appendix=und_appendix)
    K_rbv = pydoocs.read(path_cell_str+'/K')['data']
    #print(path_cell_str, 'rbv_old={}'.format(K_rbv))
    pydoocs.write(path_cell_str+'/CONTROL.USE_K',1)
    time.sleep(0.1)
    pydoocs.write(path_cell_str+'/CONTROL.USE_K',0)
    pydoocs.write(path_cell_str+'/K.SET', K_set_val)
    pydoocs.write(path_cell_str+'/CONTROL.START',1)
    time.sleep(0.1)
    pydoocs.write(path_cell_str+'/CONTROL.START',0)
    
def get_cell_K(machine, beamline, und_type, und_loc, und_appendix):
    path_cell_str = path_cell(machine=machine, beamline=beamline, und_type=und_type, und_loc=und_loc, und_appendix=und_appendix)
    K_rbv = pydoocs.read(path_cell_str+'/K')['data']
    return(K_rbv)


def find_cell_loc_idx(und_cell_nr, und_cell_nr_arr):
    idx = np.where(np.array(und_cell_nr_arr)==und_cell_nr)[0][0]
    return idx
    #return(und_loc_arr[idx])

def cell_startstop_list(und_cell_nr_arr, cell_left,cell_right):
    if cell_left > cell_right:
        cell_left, cell_right = cell_right, cell_left
    idx_l, idx_r = find_cell_loc_idx(cell_left, und_cell_nr_arr), find_cell_loc_idx(cell_right, und_cell_nr_arr)
    return np.take(und_loc_arr, np.arange(idx_l,idx_r+1,1))


# for cell in cells_to_use_arr:
#     print('')
#     print(cell)
#     K_rbv = get_cell_K(machine=machine, beamline=beamline, und_type=und_type, und_loc=cell, und_appendix=und_appendix)
#     print('rbv_old={}'.format(K_rbv))
    

def measure_I(I_address=address_xgm_mean_raw, N_trains=10):
    I_hist = []
    #plt.figure(I_address)
    #plt.clf()
    while len(I_hist)<N_trains:
        I_hist.append(pydoocs.read(I_address)['data'])
        #plt.plot(np.array(I_hist))#/np.array(I_hist)[0])
        #plt.show()
        time.sleep(0.1)
    return I_hist

def set_and_track_cell_K(machine, beamline, und_type, und_loc, und_appendix, K_set_val, I_address, epsilon=0.05, N_trains=10):
    def inepsilon_check(val_init, val_read, epsilon):
    	#check if relative undulator detuning is small (in epsilon region)
        if abs(val_read-val_init)/val_init < epsilon:
            return True
        else:
            return False
    K_hist = []
    I_hist = []
    I_hist_save = []
    status=0 #0 - wait, 1=measure, 2=exit
    mode = 0
    K_rbv = get_cell_K(machine=machine, beamline=beamline, und_type=und_type, und_loc=und_loc, und_appendix=und_appendix)
    K_hist.append(K_rbv)
    I_hist.append(pydoocs.read(I_address)['data'])
    if inepsilon_check(K_hist[0], K_set_val, 0.001):
        #print('undulator in epsilon range already')
        mode = 2
        return np.nan
    print('  cell ' + str(und_loc))
    print('  K: {:.4f} -> {:.4f}'.format(K_rbv, K_set_val))
    set_cell_K(machine, beamline, und_type, und_loc, und_appendix, K_set_val)
    while mode < 2:
        #print(K_rbv)
        time.sleep(0.1)
        K_rbv = get_cell_K(machine=machine, beamline=beamline, und_type=und_type, und_loc=und_loc, und_appendix=und_appendix)
        I = pydoocs.read(I_address)['data']##################
        K_hist.append(K_rbv)
        I_hist.append(I)
        if mode > 0: I_hist_save.append(I)
        #print('mode=',mode)
        if not inepsilon_check(K_hist[0],K_rbv,epsilon): #start reading
            mode = 1
            #print('  acquiring data from cell ' + str(und_loc))
        if len(I_hist_save) > N_trains:
            mode = 2
            #print('    acquisition done')
            return I_hist_save
        if False:
            #plt.figure(und_loc)
            plt.figure('relative detuning response')
            #plt.clf()
            plt.ylabel('uJ')
            plt.ylabel('relative K detuning')
            plt.plot(np.array(K_hist)/K_hist[0], np.array(I_hist), label=und_loc)#/np.array(I_hist)[0])
            plt.legend()
            plt.show()
        
#%%

cells_to_use_arr = cell_startstop_list(und_cell_nr_arr, cell_range[0], cell_range[1])

#K_arr = []
#for und_loc in und_loc_arr:
#    path_str_tmp = path_cell(machine, beamline, und_type, und_loc, und_appendix)+'/K'
#    print(path_str_tmp)
#    K_rbv = pydoocs.read(path_str_tmp)['data']
#    print(K_rbv)
#    K_arr.append(K_rbv)
#
#plt.figure('current_taper')
#plt.clf()
#plt.plot(und_loc_arr, K_arr,marker='*')
#plt.show()

N_bunches_intrain = pydoocs.read(address_Nbunches)['data']

t0 = time.time()
I_arr = []
I_err = []
print(' ')
print('Cell contribution scan in ' + beamline)

t_estimate = N_trains_to_read * (len(cells_to_use_arr)+1) / 10
m, s = divmod(t_estimate, 60)
h, m = divmod(m, 60)
print('estimated scan time:  {:.0f}h {:.0f}m {:.0f}s'.format(h,m,s))
print('Measuring initial pulse energy')
I = measure_I(I_address=I_address, N_trains=N_trains_to_read)
I_arr.append(np.mean(I))
I_err.append(stderr(I))
print('  <I> = {:.2f} ± {:.2f}'.format(np.mean(I), stderr(I)))
cells_to_use_arr = np.flip(cells_to_use_arr)
K_last_closed_cell = get_cell_K(machine=machine, beamline=beamline, und_type=und_type, und_loc=cells_to_use_arr[-1], und_appendix=und_appendix)
for cell_nr, cell in enumerate(cells_to_use_arr):
    print("{:.1f}% done".format((cell_nr+1)/(len(cells_to_use_arr)+1)*100))
    if cell == cells_to_use_arr[-1]:
        pass
    if K_to_open_enforce == None:
        K_to_open = K_last_closed_cell * (1 - K_to_open_offset - K_to_open_taper * cell_nr)
    else:
        K_to_open = K_to_open_enforce * (1 - K_to_open_taper * cell_nr)
    #K_to_open = K_last_cell * (1 - K_to_open_offset - K_to_open_taper * cell_nr) #detune last open cell to 10% K detuning with additional 1% K offset per cell (negative taper)
    I=set_and_track_cell_K(machine=machine, beamline=beamline, und_type=und_type, und_loc=cell, und_appendix=und_appendix, K_set_val=K_to_open, I_address=I_address, epsilon=0.01, N_trains=N_trains_to_read)
    I_arr.append(np.mean(I))
    I_err.append(stderr(I))
    print('  <I> = {:.2f} ± {:.2f}'.format(np.mean(I), stderr(I)))

I_arr = np.array(I_arr)
I_err = np.array(I_err)

#%%

if subtract_last_msrmnt_as_bkg:
    print('Background subtracted=', I_arr[-1])

if subtract_last_msrmnt_as_bkg:
    I_arr_plot = I_arr - I_arr[-1]
else:
    I_arr_plot = I_arr

emit = np.array(I_arr_plot)[:-1]-np.array(I_arr_plot)[1:]
gain = np.array(I_arr_plot)[:-1]/np.array(I_arr_plot)[1:]
emit_err = np.abs(np.array(I_err)[:-1]+np.array(I_err)[1:])
gain_err = np.abs(gain*( (np.array(I_err)[:-1]/np.array(I_arr_plot)[:-1])**2 +  (np.array(I_err)[1:]/np.array(I_arr_plot)[1:])**2   )**0.5)

t1 = time.time()
m, s = divmod((t1-t0), 60)
h, m = divmod(m, 60)

print('100.0% DONE')
print(' ')
print('Cell contribution scan in ' + beamline)
print('scan time:  {:.0f}h {:.0f}m {:.0f}s'.format(h,m,s))
#print('time = {}m'.format((t1-t0)/60))
print('Statistics = N_trains * Nbunches: ', N_trains_to_read, '*', N_bunches_intrain, '=', N_trains_to_read*N_bunches_intrain)
print('Intensity channel: ', I_address)
print('cells opened:', len(cells_to_use_arr), cells_to_use_arr)
print('I=', I_arr_plot)
print('Contribution/cell=', emit)
print('Gain/cell=', gain_err)

print('Contribution/cell error=', emit_err)
print('Gain/cell error=', gain)



#cells_to_use_arr_extended = np.insert(cells_to_use_arr, 0, cells_to_use_arr[0]+1)
cells_to_use_arr_extended = np.append(cells_to_use_arr, cells_to_use_arr[-1]-1)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True,figsize=(7,10),num='Cell contribution '+beamline)



#def forward(x):
#    return np.interp(x, np.array(und_cell_nr_arr), np.array(und_loc_arr))
#def reverse(x):
#    return np.interp(x, np.array(und_loc_arr), np.array(und_cell_nr_arr))
    
def forward(x):
    return np.interp(x, [und_cell_nr_arr[0]-2, und_cell_nr_arr[-1]+2], [und_loc_arr[0]-12, und_loc_arr[-1]+12])
def reverse(x):
    return np.interp(x, [und_loc_arr[0]-12, und_loc_arr[-1]+12], [und_cell_nr_arr[0]-2, und_cell_nr_arr[-1]+2])

np.save(args.savedir+"I_arr_plot", np.array(I_arr_plot))
np.save(args.savedir+"emit", emit)
np.save(args.savedir+"emit_err", emit_err)
np.save(args.savedir+"gain", gain)
np.save(args.savedir+"gain_err", gain_err)
np.save(args.savedir+"I_err", I_err)
np.save(args.savedir+"cells_to_use", cells_to_use_arr)
np.save(args.savedir+"cells_to_use_extended", cells_to_use_arr_extended)

secax = ax1.secondary_xaxis('top', functions=(reverse, forward))
secax.set_xlabel('cell')

ax1.step(cells_to_use_arr_extended, np.array(I_arr_plot), where='mid', linewidth=3)
ax1.errorbar(cells_to_use_arr_extended, np.array(I_arr_plot), yerr=I_err, fmt='none')
ax1.set_title('I before opening cell # I')
ax2.step(cells_to_use_arr, emit, where='mid')
ax2.fill_between(cells_to_use_arr, emit, step='mid', alpha=0.4)
ax2.errorbar(cells_to_use_arr, emit, yerr=emit_err, fmt='none')
ax2.set_title('Contribution of cell # I(n) - I(n-1)')
ax3.step(cells_to_use_arr, gain, where='mid')
ax3.fill_between(cells_to_use_arr, gain, np.ones_like(cells_to_use_arr), step='mid', alpha=0.4)
ax3.errorbar(cells_to_use_arr, gain, yerr=gain_err, fmt='none')
ax3.set_title('Gain of cell # I(n) / I(n-1)')
ax3.set_xlabel('z, [m]')

fig.savefig(args.savedir+"gain_scan_output.png")

#%%