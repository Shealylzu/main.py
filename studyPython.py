import copy
import random
import time
import math
import matplotlib.pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import numpy as np
from numpy.random import normal
from scipy.signal import gausspulse
from gwpy.timeseries import TimeSeries
from gwpy.detector import units
from astropy.units import quantity
from gwpy.timeseries.io import losc
import h5py
import librosa
import librosa.display
import matplotlib as mpl

#----------------------generate glitch------------------------------
noise = TimeSeries(normal(loc=1,size=4096*4),sample_rate=4096,epoch=-2)
snum = 100 #amount of small glitches in a large glitch
lnum = 1 #amount of large glitches

global p0
p0 = 0
global data
data = 0

for j in range(0,lnum):
    #t0 = random.uniform(-0.9,0.9)  #the center time of a large glitch
    t0 = 0
    fc = random.uniform(300, 800)  # center frequency [100,1000]
    bw = random.uniform(0.1, 1)  # bandwidth [0.1,1]
    bwr = random.uniform(-10, 0)  # Reference level at whi
    # ch fractional bandwidth is calculated (dB) [-10,0]
    # tpr = random.uniform(-80, 40)  # the cutoff time for when the pulse amplitude falls below `tpr` [-80,-40]
    #a = random.uniform(100, 600) # the coefficient of the gausspluse,[10,60]
    # for i in range(0,snum):
    # t = random.uniform(-0.02,0.02) #time of random deviation [-0.02,0.02]
    p1 = 40*gausspulse(noise.times.value , fc=300, bw=bw, bwr=bwr) #+ t0 -t  , tpr=tpr
    p0 = p0 + p1
    # p2=gausspulse(noise.times.value,fc=500,bw=1,bwr=-6,tpr=-50)*70
    # p3=gausspulse(noise.times.value+0.005,fc=600,bw=0.40,bwr=-8,tpr=-80)*40
    # p4=gausspulse(noise.times.value-0.015,fc=300,bw=0.35,bwr=-7,tpr=-40)*30
    glitch = TimeSeries(p0, sample_rate=4096)
    data = noise + glitch
#print(data)

plot1 = data.plot()

bx = plot1.gca()
#     ax.set_xscale('seconds')
#     #ax.set_yscale('log')
#     ax.set_ylim(0, 350)
#     ax.set_ylabel('Frequency[hz]')
bx.set_xlim(-0.2,0.2)
bx.set_epoch(0)
bx.grid(False)
# norm = mpl.colors.Normalize(vmin=1,vmax=350)
# bx.colorbar(cmap='viridis', label='Normalized energy',norm=norm) #,
#     plot.savefig("pict" + str(i) + ".png")
#     print("pict" + str(i))
plot1.savefig("timeseries" + ".png")
plot1.show()

# q = data.q_transform()
# plot = q.plot() # figsize=[8,4]
# ax = plot.gca()
# #     ax.set_xscale('seconds')
# #     #ax.set_yscale('log')
# #     ax.set_ylim(0, 350)
# #     ax.set_ylabel('Frequency[hz]')
# ax.set_xlim(-1, 1)
# ax.set_epoch(0)
# ax.grid(False)
# norm = mpl.colors.Normalize(vmin=1,vmax=350)
# ax.colorbar(cmap='viridis', label='Normalized energy',norm=norm) #,
# plot.savefig("spectrum" + ".png")
# #     print("pict" + str(i))
# plot.show()

#
# # ---------------------plot by librosa----------------------------------
# # f = h5py.File('a.hdf5', 'r')
# # # dset = f['strain' + '/' + 'Strain']
# # # data = TimeSeries(dset, t0=1187008512.0, sample_rate=4096)
# d = data.value
# # # d = d[362 * 4096:372 * 4096]
# # Straindset = f['strain/Strain']
# # d = Straindset[345*4096:375*4096]
# sr = 4096
# c = np.abs(librosa.cqt(d, fmin=32, hop_length=108,sr=sr, n_bins=72, sparsity=0.01,bins_per_octave=18,filter_scale=1.8,scale=False))
# fig, ax = plt.subplots(figsize=[8,4])
# threshold = 300
# c[c > threshold] = threshold
#
# img = librosa.display.specshow(c, sr=sr, bins_per_octave=18,hop_length=120, x_axis="s", y_axis='cqt_hz', ax=ax)
# # img = librosa.display.specshow(librosa.amplitude_to_db(c, ref=np.max),sr=sr,x_axis='time',y_axis='cqt_note', ax=ax)
# # img = librosa.display.specshow(librosa.power_to_db(c, ref=np.max),sr=sr,x_axis='time',y_axis='cqt_hz', ax=ax)
# # norm = mpl.colors.Normalize(vmin=-5,vmax=0)
# # fig.colorbar(img, ax=ax,format="%+2.0f db")
# # tmp=copy.copy(matplotlib.cm.viridis)
# # fig.colorbar(img, cmap=matplotlib.colormaps['viridis'], ax=ax, format="%+2.0f")
# print(img.__class__)
# img.set_cmap(matplotlib.colormaps['viridis'])
# fig.colorbar(
#     mappable=img,
#     # mappable=matplotlib.cm.ScalarMappable(cmap=matplotlib.colormaps['viridis']),
#     ax=ax, format="%+2.0f")
# plt.viridis()
# ax.grid(False)
# #ax.set_ylim(0, 512)
# plt.show()

#-----------------test glitchen---------------------------
# #-------------Burst and glitch example-----------------------
# #by default training will use a 0.5s window, centered.
#
# W, Z = train_ppca('./data/L1O3a_Tomtes_10-128.npy', q=5, plots=True)
# noiseseg = np.load('./data/L1O3a_ex_noise_64s_2048.npy')
# noiseseg = noiseseg[0]
# seglen = 64.0
# times = np.linspace(-seglen/2, seglen/2, len(noiseseg))
# noiseseg_ts = TimeSeries(noiseseg, times=times)
# # noiseseg_ts = gen_noiseseg(seglen)
# asd = noiseseg_ts.asd()
# #just to obtain an asd to whiten with
#
# #can generate gaussian noise or inject into the real noise
# def gen_noiseseg(seglen, srate=2048.):
#     draws = np.random.normal(size=int(seglen*srate))
#     times = np.linspace(-seglen/2, seglen/2, int(seglen*srate))
#     noiseseg_ts = TimeSeries(draws, times=times)
#     return noiseseg_ts
#
# csdata = np.loadtxt('./data/cusp11.txt')
# #this is a cosmic string waveform generate with LALsim.
# plt.plot(csdata)
#
#
# highpass = 10. #it's necessary to highpass, so that the signal isn't dominated by low-f content
# #which isnt in the sensitive band of the detector, and whitening isnt effectively removing.
# datasrate = 16384.
# dataseglen = len(csdata)/datasrate
# times = np.linspace(-dataseglen/2, dataseglen/2, len(csdata))
# cs_timeseries = TimeSeries(csdata, times=times)
# cs_timeseries = cs_timeseries.resample(2048)
# csh = cs_timeseries.highpass(highpass)
# cst_white = csh.whiten(asd=asd) #whiten using sample L1 noise
# srate = 2048
# noise = gen_noiseseg(len(cst_white)/srate)
# inj = noise + cst_white
# #injecting
# plt.figure(figsize=(12,4))
# plt.plot(inj, label='waveform+noise')
# plt.plot(cst_white, color='red', label='Cosmic string waveform')
# plt.xlim(-0.2,0.2)
# plt.legend()
# plt.show()
#
# glitchrec = maxL_reconst(W, inj, fixed=False, gfrac=0.1, metric='ip')
# glitchrects = TimeSeries(glitchrec, times=np.linspace(-len(cst_white)/srate/2,len(cst_white)/srate/2, int(len(cst_white))))
# plt.figure(figsize=(12,4))
# plt.plot(inj, label='Cosmic string injection')
# plt.plot(glitchrects, label='glitschen tomte reconstruction')
# plt.xlim(-.15,.15)
# mf_snr(inj, glitchrects)
# mf_snr(cst_white, cst_white)
# mf_snr(inj, glitchrects)/mf_snr(cst_white, cst_white)
# plot_corner_maxl(Z, W, inj)
#
# outlier_score('./data/L1O3a_Tomtes_10-128.npy', inj)
# test_fd = np.fft.rfft(inj)
# kdeprior = kombine.clustered_kde.optimized_kde(Z[:-1].T)
# post = Posterior(test_fd, W, 2048, prior=kdeprior)
# q=5
# nwalkers=128
# N=1000
# p0 = initializer(nwalkers, timescale=0.2)
# sampler = kombine.Sampler(nwalkers, q+1, post)
# sampler.run_mcmc(N=N, p0=p0, update_interval=100)
# for i in range(0,q+1): #look at the chains
#     plt.figure()
#     plt.plot(sampler.chain[:,:,i])
# samps=sampler.get_samples()
# np.save('./data/CS_template_test_tomtemodel.npy', samps)
# thinsamps = samps#[::10] optional thinning
# samps.shape
# #builds the timeseries reconstructions from the samples
# reconst_td = build_reconstructions(W, thinsamps, post)
# logls = gen_logls(samps, post)
# snrs = gen_snrs(samps, post)
# plot_corner_sample(Z, samps[::2], W, inj, t_ylim=30)
# tlen = len(inj) / srate
# times = np.linspace(-tlen / 2, tlen / 2, num=int(tlen * srate))
# plt.figure(figsize=(13, 6), dpi=400)
# plt.plot(times, inj, label='data', color = 'blue')
# plt.plot(times, reconst_td.T, alpha=0.05, color='orange')
# plt.plot(times, maxL_reconst(W, inj, fixed=False, gfrac=0.1, metric='ip'),color='springgreen', label='max$\mathcal{L}$ reconstruction');
# plt.ylabel('whitened amplitude')
# plt.xlim(-0.1, 0.1)
# dic(logls)
# plt.hist(snrs)



# import re  #regular expression
# f = open('a')
# pattern = ('A.C')
# for line in f:
#     name = line.strip()  #去除字符串前后的空格
#     result = re.search(pattern,name)
#     if result:
#         print('Find in {}'.format(name))
# f.close
