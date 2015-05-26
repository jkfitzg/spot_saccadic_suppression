from neo.io import AxonIO
import numpy as np
from scipy.io import loadmat
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter 
from scipy import signal
from scipy.stats import circmean, circstd
from plotting_help import *
import sys, os
import scipy.signal
from bisect import bisect
import cPickle
import math
import pandas as pd
import scipy as sp

#---------------------------------------------------------------------------#

class Flight():  
    def __init__(self, fname, protocol='wing beat analyzer'):
    # get filename and protocol type for experiment
    # save details
     
        if fname.endswith('.abf'):
            self.basename = ''.join(fname.split('.')[:-1])
            self.fname = fname
        else:
            self.basename = fname
            self.fname = self.basename + '.abf'  #check here for fname type 
        
        self.protocol = protocol
        
        if self.protocol == 'wing beat analyzer':
            self.sampling_rate = 1000
        elif self.protocol == 'optical tracking':
            self.sampling_rate = 10000
                  
    def open_abf(self,exclude_indicies=[]): 
    # input: optional list of time points to exclude
    # saves information from all channels into numpy arrays
    
        abf = read_abf(self.fname)              
        
        # different channel names for wing beat analyzer versus
        # strokelitude, although same file structures
        # assumes all channels have the same sample #s 
        
        if self.protocol == 'wing beat analyzer': 
            n_indicies = np.size(abf['stim_x'])      
            inc_indicies = np.setdiff1d(range(n_indicies),exclude_indicies);
                   
            self.xstim = np.array(abf['stim_x'])[inc_indicies]
            self.ystim = np.array(abf['stim_y'])[inc_indicies]

            # no wing processing here
            self.lwa = np.array(abf['l_wba'])[inc_indicies]
            self.rwa = np.array(abf['r_wba'])[inc_indicies]
        
            self.ao = np.array(abf['ao1'])[inc_indicies]
        
            self.vm = np.nan*np.ones_like(self.lwa)     #empty
            self.tach = np.array([])                    #empty
            
        elif self.protocol == 'optical tracking':
            n_indicies = np.size(abf['x_ch'])  
            inc_indicies = np.setdiff1d(range(n_indicies),exclude_indicies);
               
            self.xstim = np.array(abf['x_ch'])[inc_indicies]
            self.ystim = np.array(abf['y_ch'])[inc_indicies]

            lwa_v = np.array(abf['wba_l'])[inc_indicies]
            rwa_v = np.array(abf['wba_r'])[inc_indicies]    
            
            # process wing signal
            self.lwa = process_wings(lwa_v)
            self.rwa = process_wings(rwa_v)
    
            self.ao = np.array(abf['patid'])[inc_indicies]
    
            self.vm = np.array(abf['vm'])[inc_indicies] - 13 #offset for bridge potential
            self.tach = np.array(abf['tach'])[inc_indicies]
        
        # common to both versions    
        self.samples = np.arange(self.xstim.size)  
        self.t = self.samples/float(self.sampling_rate)
        self.lmr = self.lwa - self.rwa
        self.iti_s = .5         # later change this for the 5 ms iti trials
            
    def _is_flying(self, start_i, stop_i, wba_thres=0.5, flying_time_thresh=0.95):  
        # return whether the fly is flying within an index range
        # written for wing beat analyzer signals
        
        l_nonzero_samples = np.where(self.lwa[start_i:stop_i] > wba_thres)[0]
        r_nonzero_samples = np.where(self.rwa[start_i:stop_i] > wba_thres)[0]
        n_flying_samples = np.size(np.intersect1d(l_nonzero_samples,r_nonzero_samples))
        
        total_samples = stop_i-start_i
        
        is_flying = (float(n_flying_samples)/total_samples) > flying_time_thresh   
        return is_flying
        
           
#---------------------------------------------------------------------------#

class Spot_Saccadic_Supression(Flight):
    
    def process_fly(self, show_tr_time_parsing=False, ex_i=[]): 
        # create data structure for the experiment
        
        self.open_abf(ex_i)
        self.parse_trial_times(show_tr_time_parsing)
        self.parse_stim_type()
          
    def remove_non_flight_trs(self, iti=750):
        # loop through each trial and determine whether fly was flying continuously
        # if a short nonflight bout (but not during turn window), interpolate
        #
        # delete the trials with long nonflight bouts--change n_trs, tr_starts, 
        # tr_stops, looming stim on
        
        non_flight_trs = [];
        
        for tr_i in range(self.n_trs):
            this_tr_start = self.tr_starts[tr_i] - iti
            this_tr_stop = self.tr_stops[tr_i] + iti
            
            if not self._is_flying(this_tr_start,this_tr_stop):
                non_flight_trs.append(tr_i) 
        
        #print 'nonflight trials : ' + ', '.join(str(x) for x in non_flight_trs)
        print 'nonflight trial counts : ' + str(np.size(non_flight_trs)) + '/' + str(self.n_trs)
        
        #now remove nonflight trials
        self.n_nonflight_trs = np.size(non_flight_trs)
        self.n_trs = self.n_trs - np.size(non_flight_trs)
        self.tr_starts = np.delete(self.tr_starts,non_flight_trs)  
        self.tr_stops = np.delete(self.tr_stops,non_flight_trs)
                    
    def parse_trial_times(self, if_debug_fig=False):
        # parse the ao signal to determine trial start and stop index values
        # include checks for unusual starting aos, early trial ends, 
        # long itis, etc
        
        ao_diff = np.diff(self.ao)
        
        ao_d_upper_bound = 5
        ao_d_lower_bound = -5
        
        tr_start = self.samples[np.where(ao_diff > ao_d_upper_bound)]
        start_diff = np.diff(tr_start)
        redundant_starts = tr_start[np.where(start_diff < 1000)]
        clean_tr_start_candidates = np.setdiff1d(tr_start,redundant_starts)+1
        clean_tr_starts = clean_tr_start_candidates[np.where(self.ao[clean_tr_start_candidates+50] > 2)]
        
        tr_stop = self.samples[np.where(ao_diff < ao_d_lower_bound)]
        stop_diff = np.diff(tr_stop)
        redundant_stops = tr_stop[np.where(stop_diff < 1000)] 
        # now check that the y value is > -9 
        clean_tr_stop_candidates = np.setdiff1d(tr_stop,redundant_stops)+1
        
        clean_tr_stops = clean_tr_stop_candidates[np.where(self.ao[clean_tr_stop_candidates-50] > 2)]
        
        # check that first start is before first stop
        if clean_tr_stops[0] < clean_tr_starts[0]: 
            clean_tr_stops = np.delete(clean_tr_stops,0)
         
        # last stop is after last start
        if clean_tr_starts[-1] > clean_tr_stops[-1]:
            clean_tr_starts = np.delete(clean_tr_starts,len(clean_tr_starts)-1)
         
        # I should check for same # of starts and stops
        n_trs = len(clean_tr_starts)
        
        if if_debug_fig:
            figd = plt.figure()
            plt.plot(self.ao)
            plt.plot(ao_diff,color=magenta)
            y_start = np.ones(len(clean_tr_starts))
            y_stop = np.ones(len(clean_tr_stops))
            plt.plot(clean_tr_starts,y_start*7,'go')
            plt.plot(clean_tr_stops,y_stop*7,'ro')
            plt.plot(self.xstim,color=black)
            plt.plot(np.diff(self.ao),color=magenta)
        
        
        self.n_trs = n_trs 
        self.tr_starts = clean_tr_starts  #index values of starting and stopping
        self.tr_stops = clean_tr_stops
        
        ## remove all trials in which the fly is not flying. 
        #self.remove_non_flight_trs()
        
    def parse_stim_type(self):
        #calculate, save the stimulus type of each trial
        
        self.stim_types_labels = {24:'Spot on right, front to back',\
                            25:'Spot on right, back to front',\
                            38:'Spot on left, back to front',\
                            39:'Spot on left, front to back' ,\
                            44:'Spot on right, .5 p offset, front to back',\
                            46:'Spot on left, .5 p offset, front to back'}
        # patterns 24, 25, 38, 39 are all 1 panel offset from the center
        # patterns 44 and 46 are 0.5 panel offset from the center
       
        stim_types = -1*np.ones(self.n_trs,'int')
    
        start_offset = 100
        stop_offset = -100
        
        # loop through to get the ao values
        for tr in range(self.n_trs): 
            this_start = self.tr_starts[tr]+start_offset
            this_stop = self.tr_stops[tr]+stop_offset
                
            stim_types[tr] = round(10*np.nanmean(self.ao[this_start:this_stop])) 
            
            #if stim_types[tr] not in self.all_stim_types:
            #    print 'tr' + str(tr) + ' = ' + str(stim_types[tr]) + ', removing trial'
            #    
            #    self.tr_starts = np.delete(self.tr_starts,tr)
            #    self.tr_stops = np.delete(self.tr_stops,tr)
            #    self.n_trs = self.n_trs - 1
            
        self.unique_stim_types = np.unique(stim_types) 
        # print 'trial types = ' + str(self.unique_stim_types)
        
        self.stim_types = stim_types  #change to integer, although nans are also useful
        
    def plot_wba_by_cnd(self,title_txt='',long_static_spot=False,wba_lim=[-1.5,1.5],
            filter_cutoff=48,tr_range=slice(None), if_save=True): 
        # plot stacked single traces of each of the four saccadic movement conditions
        
        if self.protocol == 'optical tracking':
            wba_lim = [-30,30]
            filter_cutoff = 16
        
        sampling_rate = self.sampling_rate  
        s_iti = .25 * sampling_rate  # I'm taking a subset of the iti here  
        
        baseline_win = range(0*sampling_rate,int(s_iti/2.0)) 
        
        # get all traces _____________________________________________
        # now not detecting saccades automatically. 
        # all_fly_saccades is blank
        all_fly_traces, all_fly_saccades = self.get_traces_by_stim('this_fly',s_iti,get_saccades=False)
        
        # set up axes _____________________________________________     
        n_cols = 4 
        n_rows = 2
    
        cnds_to_plot = self.unique_stim_types
        gs = gridspec.GridSpec(n_rows,n_cols,height_ratios=[1,.1])

        #fig = plt.figure(figsize=(16.5, 9))
        fig = plt.figure(figsize=(13.5, 10))
        
        gs.update(wspace=0.03, hspace=0.03) # set the spacing between axes. 
        
        # store all subplots for formatting later           
        all_wba_ax = np.empty(n_cols,dtype=plt.Axes)
        all_stim_ax = np.empty(n_cols,dtype=plt.Axes)
        
        
        # loop through the conditions/columns ____________________________________
        for col in range(n_cols):
            cnd = cnds_to_plot[col]
           
            this_cnd_trs = all_fly_traces.loc[:,('this_fly',tr_range,cnd,'lmr')].columns.get_level_values(1).tolist()
            n_cnd_trs = np.size(this_cnd_trs)
            
            # get colormap info ______________________________________________________
            cmap = plt.cm.get_cmap('spectral') 
            cNorm  = colors.Normalize(0,n_cnd_trs)
            scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)

            # create subplots ________________________________________________________              
            if col == 0:
                wba_ax  = plt.subplot(gs[0,col]) 
                stim_ax = plt.subplot(gs[1,col],sharex=wba_ax)    
            else:
                wba_ax  = plt.subplot(gs[0,col], sharex=all_wba_ax[0],  sharey=all_wba_ax[0]) 
                stim_ax = plt.subplot(gs[1,col], sharex=all_stim_ax[0], sharey=all_stim_ax[0])    
    
            all_wba_ax[col] = wba_ax
            all_stim_ax[col] = stim_ax
    
            # loop through single trials and plot all signals ________________________________
            for tr, i in zip(this_cnd_trs,range(n_cnd_trs)):

                this_color = this_color = scalarMap.to_rgba(i)     
     
                # plot WBA signal ____________________________________________________           
                wba_trace = all_fly_traces.loc[:,('this_fly',tr,cnd,'lmr')]
    
                baseline = np.nanmean(wba_trace[baseline_win])
                wba_trace = wba_trace - baseline  
     
                non_nan_i = np.where(~np.isnan(wba_trace))[0] # trials are buffered by nans at the end
                filtered_wba_trace = butter_lowpass_filter(wba_trace[non_nan_i],cutoff=filter_cutoff,fs=self.sampling_rate)
                 
                wba_ax.plot(filtered_wba_trace,color=this_color)
                #wba_ax.plot(wba_trace,color=this_color)
               
                # plot stimulus traces ____________________________________________
                stim_ax.plot(all_fly_traces.loc[:,('this_fly',tr,cnd,'xstim')],color=this_color)
                
                
            # plot the condition mean ____________________________________________
            mean_wba_trace = np.nanmean(all_fly_traces.loc[:,('this_fly',this_cnd_trs,cnd,'lmr')],1)
            baseline = np.nanmean(mean_wba_trace[baseline_win])
            mean_wba_trace = mean_wba_trace - baseline  
            
            non_nan_i = np.where(~np.isnan(mean_wba_trace))[0] 
            filtered_mean = butter_lowpass_filter(mean_wba_trace[non_nan_i],cutoff=filter_cutoff)
            wba_ax.plot(filtered_mean,color=black,linewidth=3)
            
            wba_ax.axhline()
                            
        # format all subplots _____________________________________________________  
   
        # loop though all columns again, format each row ______________________________
        for col in range(n_cols):      
            
            # show turn window
            if long_static_spot:
                all_wba_ax[col].axvspan(2365, 2465, facecolor='grey', alpha=0.5)    
            elif self.protocol == 'optical tracking':
                all_wba_ax[col].axvspan(13700, 14700, facecolor='grey', alpha=0.5)
            else:
                all_wba_ax[col].axvspan(1370, 1470, facecolor='grey', alpha=0.5)    
            
            
            
            
            # remove all time xticklabels __________________________________
            all_wba_ax[col].tick_params(labelbottom='off')
            
            # label columns
            all_wba_ax[col].set_title(self.stim_types_labels[cnds_to_plot[col]],fontsize=12)
            
            
            if col == 0:           
                if self.protocol == 'optical tracking':
                    all_wba_ax[col].set_ylabel('L-R WBA (degrees)',fontsize=12)
                else:
                    all_wba_ax[col].set_ylabel('L-R WBA (V)',fontsize=12)
            
                all_wba_ax[col].set_ylim(wba_lim)
                all_wba_ax[col].set_yticks([wba_lim[0],0,wba_lim[1]])
                
                all_stim_ax[col].set_ylabel('stim frame',fontsize=9)
                all_stim_ax[col].tick_params(labelleft='off')
                all_stim_ax[col].tick_params(labelbottom='off')
            
                # label time x axis for just col 0 ______________________
                # divide by sampling rate _______________________________
                def div_sample_rate(x, pos): 
                    #The two args are the value and tick position 
                    #return (x-(s_iti/10))/(sampling_rate/10)
                    return x/(sampling_rate)
     
                formatter = FuncFormatter(div_sample_rate) 
                
                if long_static_spot:
                    all_wba_ax[col].set_xlim([0,2.775*sampling_rate]) #enforce max time
                else:
                    all_wba_ax[col].set_xlim([0,1.775*sampling_rate]) #enforce max time
                
                
                all_stim_ax[col].xaxis.set_major_formatter(formatter)
                all_stim_ax[col].tick_params(labelbottom='on')
                all_stim_ax[col].tick_params(labelleft='off')
                all_stim_ax[col].locator_params(tight=True, nbins=5)
                all_stim_ax[col].set_xlabel('Time (s)')
            else:
                all_wba_ax[col].tick_params(labelleft='off')
                all_stim_ax[col].tick_params(labelleft='off')
                all_stim_ax[col].tick_params(labelbottom='off')
        
        figure_txt = title_txt
        fig.text(.2,.95,figure_txt,fontsize=18) 
       
        plt.draw()

        if if_save:
            saveas_path = '/Users/jamie/bin/figures/overlaid trials/'
            plt.savefig(saveas_path + figure_txt + '_sacc_supression_wba_by_cnd.png',\
                            bbox_inches='tight',dpi=100) 

    def plot_wba_by_cnd_y_offset(self,title_txt='',long_static_spot=False,diff_thres=0.02,trs_to_mark=[],\
                    tr_range=slice(None),filter_cutoff=48,if_save=True): 
        # plot single trace of each of the four saccadic movement conditions
        
        
        if np.size(trs_to_mark):
            saccade_stim_trs = [s[0:2] for s in trs_to_mark] 
        else:
            saccade_stim_trs = []
            
        sampling_rate = self.sampling_rate
        s_iti = .25 * sampling_rate      
        tr_offset = 1 #3.0 
        
        if self.protocol == 'optical tracking':
            filter_cutoff = 16
            tr_offset = 1.0/10
            diff_thres=10
        
        #baseline_win = range(0*sampling_rate,int(.125*sampling_rate)) 
        baseline_win = range(0*sampling_rate,int(.1*sampling_rate)) 
        
        # get all traces and detect saccades ______________________________________________
        all_fly_traces, all_fly_saccades = self.get_traces_by_stim('this_fly',s_iti,get_saccades=False)
        
        # discard the traces earlier in time.
        # this value is somewhat arbitrary, but saccade times are relative it.
        new_start_t = int(3*s_iti) 
        all_fly_traces = all_fly_traces.ix[new_start_t:,slice(None)]
        new_index = np.array(all_fly_traces.index) - new_start_t
        all_fly_traces.index = new_index
        
             
        n_cols = 4 
        n_rows = 2
        
        cnds_to_plot = self.unique_stim_types
        gs = gridspec.GridSpec(n_rows,n_cols,height_ratios=[1,.05])

        fig = plt.figure(figsize=(13.5, 10))
        gs.update(wspace=0.03, hspace=0.03) # set the spacing between axes.  
        
        #store all subplots for formatting later           
        all_wba_ax = np.empty(n_cols,dtype=plt.Axes)
        all_stim_ax = np.empty(n_cols,dtype=plt.Axes)
        
        #set order of stimuli to plot
        
        # loop through the conditions/columns ____________________________________
        for col in range(n_cols):
            cnd = cnds_to_plot[col]
           
            this_cnd_trs = all_fly_traces.loc[:,('this_fly',tr_range,cnd,'lmr')].columns.get_level_values(1).tolist()
            n_cnd_trs = np.size(this_cnd_trs)
            
            # get colormap info ______________________________________________________
            cmap = plt.cm.get_cmap('spectral')  #('gnuplot2')  
            cNorm  = colors.Normalize(0,n_cnd_trs)
            scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)

            # create subplots ________________________________________________________              
            if col == 0:
                wba_ax  = plt.subplot(gs[0,col]) 
                stim_ax = plt.subplot(gs[1,col],sharex=wba_ax)    
            else:
                wba_ax  = plt.subplot(gs[0,col], sharex=all_wba_ax[0],  sharey=all_wba_ax[0]) 
                stim_ax = plt.subplot(gs[1,col], sharex=all_stim_ax[0], sharey=all_stim_ax[0])    
    
            all_wba_ax[col] = wba_ax
            all_stim_ax[col] = stim_ax
    
            # loop single trials and plot all signals ________________________________
            for tr, i in zip(this_cnd_trs,range(n_cnd_trs)):

                this_color = scalarMap.to_rgba(i)     
     
                # plot WBA signal ____________________________________________________           
                wba_trace = all_fly_traces.loc[:,('this_fly',tr,cnd,'lmr')]
            
                baseline = np.nanmean(wba_trace[baseline_win])
                wba_trace = wba_trace - baseline  
                #wba_ax.plot(wba_trace+i/2.0,color=this_color)
               
                non_nan_i = np.where(~np.isnan(wba_trace))[0]       
                filtered_wba_trace = butter_lowpass_filter(wba_trace[non_nan_i],cutoff=filter_cutoff,fs=self.sampling_rate)
                
                # check if stim,trial combination is in this list. if so, plot in 
                # a thick line
                if (col,i) in saccade_stim_trs:
                    wba_ax.plot(filtered_wba_trace+i/tr_offset,color=this_color,linewidth=5)  
                    
                    # also indicate the saccade direction and time
                    # get saccade info
                    
                    saccade_i = [e_i for e_i, v in enumerate(trs_to_mark) if v[0] == col and v[1]==i][0]
                
                    saccade_t = trs_to_mark[saccade_i][2]
                    saccade_dir = trs_to_mark[saccade_i][3]
                    
                    if saccade_dir == 'R':
                        wba_ax.plot(saccade_t,filtered_wba_trace[saccade_t]+i/tr_offset,\
                            marker='^',markersize=10,linestyle='None',color=black)
                    elif saccade_dir == 'L':
                        wba_ax.plot(saccade_t,filtered_wba_trace[saccade_t]+i/tr_offset,\
                            marker='v',markersize=10,linestyle='None',color=black)
                else:
                    wba_ax.plot(filtered_wba_trace+i/tr_offset,color=this_color,linewidth=1)    
                    #wba_ax.plot(wba_trace+i/tr_offset,color=this_color,linewidth=1)    
                
                
                # now get potential saccade start times by differentiating the filtered
                # trace and then applying and threshold
                
                #if diff_thres: # skip this if diff_thres is set to 0
                    #candidate_saccade_is = find_candidate_saccades(filtered_wba_trace,diff_thres=diff_thres)
                    #wba_ax.plot(candidate_saccade_is,filtered_wba_trace[candidate_saccade_is]+i/tr_offset,\
                    #        marker='*',linestyle='None',color=grey)
                 
                wba_ax.text(0,i/tr_offset,str(i),
                    verticalalignment='bottom', horizontalalignment='right',
                    color=this_color, fontsize=8)
            
                #now plot stimulus traces ____________________________________________
                #stim_ax.plot(all_fly_traces.loc[::10,('this_fly',tr,cnd,'xstim')],color=this_color)
                stim_ax.plot(all_fly_traces.loc[:,('this_fly',tr,cnd,'xstim')],color=this_color)
                
                            
        # format all subplots _____________________________________________________  
   
        # loop though all columns again, format each row ______________________________
        for col in range(n_cols):      
            
            # show baseline window
            all_wba_ax[col].axvspan(baseline_win[0], baseline_win[-1], facecolor='grey', alpha=0.5)    
            
            # show 400 ms before turn window
            #all_wba_ax[col].axvspan(220, 620, facecolor='black', alpha=0.33)    
            
            # show turn window
            
            if long_static_spot:
                all_wba_ax[col].axvspan(1.630*sampling_rate, 1.730*sampling_rate, facecolor='black', alpha=0.5)       
            else:
                all_wba_ax[col].axvspan(.630*sampling_rate, .730*sampling_rate, facecolor='black', alpha=0.5)        
            
            # remove all time xticklabels __________________________________
            all_wba_ax[col].tick_params(labelbottom='off')
            
            # label columns
            all_wba_ax[col].set_title(self.stim_types_labels[cnds_to_plot[col]],fontsize=12)
            
            if col == 0:           
                
                if self.protocol == 'optical tracking':
                    all_wba_ax[col].set_ylabel('L-R WBA (degrees) + offset',fontsize=12)
                else:
                    all_wba_ax[col].set_ylabel('L-R WBA (V) + offset',fontsize=12)
                
                this_ylim = all_wba_ax[col].get_ylim()
                
                if self.protocol == 'optical tracking':
                    all_wba_ax[col].set_ylim([-12.5,this_ylim[1]*.85])
                else:
                    all_wba_ax[col].set_ylim([-.5,this_ylim[1]*.975])
                
                #all_wba_ax[col].set_ylim([-.5,(i+2)/tr_offset])
                
                #all_wba_ax[col].set_ylim([-.5,10.5])
                #all_wba_ax[col].set_ylim([-.5,16])
                
                #all_wba_ax[col].set_yticks([wba_lim[0],0,wba_lim[1]])
                
                all_stim_ax[col].tick_params(labelleft='off')
                all_stim_ax[col].tick_params(labelbottom='off')
            
                # label time x axis for just col 0 ______________________
                # divide by sampling rate _______________________________
                def div_sample_rate(x, pos): 
                    #The two args are the value and tick position 
                    #return (x-(s_iti/10))/(sampling_rate/10)
                    return x/(sampling_rate)
     
                formatter = FuncFormatter(div_sample_rate) 
                
                if long_static_spot:
                    all_wba_ax[col].set_xlim([0*sampling_rate, 2*sampling_rate]) #enforce max time
                else:
                    all_wba_ax[col].set_xlim([0*sampling_rate, 1*sampling_rate]) #enforce max time

                all_stim_ax[col].xaxis.set_major_formatter(formatter)
                all_stim_ax[col].tick_params(labelbottom='on')
                all_stim_ax[col].tick_params(labelleft='off')
                all_stim_ax[col].set_xlabel('Time (s)',fontsize=10)
            else:
                all_wba_ax[col].tick_params(labelleft='off')
                all_stim_ax[col].tick_params(labelleft='off')
                all_stim_ax[col].tick_params(labelbottom='off')
        
        figure_txt = title_txt
        
        fig.text(.2,.95,figure_txt,fontsize=18) 
       
        plt.draw()

        if if_save:
            saveas_path = '/Users/jamie/bin/figures/offset trials/'
            
            if trs_to_mark:
                plt.savefig(saveas_path + figure_txt + '_wba_by_cnd_yoffset_marked_saccades.png',\
                            bbox_inches='tight',dpi=200)     
    
            else:
                plt.savefig(saveas_path + figure_txt + '_wba_by_cnd_yoffset.png',\
                            bbox_inches='tight',dpi=200)     
        
        
        
                 
        
    def plot_flight_over_time(self,title_txt='',wba_lim=[-1.5,1.5],if_save=True): 
        # clean this up --
        # first store all points by vectorizing
        # change from plot -> get with boolean for plotting
        # make a separate function for plotting the population change over time
        #
        # this seems to work well, but I need to to show the windows of the saccades
        
        
        
        sampling_rate = 1000            # in hertz ********* move to fly info
        s_iti = .25 * sampling_rate      # ********* move to fly info
        
        baseline_win = range(0,int(s_iti)) 
        
        #get all traces and detect saccades ______________________________________________
        all_fly_traces, all_fly_saccades = self.get_traces_by_stim('this_fly',s_iti,get_saccades=False)

        fig = plt.figure(figsize=(9.5,11.5))       #(16.5, 9))
        
        cnds_to_plot = np.unique(self.stim_types) #[2,3,0,1]
        all_colors = [blue,magenta,green,black]
         
        for cnd,cnd_i in zip(cnds_to_plot,range(np.size(cnds_to_plot))):
            # now loop through the trials/cnd
            this_cnd_tr_ns = all_fly_traces.loc[:,('this_fly',slice(None),cnd,'lmr')].columns.get_level_values(1).tolist()
            
            this_color = all_colors[cnd_i]
            
            for tr_n, i in zip(this_cnd_tr_ns,range(np.size(this_cnd_tr_ns))):
                wba_trace = all_fly_traces.loc[:,('this_fly',tr_n,slice(None),'lmr')]
                exp_tr_n = all_fly_traces.loc[:,('this_fly',tr_n,slice(None),'lmr')]
                
                baseline = np.nanmean(wba_trace.loc[baseline_win,:])
                turn_win = np.nanmean(wba_trace.loc[range(450,550),:]) - baseline  
         
                plt.plot(tr_n,turn_win,'.',markersize=12,color=this_color)
                plt.axhline(linewidth=.5, color=black)
                
        plt.xlabel('Trial number')
        plt.ylabel('L-R WBA in turn window')  
        plt.title(title_txt,fontsize=18)  
        plt.ylim([-1.5,1.5])
        
        for cnd,i in zip(cnds_to_plot,range(np.size(cnds_to_plot))):
            fig.text(.5,.85-.03*i,self.stim_types_labels[cnd],color=all_colors[i],fontsize=14) 
        
        if if_save:
            saveas_path = '/Users/jamie/bin/figures/'
            plt.savefig(saveas_path + title_txt + '_turn_adaptation.png',\
                                    bbox_inches='tight',dpi=100) 
                                    
                                    
    def get_flight_over_time(self,title_txt='',wba_lim=[-1.5,1.5],if_save=True): 
           
        sampling_rate = 1000            # in hertz ********* move to fly info
        s_iti = .25 * sampling_rate      # ********* move to fly info
        
        baseline_win = range(0,int(s_iti)) 
        
        #get all traces and detect saccades ______________________________________________
        all_fly_traces, all_fly_saccades = self.get_traces_by_stim('this_fly',s_iti,get_saccades=False)

        wba_trace = all_fly_traces.loc[:,('this_fly',slice(None),slice(None),'lmr')]
        
        print np.shape(wba_trace)
        baseline = np.nanmean(wba_trace.loc[range(0,250),:],0)
        turn_win_mean = np.nanmean(wba_trace.loc[range(475,650),:],0) - baseline  
        print np.shape(turn_win_mean)

        #get corresponding tr#, cnd
        tr_cnds = all_fly_traces.loc[:,('this_fly',slice(None),slice(None),'lmr')].columns.get_level_values(2)
     
        return turn_win_mean, tr_cnds
        
    def get_traces_by_stim(self,fly_name='this_fly',iti=.25*1000,get_saccades=False):
    # here extract the traces for each of the stimulus times. 
    # align to looming start, and add the first pre stim and post stim intervals
    # here return a data frame of lwa and rwa wing traces
    # self.stim_types already holds an np.array vector of the trial type indicies
   
    # using a pandas data frame with multilevel indexing! rows = time in ms
    # columns are multileveled -- genotype, fly, trial index, trial type, trace
        
        fly_df = pd.DataFrame()
        fly_saccades_df = pd.DataFrame() #keep empty if not tracking all saccades
       
        for tr in range(self.n_trs):
            this_loom_start = self.tr_starts[tr]
            this_start = this_loom_start - iti   # previously -- 2*iti. I'm not sure why.  
            this_stop = self.tr_stops[tr] + iti  #hack ----------
            
            this_stim_type = self.stim_types[tr]
            iterables = [[fly_name],
                         [tr],
                         [this_stim_type],
                         ['ao','lmr','lwa','rwa','xstim']]
            column_labels = pd.MultiIndex.from_product(iterables,names=['fly','tr_i','tr_type','trace']) 
                                                            #is the unsorted tr_type level a problem?    
                   
            tr_traces = np.asarray([self.ao[this_start:this_stop],
                                         self.lmr[this_start:this_stop],
                                         self.lwa[this_start:this_stop],
                                         self.rwa[this_start:this_stop],
                                         self.xstim[this_start:this_stop]]).transpose()  #reshape to avoid transposing
                                          
            tr_df = pd.DataFrame(tr_traces,columns=column_labels) #,index=time_points) 
            fly_df = pd.concat([fly_df,tr_df],axis=1)
            
            
            if get_saccades:
                # make a data structure of saccade times in the same format as the 
                # fly_df trace information
                # data = saccade start times. now not trying to define saccade stops
                # rows = saccade number
                # columns = fly, trial index, trial type
                
                 iterables = [[fly_name],
                             [tr],
                             [this_stim_type]]
                 column_labels = pd.MultiIndex.from_product(iterables,names=['fly','tr_i','tr_type']) 
                                                             
                 saccade_starts = find_saccades(self.lmr[this_start:this_stop])
                 tr_saccade_starts_df = pd.DataFrame(np.transpose(saccade_starts),columns=column_labels)            
                 fly_saccades_df = pd.concat([fly_saccades_df,tr_saccade_starts_df],axis=1)
            
        return fly_df, fly_saccades_df 
        
    
       
        
#---------------------------------------------------------------------------#
def moving_average(values, window):
    #next add gaussian, kernals, etc
    #pads on either end to return an equal length structure,
    #although the edges are distorted
    
    if (window % 2): #is odd 
        window = window + 1; 
    halfwin = window/2
    
    n_values = np.size(values)
    
    padded_values = np.ones(n_values+window)*np.nan
    padded_values[0:halfwin] = np.ones(halfwin)*np.mean(values[0:halfwin])
    padded_values[halfwin:halfwin+n_values] = values
    padded_values[halfwin+n_values:window+n_values+1] = np.ones(halfwin)*np.mean(values[-halfwin:n_values])
  
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(padded_values, weights, 'valid')
    return sma[0:n_values]
    
def xcorr(a, v):
    a = (a - np.mean(a)) / (np.std(a) * (len(a)-1))
    v = (v - np.mean(v)) /  np.std(v)
    xc = np.correlate(a, v, mode='same')
    return xc
    
def read_abf(abf_filename):
        fh = AxonIO(filename=abf_filename)
        segments = fh.read_block().segments
    
        if len(segments) > 1:
            print 'More than one segment in file.'
            return 0

        analog_signals_ls = segments[0].analogsignals
        analog_signals_dict = {}
        for analog_signal in analog_signals_ls:
            analog_signals_dict[analog_signal.name.lower()] = analog_signal

        return analog_signals_dict
        
def process_wings(raw_wings):
    #here shift wing signal -12 ms in time, filling end with nans
    shifted_wings = np.zeros_like(raw_wings)
    shifted_wings[0:-12] = raw_wings[12:]
    shifted_wings[-11:] = raw_wings[-1]   
    
    #now multiply to convert volts to degrees
    processed_wings = -45 + shifted_wings*33.75
    return processed_wings
        
def find_saccades(raw_lmr_trace,test_plot=False):
# this function hasn't been tuned to the current signal.
# use this more as an algorithm outline

    #first fill in nans with nearest signal
    lmr_trace = raw_lmr_trace[~np.isnan(raw_lmr_trace)] 
        #this may give different indexing than input
        #ideally fill in nans in wing processing

    # filter lmr signal
    filtered_trace = butter_lowpass_filter(lmr_trace) #6 hertz
    
    # differentiate, take the absolute value
    diff_trace = abs(np.diff(filtered_trace))
     
    # mark saccade start times -- this could be improved
    diff_thres = .01
    cross_d_thres = np.where(diff_trace > diff_thres)[0]
    
    # #use this to find saccade stops
#     saccade_start_candidate = diff_trace[1:-1] < diff_thres  
#     saccade_cont  = diff_trace[2:]   >= diff_thres
#     stacked_start_cont = np.vstack([saccade_start,saccade_cont])
#     candidate_saccade_starts = np.where(np.all(stacked_start_cont,axis=0))[0]
    
    # impose a refractory period for saccades
    d_cross_d_thres = np.diff(cross_d_thres)
    
    refractory_period = .2 * 10000
    if cross_d_thres.size:
        saccade_starts = [cross_d_thres[0]] #include first
        
        #then take those with gaps between saccade events
        other_is = np.where(d_cross_d_thres > refractory_period)[0]+1
        saccade_starts = np.hstack((saccade_starts,cross_d_thres[other_is]))
    else:
        saccade_starts = []
       
    if test_plot:
        fig = plt.figure()
        plt.plot(lmr_trace,'grey')
        plt.plot(filtered_trace,'black')
        plt.plot(1000*diff_trace,'green')
        
        
        plt.plot(cross_d_thres,np.zeros_like(cross_d_thres),'r.')
        plt.plot(saccade_starts,np.ones_like(saccade_starts),'mo')
    
    # return indicies of start and stop times + saccade magnitude 
    return saccade_starts
       
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sp.signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff=12, fs=1000, order=5): #how does the order change?
    b, a = butter_lowpass(cutoff, fs, order)
    #y = sp.signal.lfilter(b, a, data) #what's the difference here? 
    y = sp.signal.filtfilt(b, a, data)
    return y
              
def get_saccade_and_control_traces(saccades_dict,filter_fs=1000):
    # input -- dictionary in which the keys are the fly filenames
    # and the values are arrays of tuples of the stimulus condition, trial
    # within that condition/block, saccade start time, and saccade direction
    # example -- saccades_dict['03_30_0000']=[(0,0,500,'L'),(0,1,325,'R')] 
    # filter_fs -- assumes the same sampling rate for all flies in dictionary.
    #              1,000 for wing beat analyzer, 10,000 for strokelitude
    #
    # output -- two pandas data structures
    # saccade_info: row=saccade id, columns = fly, cnd, tr, start time, direction
    # saccade_wba_traces: row = times, columns are multilevel -- saccadeid 
    #                                                           pre, saccade, post traces, sacc)_stim
    # traces are filtered and baseline subtracted
    #
    
    path_name = '/Users/jamie/Dropbox/maimon lab - behavioral data/'
    
    n_saccades = len(sum(saccades_dict.values(),[])) # count the number of saccades
    max_t = 1000

    info_rows = ['fly','cnd','block','start_t','dir']
    saccade_info = pd.DataFrame(index=info_rows,columns=range(n_saccades))
    
    iterables = [range(5),
                ['post','prev','sacc','stim']]
    wba_columns = pd.MultiIndex.from_product(iterables,names=['saccade_id','trace']) 
    saccade_and_control_traces = pd.DataFrame(index=range(max_t),columns=wba_columns)
    
    
    baseline_window = range(0,150)
    saccade_i = 0
    filter_cutoff = 48
    
    print 'n saccades = ' + str(n_saccades)
    
    for f_name in saccades_dict: 
    
        #print f_name
        fly = Spot_Saccadic_Supression(path_name + '2015_'+ f_name)
        fly.process_fly(False)
        stim_types = fly.unique_stim_types
         
        all_traces, saccades = fly.get_traces_by_stim()
       
        # discard the traces earlier in time.
        # this value is somewhat arbitrary, but saccade times are relative it.
        s_iti = .25 * fly.sampling_rate    
        new_start_t = int(3*s_iti) 
        all_traces = all_traces.ix[new_start_t:,slice(None)]
        new_index = np.array(all_traces.index) - new_start_t
        all_traces.index = new_index
    
    
    
        this_fly_saccades = saccades_dict[f_name]
        
        for one_saccade_info in this_fly_saccades: 
            
            stim_i = one_saccade_info[0]
            stim_tr_i = one_saccade_info[1]
            #print stim_i, stim_tr_i
            
            this_stim_traces = all_traces.loc[:,('this_fly',slice(None),stim_types[stim_i],'lmr')]
            this_saccade_trace = this_stim_traces.iloc[0:max_t,(stim_tr_i)]
            
            this_stim_pattern_traces = all_traces.loc[:,('this_fly',slice(None),stim_types[stim_i],'xstim')]
            this_stim_trace = this_stim_pattern_traces.iloc[0:max_t,(stim_tr_i)]
        
            if stim_tr_i == 0:
                this_prev_trace = this_stim_traces.iloc[0:max_t,(1)] #take trace following
            else:
                this_prev_trace = this_stim_traces.iloc[0:max_t,(stim_tr_i-1)]
            
            # I'm assuming that I don't take saccades from the last trace. 
            # later, add an error catch here
            try:
                this_post_trace = this_stim_traces.iloc[0:max_t,(stim_tr_i+1)] 
            except:
                this_post_trace = this_stim_traces.iloc[0:max_t,(stim_tr_i-1)] 
                
            filtered_saccade_trace = butter_lowpass_filter(this_saccade_trace,cutoff=filter_cutoff,fs=filter_fs) 
            processed_saccade_trace = filtered_saccade_trace-\
                                      np.nanmean(filtered_saccade_trace[baseline_window])
        
            filtered_prev_trace = butter_lowpass_filter(this_prev_trace,cutoff=filter_cutoff,fs=filter_fs)  
            processed_prev_trace = filtered_prev_trace-\
                                      np.nanmean(filtered_prev_trace[baseline_window]) 
        
            filtered_post_trace = butter_lowpass_filter(this_post_trace,cutoff=filter_cutoff,fs=filter_fs)  
            processed_post_trace = filtered_post_trace-\
                                      np.nanmean(filtered_post_trace[baseline_window])
        
            # ['post','prev','sacc','stim']] -- keep this lexsorted for multindexing
            saccade_and_control_traces.ix[:,(saccade_i,'post')] = processed_post_trace
            saccade_and_control_traces.ix[:,(saccade_i,'prev')] = processed_prev_trace
            saccade_and_control_traces.ix[:,(saccade_i,'sacc')] = processed_saccade_trace
            saccade_and_control_traces.ix[:,(saccade_i,'stim')] = this_stim_trace
            
            saccade_info.ix['fly',saccade_i] = f_name
            saccade_info.ix['cnd',saccade_i] = stim_i
            saccade_info.ix['block',saccade_i] = stim_tr_i
            saccade_info.ix['start_t',saccade_i] = one_saccade_info[2]
            saccade_info.ix['dir',saccade_i] = one_saccade_info[3]
            
            saccade_i = saccade_i + 1    
    
    return saccade_info, saccade_and_control_traces
                         
def plot_spot_saccade_versus_spont_saccade_time():  
    # specify all input arguments
    # now plot the saccade time v. 
    # mean response in a saccade window - mean in window just before
    # separately for left and right  spots
    # next consider direction of saccade x direction of spot
    # need to consider absolute displacement? 
    # now with approximate saccade onset times
    # replot the saccade traces v. controls 
    # within the same stimulus types
    # this function is still a work in progress
   
        fig = plt.figure()
        saccade_baseline_window = range(585,620)
        saccade_window = range(665,700)

        plt.ylim([-.5,.5])
        plt.axhline()
        plt.axvline(x=0)

        for i,s,this_spont_sacc_t in zip(range(n_stim),\
                                         stim_subset,spont_sacc_ts): #range(n_saccades):
            this_color = scalarMap.to_rgba(i)

            this_saccade_mean = np.mean(all_saccade_traces.ix[saccade_window,s])
            this_control_mean = np.mean(all_control_traces.ix[saccade_window,s])
    
            this_saccade_pre_mean = np.mean(all_saccade_traces.ix[saccade_baseline_window,s])
            this_control_pre_mean = np.mean(all_control_traces.ix[saccade_baseline_window,s])
    
            sacc_diff = this_saccade_mean#-this_saccade_pre_mean
        
            plt.plot(this_spont_sacc_t-500,sacc_diff,'.',markersize=10,color=magenta)
    
        plt.ylim([-.5,.5])    
        plt.xlabel('Approximate time from saccade start (ms)')
        plt.ylabel('L-R WBA in saccade window - pre saccade(V)')
        plt.title(title+' spot')
        saveas_path = '/Users/jamie/bin/figures/'
        plt.savefig(saveas_path+' spot saccade v. spont sacc time.png',\
                    bbox_inches='tight',dpi=100)     
                 
def plot_saccade_traces_pre_pos_y_offset(saccade_info, saccade_and_control_traces): 
    # for each condition, show saccades to the left and to the right
    # + their previous and post traces in finer lines
    #
    # both with a y-offset
    # and on the bottom with no y-offset and an average
    #
    # April 30 2015 -- rewriting to show the flash responses
    # also store the ao signal here
    #
    
    stim_cnds = range(4)
    stim_titles = ['Spot on right, front->back','Spot on right, back->front',\
                    'Spot on left, back->front','Spot on left, front->back']
    
    spont_sacc_dir = ['L','R']    
    direction_titles =['left','right']
    
    overlaid_offset = .75

    for stim in stim_cnds: #3,]: #
        
        # make a figure -- three rows x two columns ______________________________________
        # rows -- y-offset wba traces, stim, overlays wba traces + mean
        # columns -- left and right spontaneous turns
        
        n_cols = 2 
        n_rows = 3
        
        gs = gridspec.GridSpec(n_rows,n_cols,height_ratios=[1,.05,1])
        fig = plt.figure(figsize=(14.5, 14.5))
        gs.update(wspace=0.1, hspace=0.025) # set the spacing between axes. 
        
        # store all subplots for formatting later           
        all_offset_wba_ax = np.empty(n_cols,dtype=plt.Axes)
        all_stim_ax = np.empty(n_cols,dtype=plt.Axes)
        all_overlaid_wba_ax = np.empty(n_cols,dtype=plt.Axes)
        
        # find all saccades of the stimulus type ______________________________________
        this_stim_trs = np.where(saccade_info.ix['cnd',:] == stim)[0]
        
        for dir in spont_sacc_dir:
        
            # create subplots ________________________________________________________              
            if dir == 'L':
                offset_wba_ax  = plt.subplot(gs[0,0]) 
                stim_ax = plt.subplot(gs[1,0])
                overlaid_wba_ax = plt.subplot(gs[2,0])   
                
                # turn off axis sharing for formatting
                #offset_wba_ax  = plt.subplot(gs[0,0]) 
                #stim_ax = plt.subplot(gs[1,0],sharex=offset_wba_ax)
                #overlaid_wba_ax = plt.subplot(gs[2,0],sharex=offset_wba_ax) 
                
                all_offset_wba_ax[0] = offset_wba_ax
                all_stim_ax[0] = stim_ax
                all_overlaid_wba_ax[0] = overlaid_wba_ax
                offset_wba_ax.set_title(stim_titles[stim]+', left spont saccade')
                   
            else:
                offset_wba_ax  = plt.subplot(gs[0,1]) 
                stim_ax = plt.subplot(gs[1,1])    
                overlaid_wba_ax = plt.subplot(gs[2,1])    
    
                # turn off axis sharing for formatting
                #offset_wba_ax  = plt.subplot(gs[0,1], sharex=all_offset_wba_ax[0],  sharey=all_offset_wba_ax[0]) 
                #stim_ax = plt.subplot(gs[1,1], sharex=all_offset_wba_ax[0], sharey=all_stim_ax[0])    
                #overlaid_wba_ax = plt.subplot(gs[2,1], sharex=all_offset_wba_ax[0], sharey=all_overlaid_wba_ax[0])    
    
                all_offset_wba_ax[1] = offset_wba_ax
                all_stim_ax[1] = stim_ax
                all_overlaid_wba_ax[1] = overlaid_wba_ax 
                offset_wba_ax.set_title(stim_titles[stim]+', right spont saccade')       
        
            # show turn window
            offset_wba_ax.axvspan(640, 740, facecolor='black', alpha=0.25)
            stim_ax.axvspan(640, 740, facecolor='black', alpha=0.25)
            overlaid_wba_ax.axvspan(640, 740, facecolor='black', alpha=0.25)

            
            # find all saccades of this direction type
            this_dir_trs = np.where(saccade_info.ix['dir',:] == dir)[0]
        
            # intersect with the same stimulus type
            this_stim_dir_trs = np.intersect1d(this_stim_trs,this_dir_trs)

            # get the number of saccades here, make a color map
            n_this_saccades = len(this_stim_dir_trs)    
            
            # get a colormap
            n_colors = n_this_saccades
            cmap = plt.cm.get_cmap('spectral')
            cNorm = colors.Normalize(0,n_colors)
            scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cmap)

            # sort saccades by onset time -- can do this later
            #spont_saccade_times.sort()
            #right_stim_sacc_t = spont_saccade_times[right_stim].order().values
    
            # loop through all saccades
            for i in range(n_this_saccades): 
                
                # map the saccade index to the saccade id 
                s = this_stim_dir_trs[i]
                this_color = scalarMap.to_rgba(i)

                this_saccade_trace = saccade_and_control_traces.ix[:,(s,'sacc')]
                this_stim_trace = saccade_and_control_traces.ix[:,(s,'stim')]
                this_prev_trace = saccade_and_control_traces.ix[:,(s,'prev')]
                this_post_trace = saccade_and_control_traces.ix[:,(s,'post')]

                # plot saccade traces with their previous traces 
                offset_wba_ax.plot(this_saccade_trace+i/2.0,color=this_color,linewidth=3)
                offset_wba_ax.plot(this_prev_trace+i/2.0,color=this_color,linewidth=1)
                
                # plot the stimulus traces
                stim_ax.plot(this_stim_trace,color=this_color)         
            
                # now show overlaid traces     
                
                overlaid_wba_ax.plot(this_prev_trace-overlaid_offset,color=blue,linewidth=.75)
                overlaid_wba_ax.plot(this_saccade_trace,color=magenta,linewidth=.75)
                overlaid_wba_ax.plot(this_post_trace+overlaid_offset,color=black,linewidth=.75)
                     
                # give pre and post-offset
            
            # now plot the means here
            this_dir_trs_all_traces = saccade_and_control_traces.loc[:,this_stim_dir_trs.tolist()]
            
            prev_traces = this_dir_trs_all_traces.loc[:,(slice(None),'prev')].values
            prev_means = np.nanmean(prev_traces,axis=1)
            overlaid_wba_ax.plot(prev_means-overlaid_offset,color=blue,linewidth=3)
            
            saccade_traces = this_dir_trs_all_traces.loc[:,(slice(None),'sacc')].values
            saccade_means = np.nanmean(saccade_traces,axis=1)
            overlaid_wba_ax.plot(saccade_means,color=magenta,linewidth=3)

            post_traces = this_dir_trs_all_traces.loc[:,(slice(None),'post')].values
            post_means = np.nanmean(post_traces,axis=1)
            overlaid_wba_ax.plot(post_means+overlaid_offset,color=black,linewidth=3)
            

        #all_offset_wba_ax[0].set_ylim([-1,6])
        all_stim_ax[0].set_ylim([0,10])
        #all_overlaid_wba_ax[0].set_ylim([-1.5,1.5])
        
        #all_offset_wba_ax[1].set_ylim([-1,17])
        all_stim_ax[0].set_ylim([0,10])
        all_stim_ax[1].set_ylim([0,10])
        all_overlaid_wba_ax[0].set_ylim([-1.25,1.25])
        all_overlaid_wba_ax[1].set_ylim([-1.25,1.25])             
             
             
        all_offset_wba_ax[0].set_ylabel('L-R WBA (V)+tr offset')
        all_overlaid_wba_ax[0].set_ylabel('L-R WBA (V)+tr offset')
        all_overlaid_wba_ax[0].set_xlabel('Time (ms)')

        # remove extra xtick labels
        all_offset_wba_ax[0].xaxis.set_ticklabels([])
        all_offset_wba_ax[1].xaxis.set_ticklabels([])

        all_stim_ax[0].xaxis.set_ticklabels([])
        all_stim_ax[0].yaxis.set_ticklabels([])
        all_stim_ax[1].xaxis.set_ticklabels([])
        all_stim_ax[1].yaxis.set_ticklabels([])

        #all_offset_wba_ax[1].yaxis.set_ticklabels([])
        all_overlaid_wba_ax[1].yaxis.set_ticklabels([])
        all_overlaid_wba_ax[1].xaxis.set_ticklabels([])

        saveas_path = '/Users/jamie/bin/figures/'
        plt.savefig(saveas_path+ str(stim) + dir +' spot saccade.png',\
                    bbox_inches='tight',dpi=100)
        
def plot_saccade_traces_pre_pos_y_offset_time_sorted(saccade_info, saccade_and_control_traces,
                                min_pre_saccade_t,max_pre_saccade_t,plot_by_stim=False): 
    # for each condition, show saccades to the left and to the right
    # + their previous and post traces in finer lines
    #
    # both with a y-offset
    # and on the bottom with no y-offset and an average
    #
    # modify this to allow sorting by spot position (two)
    # or by stimulus (four)
    
    stim_cnds = range(4)
    stim_titles = ['Spot on right, front->back','Spot on right, back->front',\
                   'Spot on left, back->front','Spot on left, front->back']
    
    spont_sacc_dir = ['R','L']    
    direction_titles =['right','left']
    
    overlaid_offset = 0

    # make a figure -- three rows x two columns ______________________________________
    # rows -- y-offset wba traces, stim, overlays wba traces + mean
    # columns -- left and right spontaneous turns
    
    if plot_by_stim:
        n_cols = 4
    else:    
        n_cols = 2
     
    n_rows = 3
        
    gs = gridspec.GridSpec(n_rows,n_cols,height_ratios=[1,.05,.66])
    fig = plt.figure(figsize=(14.5, 14.5))
    gs.update(wspace=0.1, hspace=0.025) # set the spacing between axes. 
     
    # store all subplots for formatting later           
    all_offset_wba_ax = np.empty(n_cols,dtype=plt.Axes)
    all_stim_ax = np.empty(n_cols,dtype=plt.Axes)
    all_overlaid_wba_ax = np.empty(n_cols,dtype=plt.Axes) 
    
    # specify conditions for looping,
    # add a clean way to format the left most column
    if plot_by_stim:
        col_iter = stim_cnds 
    else:
        col_iter = direction_titles
    first_col = col_iter[0]    
     
    for col in col_iter: 
        
        # find all saccades of the stimulus type ______________________________________
        
        if plot_by_stim:
            this_stim_trs = np.where(saccade_info.ix['cnd',:] == col)[0]
        else:
            if col == 'left':
                this_stim_dirs = [2,3]
            else:
                this_stim_dirs = [0,1]
        
            stim1_trs = np.where(saccade_info.ix['cnd',:] == this_stim_dirs[0])[0]
            stim2_trs = np.where(saccade_info.ix['cnd',:] == this_stim_dirs[1])[0]
        
            this_stim_trs = np.hstack([stim1_trs,stim2_trs])
            
        # create subplots ________________________________________________________              
        if col == first_col:
            offset_wba_ax  = plt.subplot(gs[0,0]) 
            stim_ax = plt.subplot(gs[1,0],sharex=offset_wba_ax)
            overlaid_wba_ax = plt.subplot(gs[2,0],sharex=offset_wba_ax)   
        
            all_offset_wba_ax[0] = offset_wba_ax
            all_stim_ax[0] = stim_ax
            all_overlaid_wba_ax[0] = overlaid_wba_ax  

        else: 
            if plot_by_stim:
                c_i = col 
            else:
                c_i = 1
                
            offset_wba_ax  = plt.subplot(gs[0,c_i], sharex=all_offset_wba_ax[0], sharey=all_offset_wba_ax[0]) 
            stim_ax = plt.subplot(gs[1,c_i], sharex=all_offset_wba_ax[0], sharey=all_stim_ax[0])    
            overlaid_wba_ax = plt.subplot(gs[2,c_i], sharex=all_offset_wba_ax[0], sharey=all_overlaid_wba_ax[0])    

            all_offset_wba_ax[c_i] = offset_wba_ax
            all_stim_ax[c_i] = stim_ax
            all_overlaid_wba_ax[c_i] = overlaid_wba_ax  
        
        if plot_by_stim:
            offset_wba_ax.set_title(stim_titles[col])
        else:
            offset_wba_ax.set_title('Spot on the ' + col)   

        # show turn window
        offset_wba_ax.axvspan(640, 740, facecolor='black', alpha=0.25)
        stim_ax.axvspan(640, 740, facecolor='black', alpha=0.25)
        overlaid_wba_ax.axvspan(640, 740, facecolor='black', alpha=0.25)

        # intersect direction with the same stimulus type
        this_stim_dir_trs = this_stim_trs  # just combine all directions of saccades for now
        
        #  get the saccade start times for this stimulus direction
        this_stim_saccade_start_ts = saccade_info.ix['start_t',this_stim_dir_trs]
        
        # now just select saccades that start in this time range
        spot_move_t = 575 # approximate -- if I care about ms precision, calculate this for each trial
        this_min_t  = min_pre_saccade_t + spot_move_t
        this_max_t = max_pre_saccade_t + spot_move_t
        
        offset_wba_ax.axvspan(this_min_t, this_max_t, facecolor='black', alpha=0.125)
        
        saccades_in_t1 = np.where(this_stim_saccade_start_ts > this_min_t)[0]
        saccades_in_t2 = np.where(this_stim_saccade_start_ts < this_max_t)[0]
        saccade_in_t_win_i = np.intersect1d(saccades_in_t1,saccades_in_t2)
        this_stim_t_saccades = this_stim_dir_trs[saccade_in_t_win_i]
        
        # get the number of saccades here, make a color map
        n_this_saccades = len(this_stim_t_saccades)    
    
        # get a colormap
        n_colors = n_this_saccades
        cmap = plt.cm.get_cmap('spectral')
        cNorm = colors.Normalize(0,n_colors)
        scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cmap)

        # sort saccades by onset time -- can do this later
        #spont_saccade_times.sort()
        #right_stim_sacc_t = spont_saccade_times[right_stim].order().values

        # loop through all saccades
        for i in range(n_this_saccades): 
        
            # map the saccade index to the saccade id 
            s = this_stim_t_saccades[i]
            this_color = scalarMap.to_rgba(i)

            this_saccade_trace = saccade_and_control_traces.ix[:,(s,'sacc')]
            this_stim_trace = saccade_and_control_traces.ix[:,(s,'stim')]
            this_prev_trace = saccade_and_control_traces.ix[:,(s,'prev')]
            this_post_trace = saccade_and_control_traces.ix[:,(s,'post')]

            # plot saccade traces with their previous traces 
            offset_wba_ax.plot(this_saccade_trace+i/2.0,color=this_color,linewidth=3)
            offset_wba_ax.plot(this_prev_trace+i/2.0,color=this_color,linewidth=1)
        
            # plot the stimulus traces
            stim_ax.plot(this_stim_trace,color=this_color)         
    
            # now show overlaid traces     
            
            #sub_window = range(520,570)
            
            this_prev_trace = this_prev_trace #- np.nanmean(this_prev_trace[sub_window])
            this_saccade_trace = this_saccade_trace #- np.nanmean(this_saccade_trace[sub_window])
            this_post_trace = this_post_trace #- np.nanmean(this_post_trace[sub_window])
            
            
            overlaid_wba_ax.plot(this_prev_trace-overlaid_offset,color=blue,linewidth=.5)
            overlaid_wba_ax.plot(this_saccade_trace,color=magenta,linewidth=.5)
            overlaid_wba_ax.plot(this_post_trace+overlaid_offset,color=black,linewidth=.5)
             
            # give pre and post-offset
    
        # now plot the means here
        this_dir_trs_all_traces = saccade_and_control_traces.loc[:,this_stim_t_saccades.tolist()]
        t = range(1000)
    
        # prev traces ____________________________________
        prev_traces = this_dir_trs_all_traces.loc[:,(slice(None),'prev')].values
        prev_means = np.nanmean(prev_traces,axis=1) # - np.nanmean(prev_traces[sub_window])
        prev_ste = np.nanstd(prev_traces,axis=1)/math.sqrt(n_this_saccades)
        overlaid_wba_ax.plot(prev_means-overlaid_offset,color=blue,linewidth=3)
    
        # get,plot error bars
        lower_bound = prev_means - prev_ste
        upper_bound = prev_means + prev_ste
        overlaid_wba_ax.fill_between(t, lower_bound, upper_bound, facecolor=blue, alpha=0.25)
        
        # post traces ____________________________________
        post_traces = this_dir_trs_all_traces.loc[:,(slice(None),'post')].values
        post_means = np.nanmean(post_traces,axis=1) # - np.nanmean(post_traces[sub_window])
        post_ste = np.nanstd(post_traces,axis=1)/math.sqrt(n_this_saccades)
        overlaid_wba_ax.plot(post_means+overlaid_offset,color=black,linewidth=3)
    
        # get,plot error bars
        lower_bound = post_means - post_ste
        upper_bound = post_means + post_ste
        overlaid_wba_ax.fill_between(t, lower_bound, upper_bound, facecolor=black, alpha=0.25)

         # saccades traces ____________________________________
        saccade_traces = this_dir_trs_all_traces.loc[:,(slice(None),'sacc')].values
        saccade_means = np.nanmean(saccade_traces,axis=1) # - np.nanmean(saccade_traces[sub_window])
        saccade_ste = np.nanstd(saccade_traces,axis=1)/math.sqrt(n_this_saccades)
        overlaid_wba_ax.plot(saccade_means,color=magenta,linewidth=3)
        
        # get,plot error bars
        lower_bound = saccade_means - saccade_ste
        upper_bound = saccade_means + saccade_ste
        overlaid_wba_ax.fill_between(t, lower_bound, upper_bound, facecolor=magenta, alpha=0.25)
        

    title_text = 'Spontaneous saccades '+str(-1*min_pre_saccade_t) + ' to ' + \
                        str(-1*max_pre_saccade_t) + ' ms before spot movement'
                        
                        
    # now clean up axes ____________________________________________
    
    # first column needs full labels ________        
    all_stim_ax[0].set_ylim([0,10])
    all_overlaid_wba_ax[0].set_ylim([-.7,.7])
    all_overlaid_wba_ax[0].locator_params(tight=True,nbins=5,axis='y') #set n bins to 3
             
    all_offset_wba_ax[0].set_ylabel('L-R WBA (V)+tr offset')
    all_overlaid_wba_ax[0].set_ylabel('L-R WBA (V)')
    all_overlaid_wba_ax[0].set_xlabel('Time (ms)')

    # remove extra xtick labels
    for col in range(n_cols):
        plt.setp(all_offset_wba_ax[col].get_xticklabels(), visible=False)
        plt.setp(all_stim_ax[col].get_xticklabels(), visible=False)
        plt.setp(all_stim_ax[col].get_yticklabels(), visible=False)
        
        #now annotate stimulus positions, title ______________________________________      
        fig.text(.04,.34,'previous tr',color=blue,fontsize=10)
        fig.text(.04,.32,'saccade tr',color=magenta,fontsize=10)                    
        fig.text(.04,.3,'post tr',color=black,fontsize=10)
    
        if col > 0:
            plt.setp(all_offset_wba_ax[col].get_yticklabels(), visible=False)
            plt.setp(all_overlaid_wba_ax[col].get_yticklabels(), visible=False)
            plt.setp(all_overlaid_wba_ax[col].get_xticklabels(), visible=False)
    
    fig.text(.05,.95,title_text,fontsize=14)
    
    saveas_path = '/Users/jamie/bin/figures/'
    plt.savefig(saveas_path+title_text+' population.png',\
                bbox_inches='tight',dpi=100)


def plot_spot_saccade_amplitude_three_trs(saccade_info, saccade_and_control_traces,
                                min_pre_saccade_t,max_pre_saccade_t,plot_by_stim=False): 
    # for each condition, the amplitude of the spot-evoked saccade
    # for the trials with spont saccades
    # + their previous and post traces in finer lines
    #
    # parameter plot_by_stim allows plotting by spot position (two cnd)
    # or by stimulus (four cnds)
    
    stim_cnds = range(4)
    stim_titles = ['Spot on right, front->back','Spot on right, back->front',\
                   'Spot on left, back->front','Spot on left, front->back']
    
    spont_sacc_dir = ['R','L']    
    direction_titles =['right','left']
    
    # make a figure -- one row x two or four columns ______________________________________
    # columns -- stim type or spot position
    
    if plot_by_stim:
        n_cols = 4
    else:    
        n_cols = 2
     
    n_rows = 1
        
    gs = gridspec.GridSpec(n_rows,n_cols)
    fig = plt.figure(figsize=(14.5, 14.5))
    gs.update(wspace=0.1, hspace=0.025) # set the spacing between axes. 
     
    # store all subplots for formatting later           
    all_sacc_amp_ax = np.empty(n_cols,dtype=plt.Axes)
    
    # specify conditions for looping,
    # add a clean way to format the left most column
    if plot_by_stim:
        col_iter = stim_cnds 
    else:
        col_iter = direction_titles
    first_col = col_iter[0]    
     
     
    saccade_amplitudes = get_all_saccade_amplitudes(saccade_info,saccade_and_control_traces)
     
    for col in col_iter: 
        
        # find all saccades of the stimulus type ______________________________________
        
        if plot_by_stim:
            this_stim_trs = np.where(saccade_info.ix['cnd',:] == col)[0]
        else:
            if col == 'left':
                this_stim_dirs = [2,3]
            else:
                this_stim_dirs = [0,1]
        
            stim1_trs = np.where(saccade_info.ix['cnd',:] == this_stim_dirs[0])[0]
            stim2_trs = np.where(saccade_info.ix['cnd',:] == this_stim_dirs[1])[0]
        
            this_stim_trs = np.hstack([stim1_trs,stim2_trs])
            
        # create subplots ________________________________________________________              
        if col == first_col:
            sacc_amp_ax  = plt.subplot(gs[0,0]) 
            all_sacc_amp_ax[0] = sacc_amp_ax
            
        else: 
            if plot_by_stim:
                c_i = col 
            else:
                c_i = 1
            sacc_amp_ax  = plt.subplot(gs[0,c_i], sharex=all_sacc_amp_ax[0], sharey=all_sacc_amp_ax[0]) 
            all_sacc_amp_ax[c_i] = sacc_amp_ax
            
        if plot_by_stim:
            sacc_amp_ax.set_title(stim_titles[col])
        else:
            sacc_amp_ax.set_title('Spot on the ' + col)   

        # intersect direction with the same stimulus type
        this_stim_dir_trs = this_stim_trs  # just combine all directions of saccades for now
        
        #  get the saccade start times for this stimulus direction
        this_stim_saccade_start_ts = saccade_info.ix['start_t',this_stim_dir_trs]
        
        # now just select saccades that start in this time range
        spot_move_t = 580 # approximate -- if I care about ms precision, calculate this for each trial
        this_min_t  = min_pre_saccade_t + spot_move_t
        this_max_t = max_pre_saccade_t + spot_move_t
        
        saccades_in_t1 = np.where(this_stim_saccade_start_ts > this_min_t)[0]
        saccades_in_t2 = np.where(this_stim_saccade_start_ts < this_max_t)[0]
        saccade_in_t_win_i = np.intersect1d(saccades_in_t1,saccades_in_t2)
        this_stim_t_saccades = this_stim_dir_trs[saccade_in_t_win_i]
        
        # get the number of saccades here
        n_this_saccades = len(this_stim_t_saccades)    
    
        sacc_amp_ax.axhline(color=green)
        # loop through all saccades
        for i in this_stim_t_saccades: 
            sacc_amp_ax.plot(saccade_amplitudes.ix[['prev','sacc','post'],i],color=grey)
            #show each point in a separate color
            sacc_amp_ax.plot(0,saccade_amplitudes.ix[['prev'],i],color=blue,marker='+',markersize=15,markeredgewidth=1.5)
            sacc_amp_ax.plot(1,saccade_amplitudes.ix[['sacc'],i],color=magenta,marker='+',markersize=15,markeredgewidth=1.5)
            sacc_amp_ax.plot(2,saccade_amplitudes.ix[['post'],i],color=black,marker='+',markersize=15,markeredgewidth=1.5)
        
        sacc_amp_ax.set_xlim([-.5, 2.5])
        
# overlay boxplot
#         bp = plt.boxplot(saccade_amplitudes.ix[['prev'],this_stim_t_saccades].as_matrix()\
#                         ,0,sym=' ',positions=[0])
#         plt.setp(bp['boxes'], color='black')
#         plt.setp(bp['whiskers'], color='black') 
#         plt.setp(bp['medians'], color='blue') 
#         
#         bp = plt.boxplot(saccade_amplitudes.ix[['sacc'],this_stim_t_saccades].as_matrix()\
#                         ,0,sym=' ',positions=[1])
#         plt.setp(bp['boxes'], color='black')
#         plt.setp(bp['whiskers'], color='black') 
#         plt.setp(bp['medians'], color='magenta') 
#         
#         bp = plt.boxplot(saccade_amplitudes.ix[['post'],this_stim_t_saccades].as_matrix()\
#                         ,0,sym=' ',positions=[2])
#         plt.setp(bp['boxes'], color='black')
#         plt.setp(bp['whiskers'], color='black') 
#         plt.setp(bp['medians'], color='black') 
      
      
    if not plot_by_stim:
        title_text = 'Spontaneous saccades '+str(-1*min_pre_saccade_t) + ' to ' + \
                        str(-1*max_pre_saccade_t) + ' ms before spot movement_collapsed_position'
    else:
        title_text = 'Spontaneous saccades '+str(-1*min_pre_saccade_t) + ' to ' + \
                        str(-1*max_pre_saccade_t) + ' ms before spot movement'
                        
                        
    # now clean up axes ____________________________________________
    
    # first column needs full labels ________        
    #all_sacc_amp_ax[0].set_ylim([0,10])
    
    # add a title 
             
    all_sacc_amp_ax[0].set_ylabel('Saccade amplitude (V)')
    all_sacc_amp_ax[0].set_xlabel('Trial')
    all_sacc_amp_ax[0].set_xticks([0,1,2])
    all_sacc_amp_ax[0].set_xticklabels(['previous','saccade','post'])
    
    #all_sacc_amp_ax[0].set_xticks([0,1,2],['previous','saccade','post'])
    
    # remove extra xtick labels
    for col in range(n_cols):
        if col > 0:
            plt.setp(all_sacc_amp_ax[col].get_yticklabels(), visible=False)
            plt.setp(all_sacc_amp_ax[col].get_xticklabels(), visible=False)
        
    fig.text(.05,.95,title_text,fontsize=14)
    
    saveas_path = '/Users/jamie/bin/figures/'
    plt.savefig(saveas_path+title_text+' amplitude population.png',\
                bbox_inches='tight',dpi=100)


def get_all_saccade_amplitudes(saccade_info,saccade_and_control_traces):
    # calculate the saccade amplitudes for all traces.
    # create a data structure in the same format as 
    # saccade info -- rows: post, prev, sacc columns: saccade id

    n_saccades = np.shape(saccade_info)[1]
    row_names = ['post','prev','sacc']
    saccade_amplitudes = pd.DataFrame(index=row_names,\
                                      columns=range(n_saccades))
    # stimulus starts moving at ~580 ms
    spot_move_t = 580
    saccade_baseline_window = range(625,650)
    saccade_window = range(650,850) 
    # window in which the position is usually at its peak or trough

    for i in range(n_saccades):
        for trace in row_names:
            baseline_lmr_mean = saccade_and_control_traces.\
                        ix[saccade_baseline_window,(i,trace)].mean()
            saccade_win_trace = saccade_and_control_traces.\
                        ix[saccade_window,(i,trace)].values
            extreme_i = np.argmax(np.abs(saccade_win_trace))

            this_saccade_amp = saccade_win_trace[extreme_i]-baseline_lmr_mean
            
            saccade_amplitudes.ix[trace,i] = this_saccade_amp
    
    return saccade_amplitudes