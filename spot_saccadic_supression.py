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
    def __init__(self, fname, protocol=''):
        if fname.endswith('.abf'):
            self.basename = ''.join(fname.split('.')[:-1])
            self.fname = fname
        else:
            self.basename = fname
            self.fname = self.basename + '.abf'  #check here for fname type 
        
        self.protocol = protocol
                  
    def open_abf(self,exclude_indicies=[]):  
        abf = read_abf(self.fname)
        # added features to exclude specific time intervals
        n_indicies = np.size(abf['stim_x']) #assume all channels have the same sample #s 
        inc_indicies = np.setdiff1d(range(n_indicies),exclude_indicies);
                   
        self.xstim = np.array(abf['stim_x'])[inc_indicies]
        self.ystim = np.array(abf['stim_y'])[inc_indicies]

        self.samples = np.arange(self.xstim.size)  #this is adjusted
        self.t = self.samples/float(1000) # sampled at 10,000 hz -- encode here?
 
        lwa_v = np.array(abf['l_wba'])[inc_indicies]
        rwa_v = np.array(abf['r_wba'])[inc_indicies]
        
        #store the raw wing beat amplitude for checking for nonflight
        self.lwa = lwa_v
        self.rwa = rwa_v
        
        self.lmr = self.lwa - self.rwa
        
        self.wbf = np.array(abf['wbf'])
        if 'in 10' in abf:
            self.ao = np.array(abf['in 10'])
        else:
            self.ao = np.array(abf['ao1'])
            
    def _is_flying(self, start_i, stop_i, wba_thres=0.5, flying_time_thresh=0.95):  #fix this critera
        #check that animal is flying 
        l_nonzero_samples = np.where(self.lwa[start_i:stop_i] > wba_thres)[0]
        r_nonzero_samples = np.where(self.rwa[start_i:stop_i] > wba_thres)[0]
        n_flying_samples = np.size(np.intersect1d(l_nonzero_samples,r_nonzero_samples))
        
        total_samples = stop_i-start_i
        
        is_flying = (float(n_flying_samples)/total_samples) > flying_time_thresh   
        return is_flying
        
           
#---------------------------------------------------------------------------#

class Spot_Saccadic_Supression(Flight):
    
    def process_fly(self, show_tr_time_parsing=False, ex_i=[]):  #does this interfere with the Flight_Phys init?
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
        print 'nonflight trials : ' + str(np.size(non_flight_trs)) + '/' + str(self.n_trs)
        
        #now remove these
        self.n_nonflight_trs = np.size(non_flight_trs)
        self.n_trs = self.n_trs - np.size(non_flight_trs)
        self.tr_starts = np.delete(self.tr_starts,non_flight_trs)  #index values of starting and stopping
        self.tr_stops = np.delete(self.tr_stops,non_flight_trs)
        #self.pre_loom_stim_ons = np.delete(self.pre_loom_stim_ons,non_flight_trs)
                    
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
        #now check that the y value is > -9 
        clean_tr_stop_candidates = np.setdiff1d(tr_stop,redundant_stops)+1
        
        clean_tr_stops = clean_tr_stop_candidates[np.where(self.ao[clean_tr_stop_candidates-50] > 2)]
        
        #check that first start is before first stop
        if clean_tr_stops[0] < clean_tr_starts[0]: 
            clean_tr_stops = np.delete(clean_tr_stops,0)
         
        #last stop is after last start
        if clean_tr_starts[-1] > clean_tr_stops[-1]:
            clean_tr_starts = np.delete(clean_tr_starts,len(clean_tr_starts)-1)
         
        #should check for same # of starts and stops
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
        
        #here remove all trials in which the fly is not flying. 
        #self.remove_non_flight_trs()
        
    def parse_stim_type(self):
        #calculate the stimulus type
       
        ## 3.80; % [1] Spot on L, moving back to front (4th stimulus, in order)
        ## 3.90; % [2] Spot on L, moving front to back (3rd stimulus, in order)
        ## 2.50; % [3] Spot on R, moving back to front (2nd stimulus, in order)
        ## 2.40; % [4] Spot on R, moving front to back (1st stimulus, in order)
        
        self.stim_types_labels = {24:'Spot on right, 1 p offset, front to back',\
                            25:'Spot on right, 1 p offset, back to front',\
                            38:'Spot on left, 1 p offset, back to front',\
                            39:'Spot on left, 1 p offset, front to back' ,\
                            44:'Spot on right, .5 p offset, front to back',\
                            46:'Spot on left, .5 p offset, front to back'}
       
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
        
        
        
            
    def plot_wba_by_cnd(self,title_txt='',long_static_spot=False,wba_lim=[-1.5,1.5],filter_cutoff=12,tr_range=slice(None), if_save=True): 
        # plot single trace of each of the four saccadic movement conditions
        
        sampling_rate = 1000            # in hertz ********* move to fly info
        s_iti = .25 * sampling_rate      # ********* move to fly info
        
        baseline_win = range(0*sampling_rate,int(.125*sampling_rate)) 
        
        #get all traces and detect saccades ______________________________________________
        all_fly_traces, all_fly_saccades = self.get_traces_by_stim('this_fly',s_iti,get_saccades=False)
             
        n_cols = 4 
        n_rows = 2
        
        cnds_to_plot = self.unique_stim_types
        gs = gridspec.GridSpec(n_rows,n_cols,height_ratios=[1,.15,])

        fig = plt.figure(figsize=(16.5, 9))
        gs.update(wspace=0.1, hspace=0.2) # set the spacing between axes. 
        
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
    
            # loop single trials and plot all signals ________________________________
            for tr, i in zip(this_cnd_trs,range(n_cnd_trs)):

                this_color = this_color = scalarMap.to_rgba(i)     
     
                # plot WBA signal ____________________________________________________           
                wba_trace = all_fly_traces.loc[:,('this_fly',tr,cnd,'lmr')]
    
                baseline = np.nanmean(wba_trace[baseline_win])
                wba_trace = wba_trace - baseline  
     
                non_nan_i = np.where(~np.isnan(wba_trace))[0] 
                filtered_wba_trace = butter_lowpass_filter(wba_trace[non_nan_i],cutoff=filter_cutoff)
                 
                wba_ax.plot(filtered_wba_trace,color=this_color)
                #wba_ax.plot(wba_trace,color=this_color)
               
                #now plot stimulus traces ____________________________________________
                stim_ax.plot(all_fly_traces.loc[:,('this_fly',tr,cnd,'xstim')],color=this_color)
                
                
            # now plot the condition mean ____________________________________________
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
                all_wba_ax[col].axvspan(1630, 1730, facecolor='grey', alpha=0.5)    
            else:
                all_wba_ax[col].axvspan(630, 730, facecolor='grey', alpha=0.5)    
            
            # remove all time xticklabels __________________________________
            all_wba_ax[col].tick_params(labelbottom='off')
            
            # label columns
            all_wba_ax[col].set_title(self.stim_types_labels[cnds_to_plot[col]],fontsize=12)
            
            
            if col == 0:           
                all_wba_ax[col].set_ylabel('L-R WBA (V)')
            
                all_wba_ax[col].set_ylim(wba_lim)
                all_wba_ax[col].set_yticks([wba_lim[0],0,wba_lim[1]])
                
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
                    all_wba_ax[col].set_xlim([sampling_rate,2*sampling_rate]) #enforce max time
                else:
                    all_wba_ax[col].set_xlim([0, 1*sampling_rate]) #enforce max time
                
                
                all_stim_ax[col].xaxis.set_major_formatter(formatter)
                all_stim_ax[col].tick_params(labelbottom='on')
                all_stim_ax[col].tick_params(labelleft='off')
                all_stim_ax[col].set_xlabel('Time (s)')
            else:
                all_wba_ax[col].tick_params(labelleft='off')
                all_stim_ax[col].tick_params(labelleft='off')
                all_stim_ax[col].tick_params(labelbottom='off')
        
        figure_txt = title_txt
        fig.text(.2,.95,figure_txt,fontsize=18) 
       
        plt.draw()

        if if_save:
            saveas_path = '/Users/jamie/bin/figures/'
            plt.savefig(saveas_path + figure_txt + '_sacc_supression_wba_by_cnd_filtered_cutoff'+str(filter_cutoff)+'.png',\
                            bbox_inches='tight',dpi=100) 

    def plot_wba_by_cnd_y_offset(self,title_txt='',long_static_spot=False,trs_to_mark=[],tr_range=slice(None),filter_cutoff=24,if_save=True): 
        # plot single trace of each of the four saccadic movement conditions
        
        sampling_rate = 1000            # in hertz ********* move to fly info
        s_iti = .25 * sampling_rate      # ********* move to fly info
        tr_offset = 5.5
        
        
        #baseline_win = range(0*sampling_rate,int(.125*sampling_rate)) 
        baseline_win = range(0*sampling_rate,int(.1*sampling_rate)) 
        
        
        #get all traces and detect saccades ______________________________________________
        all_fly_traces, all_fly_saccades = self.get_traces_by_stim('this_fly',s_iti,get_saccades=False)
             
        n_cols = 4 
        n_rows = 2
        
        cnds_to_plot = self.unique_stim_types
        gs = gridspec.GridSpec(n_rows,n_cols,height_ratios=[1,.05])

        fig = plt.figure(figsize=(14.5, 14.5))
        gs.update(wspace=0.1, hspace=0.025) # set the spacing between axes. 
        
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
                filtered_wba_trace = butter_lowpass_filter(wba_trace[non_nan_i],cutoff=filter_cutoff)
                
                # check if stim,trial combination is in this list. if so, plot in 
                # a thick line
                if (col,i) in trs_to_mark:
                    wba_ax.plot(filtered_wba_trace+i/tr_offset,color=this_color,linewidth=5)    
                else:
                    wba_ax.plot(filtered_wba_trace+i/tr_offset,color=this_color,linewidth=1)    
                
                # now get potential saccade start times by differentiating the filtered
                # trace and then applying and threshold
                candidate_saccade_is = find_candidate_saccades(filtered_wba_trace,diff_thres=0.01)
                
                wba_ax.plot(candidate_saccade_is,filtered_wba_trace[candidate_saccade_is]+i/tr_offset,\
                            marker='*',linestyle='None',color=black)
                 
                 
                wba_ax.text(0,i/tr_offset,str(i),
                    verticalalignment='bottom', horizontalalignment='right',
                    color=this_color, fontsize=8)
                #transform=wba_ax.transAxes,
                    
               
                #cutoff=12
               
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
                all_wba_ax[col].axvspan(1630, 1730, facecolor='black', alpha=0.5)    
            else:
                all_wba_ax[col].axvspan(630, 730, facecolor='black', alpha=0.5)    
                
            
            # remove all time xticklabels __________________________________
            all_wba_ax[col].tick_params(labelbottom='off')
            
            # label columns
            all_wba_ax[col].set_title(self.stim_types_labels[cnds_to_plot[col]],fontsize=10)
            
            if col == 0:           
                all_wba_ax[col].set_ylabel('L-R WBA (V)',fontsize=10)
                this_ylim = all_wba_ax[col].get_ylim()
                #all_wba_ax[col].set_ylim([-.5,this_ylim[1]*.975])
                
                all_wba_ax[col].set_ylim([-.5,(i+2)/tr_offset])
                
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
                    all_wba_ax[col].set_xlim([sampling_rate, 2*sampling_rate]) #enforce max time
                else:
                    all_wba_ax[col].set_xlim([0, 1*sampling_rate]) #enforce max time
                
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
            saveas_path = '/Users/jamie/bin/figures/'
            plt.savefig(saveas_path + figure_txt + '_wba_by_cnd_overlay_filter_cutoff'+str(filter_cutoff)+'.png',\
                            bbox_inches='tight',dpi=100)     
        
        
        
                 
        
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
            this_start = this_loom_start + 2*iti  #- iti
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
        
def find_candidate_saccades(filtered_lmr_trace,diff_thres=0.005):
    # return the index positions within filtered_lmr_trace of candidate 
    # saccades, as shown by high derivates
    
    lmr_diff_above_thres = np.where(np.abs(np.diff(filtered_lmr_trace)) >= diff_thres)[0]

    # then find the first of sequential points 
    min_separation = 5
    above_thres_diff = np.where(np.diff(lmr_diff_above_thres) > min_separation)[0]
    potential_saccade_start_is = lmr_diff_above_thres[above_thres_diff+1]
    
    # now add the first start
    if lmr_diff_above_thres.size:
        potential_saccade_start_is = np.hstack([lmr_diff_above_thres[0],potential_saccade_start_is])
        potential_saccade_start_is = potential_saccade_start_is - 1
    
    return potential_saccade_start_is
    
             
def find_saccades(raw_lmr_trace,test_plot=False):
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

def butter_lowpass_filter(data, cutoff=12, fs=2000, order=5): #how does the order change?
    b, a = butter_lowpass(cutoff, fs, order)
    #y = sp.signal.lfilter(b, a, data) #what's the difference here? 
    y = sp.signal.filtfilt(b, a, data)
    return y
      
def write_to_pdf(f_name,figures_list):
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(fname)
    for f in figures_list:
        pp.savefig(f)
    pp.close()

def plot_many_flies(path_name, filenames_df):    

    #loop through all genotypes
    genotypes = (pd.unique(filenames_df.values[:,1]))
    print genotypes
    
    for g in genotypes:
        these_genotype_indicies = np.where(filenames_df.values[:,1] == g)[0]
    
        for index in these_genotype_indicies:
            print index
        
            fly = Looming_Behavior(path_name + filenames_df.values[index,0])
            title_txt = filenames_df.values[index,1] + '  ' + filenames_df.values[index,0]
            fly.process_fly()
            fly.plot_wba_stim(title_txt)
        
            saveas_path = '/Users/jamie/bin/figures/'
            plt.savefig(saveas_path + title_txt + '_kir_looming.png',dpi=100)
            plt.close('all')
                    
def get_pop_traces_df(path_name, population_f_names, one_task):  
    
    # within a task, loop through all genotypes
    # structure row = time points, aligned to trial  start
    # columns: genotype, fly, trial index, trial type, lmr/xstim

    this_experiment_flies = np.where(population_f_names['experiment'] == one_task)[0]
    
    #genotypes must be sorted to the labels for columns 
    genotypes = (pd.unique(population_f_names.loc[this_experiment_flies,'genotype']))
    genotypes = np.sort(genotypes)
    
    population_df = pd.DataFrame()
        
   #loop through each genotype  
    for g in genotypes: # remove this restriction ---------------
        this_genotype_indicies = this_experiment_flies[np.where(population_f_names.values[this_experiment_flies,1] == g)[0]]
        
        for index in this_genotype_indicies:  # remove this restriction ---------------
            
            fly_f_name = population_f_names.values[index,0]
            fly_col_name = g+' __ '+str(index).zfill(2)+' __ '+fly_f_name
            print fly_col_name
            
            fly = Spot_Adaptation_Flight(path_name + fly_f_name)
            fly.process_fly()
            fly_df, saccades_df = fly.get_traces_by_stim(fly_col_name)
            fly_df_subset = fly_df.loc[:,(slice(None),slice(None),slice(None),['lmr','xstim'])]
            
            population_df = pd.concat([population_df,fly_df],axis=1) # change this to pre-initialize the df size
            
    return population_df
     
def plot_pop_flight_behavior_histograms(population_df, wba_lim=[-3,3],cnds_to_plot=range(9)):  
    #for the looming data, plot histograms over time of all left-right
    #wba traces
    
    #instead send the population dataframe as a parameter
    
    #get a two-dimensional multi-indexed data frame with the population data
    #population_df = get_pop_flight_traces(path_name, population_f_names)
   
    #loop through each genotype  --- genotypes must be sorted to be column labels
    #change code so I just do this in the get_pop_flight_traces
    all_genotype_fields = population_df.columns.get_level_values(0)
    genotypes = np.unique(all_genotype_fields)
    
    x_lim = [0, 4075]
    
    for g in genotypes:
        print g
        
        #calculate the number of cells/genotype
        all_cell_names = population_df.loc[:,(g)].columns.get_level_values(0)
        n_cells = np.size(np.unique(all_cell_names))
        
        title_txt = g + ' __ ' + str(n_cells) + ' flies' #also add number of flies and trials here 
        #calculate the number of flies and trials for the caption
    
        fig = plt.figure(figsize=(16.5, 9))
        #change this so I'm not hardcoding the number of axes
        gs = gridspec.GridSpec(6,3,width_ratios=[1,1,1],height_ratios=[4,1,4,1,4,1])
    
        #loop through conditions -- later restrict these
        for cnd in cnds_to_plot:
            grid_row = int(2*math.floor(cnd/3)) #also hardcoding
            grid_col = int(cnd%3)
     
            #plot WBA histogram signal -----------------------------------------------------------    
            wba_ax = plt.subplot(gs[grid_row,grid_col])     
        
            g_lwa = population_df.loc[:,(g,slice(None),slice(None),cnd,'lwa')].as_matrix()
            g_rwa = population_df.loc[:,(g,slice(None),slice(None),cnd,'rwa')].as_matrix()    
            g_lmr = g_lwa - g_rwa
        
            #get baseline, substract from traces
            baseline = np.nanmean(g_lmr[200:700,:],0) #parametize this
            g_lmr = g_lmr - baseline
        
            #just plot the mean for debugging
            #wba_ax.plot(np.nanmean(g_lmr,1))
        
            #now plot the histograms over time. ------------
            max_t = np.shape(g_lmr)[0]
            n_trs = np.shape(g_lmr)[1]
                     
            t_points = range(max_t)
            t_matrix = np.tile(t_points,(n_trs,1))
            t_matrix_t = np.transpose(t_matrix)

            t_flat = t_matrix_t.flatten() 
            g_lmr_flat = g_lmr.flatten()

            #now remove nans
            g_lmr_flat = g_lmr_flat[~np.isnan(g_lmr_flat)]
            t_flat = t_flat[~np.isnan(g_lmr_flat)]

            #calc, plot histogram
            h2d, xedges, yedges = np.histogram2d(t_flat,g_lmr_flat,bins=[200,50],range=[[0, 4200],[-3,3]],normed=True)
            wba_ax.pcolormesh(xedges, yedges, np.transpose(h2d))
        
           
            #plot white line for 0 -----------
            wba_ax.axhline(color=white)
        
            wba_ax.set_xlim(x_lim) 
            
            if grid_row == 0 and grid_col == 0:
                wba_ax.yaxis.set_ticks(wba_lim)
                wba_ax.set_ylabel('L-R WBA (mV)')
            else:
                wba_ax.yaxis.set_ticks([])
            wba_ax.xaxis.set_ticks([])
              
            #now plot stim -----------------------------------------------------------
            stim_ax = plt.subplot(gs[grid_row+1,grid_col])
        
            #assume the first trace of each is typical
            y_stim = population_df.loc[:,(g,slice(None),slice(None),cnd,'ystim')]
            stim_ax.plot(y_stim.iloc[:,0],color=blue)
        
            stim_ax.set_xlim(x_lim) 
            stim_ax.set_ylim([0, 10]) 
        
            if grid_row == 4 and grid_col == 0:
                stim_ax.xaxis.set_ticks(x_lim)
                stim_ax.set_xticklabels(['0','.4075'])
                stim_ax.set_xlabel('Time (s)') 
            else:
                stim_ax.xaxis.set_ticks([])
            stim_ax.yaxis.set_ticks([])
            
        #now annotate        
        fig.text(.06,.8,'left',fontsize=14)
        fig.text(.06,.53,'center',fontsize=14)
        fig.text(.06,.25,'right',fontsize=14)
        
        fig.text(.22,.905,'22 l/v',fontsize=14)
        fig.text(.495,.905,'44 l/v',fontsize=14)
        fig.text(.775,.905,'88 l/v',fontsize=14)
        
        fig.text(.425,.95,title_txt,fontsize=18)        
        plt.draw() 

        saveas_path = '/Users/jamie/bin/figures/'
        plt.savefig(saveas_path + title_txt + '_population_kir_looming_histograms.png',dpi=100)
        #plt.close('all')

def plot_pop_flight_behavior_means(population_df, protocol, wba_lim=[-1,1]):  
    #plot the means of all left-right by condition wba traces
    #inputs --  population dataframe 
    
    all_fly_names = np.unique(population_df.columns.get_level_values(0))
    n_flies = np.size(all_fly_names)
        
    x_lim = [0, 4075]  #update this _________________________
    
    
    n_tr_thres = 40
    n_cols = 2
        
    if protocol == '1/f grating':
        cnds_to_plot = np.asarray([[3,2],[3,2],[1,0]])
        n_rows = 3
        stim_types_labels =['right 1/f grating',
                        'left 1/f grating',
                        'right middle spot',
                        'left middle spot',
                        'right middle spot',
                        'left middle spot']
        
    elif protocol == 'grating iti':
        cnds_to_plot = np.asarray([[1,0],[1,0]])
        n_rows = 2
        stim_types_labels =['right, middle height',
                            'left, middle height']
                            
    elif protocol == 'looming':
        cnds_to_plot = np.asarray([[2,1],[2,1],[0,0]])
        n_rows = 3
        stim_types_labels =['center looming',
                        'right middle spot',
                        'left middle spot',]
        
    elif protocol == 'control':
        cnds_to_plot = np.asarray([[1,0],[1,0],[1,0],[3,2]])
        n_rows = 4
        stim_types_labels =['right, middle height',
                    'left, middle height',
                    'right, top height',
                    'left, top height']
    
    fig = plt.figure(figsize=(7, 9))
    gs = gridspec.GridSpec(n_rows*2,n_cols,height_ratios=np.tile([4,1],n_rows))
        
        
    for fly in all_fly_names:
        print fly
        
        title_txt = protocol + ' ' + str(n_flies) + ' flies'  # add task name here
        
        if protocol == '1/f grating':
            title_txt = '1f grating ' + str(n_flies) + ' flies'
        
        # add block threshold selection in here _________________________    
        #loop through conditions  
        for row in range(n_rows):
            for col in range(n_cols):
                cnd = cnds_to_plot[row][col]
            
                # make the axis --------------------------------
                wba_ax = plt.subplot(gs[row*2,col])     
        
                # plot the mean of each fly --------------------------------
                for fly_name in all_fly_names:     
                    if row == 0:
                        fly_lmr = population_df.loc[:,(fly_name,slice(0,n_tr_thres),cnd,'lmr')].as_matrix()
                    else:
                        fly_lmr = population_df.loc[:,(fly_name,slice(n_tr_thres,200),cnd,'lmr')].as_matrix()
                        
                    # get baseline, substract from traces
                    baseline = np.nanmean(fly_lmr[0:1000,:],0) #parametize this # ____ parametize the baseline window
                    fly_lmr = fly_lmr - baseline
            
                    wba_ax.plot(np.nanmean(fly_lmr,1),color=black,linewidth=.5)      
        
                # plot the genotype mean -------------------------------- 
                if row == 0:
                    cnd_lmr = population_df.loc[:,(slice(None),slice(0,n_tr_thres),cnd,'lmr')].as_matrix()
                else:
                    cnd_lmr = population_df.loc[:,(slice(None),slice(n_tr_thres,200),cnd,'lmr')].as_matrix()
                
                # get baseline, subtract from traces
                cnd_baseline = np.nanmean(cnd_lmr[0:1000,:],0) #parametize this
                cnd_lmr = cnd_lmr - cnd_baseline
            
                wba_ax.plot(np.nanmean(cnd_lmr,1),color=magenta,linewidth=2)
               
                #format axis --------------------------------
                wba_ax.set_title(stim_types_labels[cnd])
                wba_ax.axhline(color=blue)
        
                wba_ax.set_xlim([5000,5500]) 
                wba_ax.set_ylim(wba_lim)
                         
                if row == 0 and col == 0:
                    wba_ax.yaxis.set_ticks(wba_lim)
                    wba_ax.set_ylabel('L-R WBA (mV)')
                else:
                    wba_ax.set_yticklabels([])
                wba_ax.set_xticklabels([])
              
                # now plot stim -----------------------------------------------------------
                stim_ax = plt.subplot(gs[row*2+1,col])
            
                # assume the first trace of each is typical
                x_stim = population_df.loc[:,(slice(None),slice(None),cnd,'xstim')].as_matrix()
                stim_ax.plot(x_stim[:,0])

                x_lim = [5000,5500]
                stim_ax.set_xlim([5000,5500])  
                stim_ax.set_ylim([0, 10]) 
         
                if row == n_rows-1 and col == 0:
                    stim_ax.xaxis.set_ticks(x_lim)
                    stim_ax.set_xticklabels(['0','0.5'])
                    stim_ax.set_xlabel('Time (s)') 
                else:
                    stim_ax.set_xticklabels([''])
                stim_ax.set_yticklabels([''])
        
        
        fig.text(.425,.95,title_txt,fontsize=18)        
        plt.draw() 

        saveas_path = '/Users/jamie/bin/figures/'
        plt.savefig(saveas_path + title_txt + '_population_tethered_flight.png',dpi=100)

        
def plot_pop_flight_behavior_means_overlay(population_df, two_genotypes, wba_lim=[-3,3], cnds_to_plot=range(9)):  
    #for the looming data, plot the means of all left-right
    #wba traces
    
    #instead send the population dataframe as a parameter
    
    #get a two-dimensional multi-indexed data frame with the population data
    #population_df = get_pop_flight_traces(path_name, population_f_names)
   
    #loop through each genotype  --- genotypes must be sorted to be column labels
    #change code so I just do this in the get_pop_flight_traces
    all_genotype_fields = population_df.columns.get_level_values(0)
    genotypes = np.unique(all_genotype_fields)
    
    x_lim = [0, 4075]
    speed_x_lims = [range(0,2600),range(0,3115),range(0,4075)] #restrict the xlims by condition to not show erroneously long traces
    
    fig = plt.figure(figsize=(16.5, 9))
    #change this so I'm not hardcoding the number of axes
    gs = gridspec.GridSpec(6,3,width_ratios=[1,1,1],height_ratios=[4,1,4,1,4,1])
    
    genotype_colors = [magenta, blue]
    
    i = 0 
    title_txt = '';
    for g in two_genotypes:
        print g
        
        #calculate the number of cells/genotype
        all_fly_names = population_df.loc[:,(g)].columns.get_level_values(0)
        unique_fly_names = np.unique(all_fly_names)
        n_cells = np.size(unique_fly_names)
        
        title_txt = title_txt + g + ' __ ' + str(n_cells) + ' flies ' #also add number of flies and trials here 
        #calculate the number of flies and trials for the caption
        
        #loop through conditions -- later restrict these
        for cnd in cnds_to_plot:
            grid_row = int(2*math.floor(cnd/3)) #also hardcoding
            grid_col = int(cnd%3)
            this_x_lim = speed_x_lims[grid_col]
     
            #make the axis --------------------------------
            wba_ax = plt.subplot(gs[grid_row,grid_col])     
        
            #plot the mean of each fly --------------------------------
            for fly_name in unique_fly_names:
                fly_lwa = population_df.loc[:,(g,fly_name,slice(None),cnd,'lwa')].as_matrix()
                fly_rwa = population_df.loc[:,(g,fly_name,slice(None),cnd,'rwa')].as_matrix()    
                fly_lmr = fly_lwa - fly_rwa
        
                #get baseline, substract from traces
                baseline = np.nanmean(fly_lmr[200:700,:],0) #parametize this
                fly_lmr = fly_lmr - baseline
            
                wba_ax.plot(np.nanmean(fly_lmr[this_x_lim,:],1),color=genotype_colors[i],linewidth=.25)        
        
            #plot the genotype mean --------------------------------   
            g_lwa = population_df.loc[:,(g,slice(None),slice(None),cnd,'lwa')].as_matrix()
            g_rwa = population_df.loc[:,(g,slice(None),slice(None),cnd,'rwa')].as_matrix()    
            g_lmr = g_lwa - g_rwa
        
            #get baseline, substract from traces
            baseline = np.nanmean(g_lmr[200:700,:],0) #parametize this
            g_lmr = g_lmr - baseline
            
            wba_ax.plot(np.nanmean(g_lmr[this_x_lim,:],1),color=genotype_colors[i],linewidth=2)
              
            #plot black line for 0 --------------------------------
            wba_ax.axhline(color=black)

            #format axis --------------------------------
            wba_ax.set_xlim(x_lim) 
            wba_ax.set_ylim(wba_lim)

            if grid_row == 0 and grid_col == 0:
                wba_ax.yaxis.set_ticks(wba_lim)
                wba_ax.set_ylabel('L-R WBA (mV)')
            else:
                wba_ax.yaxis.set_ticks([])
            wba_ax.xaxis.set_ticks([])
          
            #now plot stim -----------------------------------------------------------
            stim_ax = plt.subplot(gs[grid_row+1,grid_col])

            #assume the first trace of each is typical
            y_stim = population_df.loc[:,(g,slice(None),slice(None),cnd,'ystim')]
            stim_ax.plot(y_stim.iloc[:,0],color=black)

            stim_ax.set_xlim(x_lim) 
            stim_ax.set_ylim([0, 10]) 

            if grid_row == 4 and grid_col == 0:
                stim_ax.xaxis.set_ticks(x_lim)
                stim_ax.set_xticklabels(['0','.4075'])
                stim_ax.set_xlabel('Time (s)') 
            else:
                stim_ax.xaxis.set_ticks([])
            stim_ax.yaxis.set_ticks([])
            
        i = i + 1
        
    #now annotate        
    fig.text(.06,.8,'left',fontsize=14)
    fig.text(.06,.53,'center',fontsize=14)
    fig.text(.06,.25,'right',fontsize=14)
    
    fig.text(.22,.905,'22 l/v',fontsize=14)
    fig.text(.495,.905,'44 l/v',fontsize=14)
    fig.text(.775,.905,'88 l/v',fontsize=14)        

    fig.text(.1,.95,two_genotypes[0],color='magenta',fontsize=18)
    fig.text(.2,.95,two_genotypes[1],color='blue',fontsize=18)
    plt.draw()
    
    saveas_path = '/Users/jamie/bin/figures/'
    plt.savefig(saveas_path + title_txt + '_population_kir_looming_means_overlay_' 
        + two_genotypes[0] + '_' + two_genotypes[1] + '.png',dpi=100)
    #plt.close('all')

def plot_pop_flight_over_time(all_fnames,protocol,title_txt='',wba_lim=[-1.5,1.5],if_save=True): 
        # clean this up --
        # first store all points by vectorizing
        # change from plot -> get with boolean for plotting
        # make a separate function for plotting the population change over time
        #
        # this seems to work well, but I need to to show the windows of the saccades
        
        #get all traces and detect saccades ______________________________________________
        
        
        behavior_path = '/Users/jamie/Dropbox/maimon lab - behavioral data/'
        fig = plt.figure(figsize=(9.5,11.5))       #(16.5, 9))
        
        if protocol == 'looming':
            n_cnds = 3
        else:
            n_cnds = 4
            
        cnds_to_plot = range(n_cnds)
        all_colors = [blue,magenta,green,black]
        
        n_flies = np.size(all_fnames)
        print n_flies
        n_cnds = n_cnds
        max_trs = 150
        
        fly_traces_by_cnd = np.nan*np.ones([n_flies,n_cnds,max_trs])
         
        for f_name,fly_i in zip(all_fnames,range(n_flies)): #each fly
            # fly init
            # get fly conditions and traces
            
            fly = Spot_Adaptation_Flight(behavior_path + f_name,True)
            fly.process_fly()
            lmr_avg, cnd_types = fly.get_flight_over_time()

            
            for cnd in cnds_to_plot:
                this_cnd_trs = np.where(cnd_types == cnd)[0]
                n_trs = np.size(this_cnd_trs)
                this_color = all_colors[cnd]
                
                if cnd >= 2 and protocol == 'control':
                    plt.plot(np.arange(n_trs)+50,lmr_avg[this_cnd_trs],'-',color=this_color)
                elif cnd < 2 and protocol == '1f grating':
                    plt.plot(np.arange(n_trs)+20,lmr_avg[this_cnd_trs],'-',color=this_color)
                elif cnd == 0 and protocol == 'looming':
                    plt.plot(np.arange(n_trs)+20,lmr_avg[this_cnd_trs],'-',color=this_color)
                else:
                    plt.plot(range(n_trs),lmr_avg[this_cnd_trs],'-',color=this_color)
        
                #save all means/fly
                fly_traces_by_cnd[fly_i,cnd,range(0,n_trs)] = lmr_avg[this_cnd_trs]
        
        for cnd in cnds_to_plot:
            if cnd >= 2 and protocol == 'control':
                plt.plot(np.arange(max_trs)+50,np.nanmean(fly_traces_by_cnd,0)[cnd],color=all_colors[cnd],linewidth=4)
            elif cnd < 2 and protocol == '1f grating':
                plt.plot(np.arange(max_trs)+20,np.nanmean(fly_traces_by_cnd,0)[cnd],color=all_colors[cnd],linewidth=4)
            elif cnd == 0 and protocol == 'looming':
                plt.plot(np.arange(max_trs)+20,np.nanmean(fly_traces_by_cnd,0)[cnd],color=all_colors[cnd],linewidth=4)
            else:
                plt.plot(np.arange(max_trs),np.nanmean(fly_traces_by_cnd,0)[cnd],color=all_colors[cnd],linewidth=4)
            
            
        plt.axhline(linewidth=.5, color=black)
                
        plt.xlabel('Trial number')
        plt.ylabel('L-R WBA in turn window')  
        plt.title(title_txt,fontsize=18)  
        plt.ylim([-1.5,1.5])
        plt.xlim([0,60])
       
        if protocol == '1f grating':
            stim_types_labels =['right 1/f grating',
                        'left 1/f grating',
                        'right middle spot',
                        'left middle spot',
                        'right middle spot',
                        'left middle spot']
                        
        elif protocol == 'looming':
            stim_types_labels =['center looming',
                        'right middle spot',
                        'left middle spot',]
                            
        else: # grating iti, control
            stim_types_labels =['right, middle height',
                                'left, middle height',
                                'right, top height',
                                'left, top height']
        
        for cnd in range(np.size(cnds_to_plot)):
            fig.text(.7,.85-.03*cnd,stim_types_labels[cnd],color=all_colors[cnds_to_plot[cnd]],fontsize=14) 
        
       
        if if_save:
            saveas_path = '/Users/jamie/bin/figures/'
            if protocol == '1/f grating':
                plt.savefig(saveas_path + title_txt + 'population_turn_adaptation.png',\
                                    bbox_inches='tight',dpi=100) 
        
            else:
                plt.savefig(saveas_path + title_txt + 'population_turn_adaptation.png',\
                                    bbox_inches='tight',dpi=100) 
        
def get_saccade_and_control_traces(saccades_dict):
    # build a matrix of all traces with a saccade and their pre and post trials
    # includes all filtering and baseline substraction while building the matricies 
    # 
    # input: saccades dictionary in format
    # saccades_dict['03_30_0000']=[(0,0),(0,1)]  
    # fly filename -- list of tupes with condition #, tr within the condition

    all_saccades_structured = saccades_dict.values()
    n_saccades = len(sum(all_saccades_structured,[]))
    print n_saccades
    max_t = 1000
    
    all_saccade_traces = pd.DataFrame(index=range(max_t),columns=range(n_saccades))
    all_pre_traces = pd.DataFrame(index=range(max_t),columns=range(n_saccades))
    all_post_traces = pd.DataFrame(index=range(max_t),columns=range(n_saccades))

    all_stim_types = pd.Series(index=range(n_saccades),dtype=int)
    all_cell_names = pd.Series(index=range(n_saccades),dtype=str)
    
    baseline_window = range(0,150)
    
    saccade_i = 0
    
    for f_name in saccades_dict.keys():
    
        fly = Spot_Saccadic_Supression(path_name + '2015_'+ f_name)
        fly.process_fly(False)
    
        all_traces, saccades = fly.get_traces_by_stim()
        stim_types = fly.unique_stim_types
    
        this_fly_saccades = saccades_dict[f_name]
        for saccade_stim_tr in this_fly_saccades: 
            stim_i = saccade_stim_tr[0]
            stim_tr_i = saccade_stim_tr[1]
        
            this_stim_traces = all_traces.loc[:,('this_fly',slice(None),stim_types[stim_i],'lmr')]
            this_saccade_trace = this_stim_traces.iloc[0:max_t,(stim_tr_i)]
        
            if stim_tr_i == 0:
                this_prev_trace = this_stim_traces.iloc[0:max_t,(1)] #take trace following
            else:
                this_prev_trace = this_stim_traces.iloc[0:max_t,(stim_tr_i-1)]
            
            this_post_trace = this_stim_traces.iloc[0:max_t,(stim_tr_i+1)]
            
            
            filtered_saccade_trace = butter_lowpass_filter(this_saccade_trace,cutoff=48) 
            processed_saccade_trace = filtered_saccade_trace-\
                                      np.nanmean(filtered_saccade_trace[baseline_window])
        
            filtered_prev_trace = butter_lowpass_filter(this_prev_trace,cutoff=48) 
            processed_prev_trace = filtered_prev_trace-\
                                      np.nanmean(filtered_prev_trace[baseline_window]) 
        
            filtered_post_trace = butter_lowpass_filter(this_post_trace,cutoff=48) 
            processed_post_trace = filtered_prev_trace-\
                                      np.nanmean(filtered_post_trace[baseline_window])
        
            all_saccade_traces.ix[:,saccade_i] = processed_saccade_trace
            all_prev_traces.ix[:,saccade_i] = processed_prev_trace
            all_post_traces.ix[:,saccade_i] = processed_post_trace
        
            all_stim_types[saccade_i] = stim_i
            all_cell_names[saccade_i] = f_name
            saccade_i = saccade_i + 1    

def calculate_saccade_latencies(all_saccade_traces):
    # find the onset time of spont saccades
    # inputs -- all_saccade_traces matrix of n_saccades x t
    
    n_saccades = np.size(all_saccade_traces)
    
    spont_saccade_times = pd.Series(index=range(n_saccades))
    plot_on = False
    for s in range(n_saccades):
        trace_abs_diff = 10*abs(np.diff(all_saccade_traces.ix[:,s]))
        max_i = np.argmax(trace_abs_diff[150:600])+150
    
        if plot_on:
            fig = plt.figure()
            plt.plot(all_saccade_traces.ix[:,s])
            plt.plot(trace_abs_diff,color=magenta)
            plt.plot(max_i,0,'go')
            plt.title(s)
    
    spont_saccade_times[s] = max_i
    
def plot_saccade_traces_with_controls_y_offset(right_stim_sorted,left_stim_sorted,\
                        right_stim_sacc_t,left_stim_sacc_t): #what parameters do I need? 

    for stim_subset,spont_sacc_ts,title in \
        zip([right_stim_sorted,left_stim_sorted],\
            [right_stim_sacc_t,left_stim_sacc_t],
            ['Right','Left']): 

        # plot right stim ________________________________________________
        # get a colormap
        n_stim = len(stim_subset)
        n_colors = n_stim
        cmap = plt.cm.get_cmap('spectral')
        cNorm = colors.Normalize(0,n_colors)
        scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cmap)

        # now combine the traces in a single plot
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

    

def plot_spot_saccade_versus_spont_saccade_time():  #specify all input arguments
    # now plot the saccade time v. 
    # mean response in a saccade window - mean in window just before
    # separately for left and right  spots
    # next consider direction of saccade x direction of spot
    # need to consider absolute displacement? 
    # now with approximate saccade onset times
    # replot the saccade traces v. controls 
    # within the same stimulus types
    # this function is still a work in progress
    # ** important -- I need to separate the directions of the saccades
    
    spont_saccade_times.sort()
    
    right_stim = np.hstack([np.where(all_stim_types == 0)[0],\
                       np.where(all_stim_types == 1)[0]])
    left_stim = np.hstack([np.where(all_stim_types == 2)[0],\
                       np.where(all_stim_types == 3)[0]])
                       
    right_stim_sorted = spont_saccade_times[right_stim].order().index
    right_stim_sacc_t = spont_saccade_times[right_stim].order().values
    left_stim_sorted = spont_saccade_times[left_stim].order().index
    left_stim_sacc_t = spont_saccade_times[left_stim].order().values
    
    for stim_subset,title in zip([right_stim_sorted,left_stim_sorted],\
                             ['Right','Left']): 

        # plot right stim ________________________________________________
        # get a colormap
        n_stim = len(stim_subset)
        n_colors = n_stim
        cmap = plt.cm.get_cmap('spectral')
        cNorm = colors.Normalize(0,n_colors)
        scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cmap)

        # now combine the traces in a single plot
        fig = plt.figure(figsize=(5,9.5))
        baseline_window = range(0,150)

        for i,s in zip(range(n_stim),stim_subset): #range(n_saccades):
            this_color = scalarMap.to_rgba(i)

            this_saccade_trace = all_saccade_traces.ix[:,s]
            this_control_trace = all_control_traces.ix[:,s]

            processed_saccade_trace = this_saccade_trace-np.nanmean(this_saccade_trace[baseline_window])
            processed_control_trace = this_control_trace-np.nanmean(this_control_trace[baseline_window])

            plt.plot(processed_saccade_trace+i/2.0,color=this_color,\
                     linewidth=3)
            plt.plot(processed_control_trace+i/2.0,color=this_color,\
                     linewidth=1)


        if title == 'Right':
            plt.ylim([-.55,9.5])
        else:
            plt.ylim([-.4,13])
        
        #plt.ylim([-.55,9.25])
        # show turn window
        plt.axvspan(620, 770, facecolor='black', alpha=0.5)             
        plt.xlabel('Time (ms)')
        plt.ylabel('L-R WBA (V)+tr offset')
        plt.title(title+' spot saccade traces, sorted by spont saccade')
        saveas_path = '/Users/jamie/bin/figures/'
        plt.savefig(saveas_path+ title+' spot saccade traces - sorted.png',\
                    bbox_inches='tight',dpi=100)

