import numpy as np
import os,sys
import pandas as pd
import torch
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap


class model_result():
    
    def __init__(self, prefix, sigfile, bkgfile, sigembedfile, bkgembedfile, aetype, bins, tpr_thresholds, loss_plotrange, xrange, yrange, areacutoff):
        with open(os.path.join(prefix, sigfile), 'rb') as handle:
            self.sigloss = pickle.load(handle)


        with open(os.path.join(prefix, bkgfile), 'rb') as handle:
            self.bkgloss = pickle.load(handle)

        with open(os.path.join(prefix, sigembedfile), 'rb') as handle:
            self.sigembedded = pickle.load(handle)

        with open(os.path.join(prefix, bkgembedfile), 'rb') as handle:
            self.bkgembedded = pickle.load(handle)

        self.aetype = aetype
        self.bins   = bins
        self.tpr_thresholds = tpr_thresholds
        self.loss_plotrange = loss_plotrange
        self.xrange = xrange
        self.yrange = yrange
        self.areacutoff = areacutoff

    def get_tpr_fpr(self):
        tpr = []
        fpr = []
        for cut in self.bins:
            if self.aetype == 'sig':
                tpr.append(np.where(self.sigloss<cut)[0].shape[0]/len(self.sigloss))
                fpr.append(np.where(self.bkgloss<cut)[0].shape[0]/len(self.bkgloss))
            if self.aetype == 'bkg':
                tpr.append(np.where(self.sigloss>cut)[0].shape[0]/len(self.sigloss))
                fpr.append(np.where(self.bkgloss>cut)[0].shape[0]/len(self.bkgloss))
        
        self.tpr = np.array(tpr)
        self.fpr = np.array(fpr)

        return self.tpr,self.fpr
    
    def get_threshold_at_tpr(self, tpr_val):

        return self.bins[np.where(self.tpr < tpr_val)[0][0]]


    def get_fpr_at_tpr(self, tpr_val):

        return self.fpr[np.where(self.tpr < tpr_val)[0][0]]


    def get_threshold_at_fpr(self, fpr_val):

        return self.bins[np.where(self.fpr < fpr_val)[0][0]]


    def plot_TPR_selected(self, ax, tpr_val):

        _, bins_sig, patches_sig = ax.hist(self.sigloss, bins=self.loss_plotrange, density=True, alpha=0.6, label='Signal Mixture');
        threshold = self.get_threshold_at_tpr(tpr_val)
        ax.hist(self.bkgloss, bins=self.loss_plotrange, density=True, alpha=0.4, label='Background');
        ax.axvline(threshold, color='red', label=f'Threshold@{int(tpr_val*100)}TPR')

        for i in range(len(bins_sig)-1):
            if bins_sig[i] >= threshold:
                plt.setp(patches_sig[i], facecolor="darkblue",alpha=0.4,  hatch='///')


        ax.legend()
        ax.set_title('Loss Distributions')


    def plot_FPR_selected(self, ax, tpr_val):
        ax.hist(self.sigloss, bins=self.loss_plotrange, density=True, alpha=0.4, label='Signal Mixture');
        _, bins_qcd, patches_qcd = ax.hist(self.bkgloss, bins=self.loss_plotrange, density=True, alpha=0.4, label='Background');
        ax.axvline(self.get_threshold_at_tpr(tpr_val), color='red', label=f'Threshold@{int(tpr_val*100)}TPR')

        for i in range(len(bins_qcd)-1):
            #print(bins_h[i])
            if bins_qcd[i] >= self.get_threshold_at_tpr(tpr_val):
                plt.setp(patches_qcd[i], facecolor="darkorange",alpha=0.7, hatch='///')
        ax.legend()

    def plot_ROC_with_point(self, ax, tpr_val, label):
        ax.plot(self.fpr,self.tpr,label=f"{label} ROC")
        ax.plot(self.get_fpr_at_tpr(tpr_val), tpr_val, marker="*", markersize=20, markeredgecolor="red", markerfacecolor="red")
        
        legend_elements = [Line2D([0], [0], color='C0', lw=4, label=f"{label} ROC"),
                           Line2D([0], [0],  marker="*", markersize=20, markeredgecolor="red", markerfacecolor="red", linestyle='None', label='Point on ROC'),
                           ]
        
        ax.set_xlabel(r'$1-\epsilon_{bkg}$', fontsize=20)
        ax.set_ylabel(r'$\epsilon_{sig}$', fontsize=20)

        ax.legend(handles=legend_elements, loc='lower right')
        ax.set_xlim([0.0,1.0])
        ax.set_ylim([0.0,1.0])
        ax.set_title('ROC')

    def plot_area_tpr(self, ax, tpr_val):
        xrange = self.xrange
        yrange = self.yrange
        patches = []
        sig_selected = np.where(self.sigloss>self.get_threshold_at_tpr(tpr_val))
        sig_hist_selected, xedges, yedges = np.histogram2d(self.sigembedded[sig_selected,0][0], self.sigembedded[sig_selected,1][0], (xrange, yrange))
        sig_hist_all, xedges, yedges = np.histogram2d(self.sigembedded[:,0], self.sigembedded[:,1], (xrange, yrange))
        print("range", xrange[0], xrange[-1])
        print("edges", xedges[0], xedges[-1])
        cmap0 = LinearSegmentedColormap.from_list('', ['white', 'darkblue'])
        cmap1 = LinearSegmentedColormap.from_list('', ['white', 'C0'])
        im = ax.imshow(np.where(sig_hist_selected>self.areacutoff, 1, 0).T, origin='lower', extent=[xrange[0], xrange[-1], yrange[0],yrange[-1]], cmap=cmap0,aspect='auto', alpha=0.6, interpolation='none')
        patches.append(Patch(color=im.cmap(im.norm(1.)), label = "Selected Signal", alpha=0.6) )
        im = ax.imshow(np.where(sig_hist_all>self.areacutoff, 1, 0).T, origin='lower', extent=[xrange[0], xrange[-1], yrange[0],yrange[-1]], cmap=cmap1,aspect='auto', alpha=0.3, interpolation='none')
        patches.append(Patch(color=im.cmap(im.norm(1.)), label = "Total Signal", alpha=0.3))
        ax.legend(handles=patches)
        ax.set_title('Embedding Space')

    def plot_area_fpr(self, ax, tpr_val):
        xrange = self.xrange
        yrange = self.yrange
        patches = []
        bkg_selected = np.where(self.bkgloss>self.get_threshold_at_tpr(tpr_val))
        bkg_hist_selected, xedges, yedges = np.histogram2d(self.bkgembedded[bkg_selected,0][0], self.bkgembedded[bkg_selected,1][0], (xrange, yrange))
        bkg_hist_all, xedges, yedges = np.histogram2d(self.bkgembedded[:,0], self.bkgembedded[:,1], (xrange, yrange))
        cmap0 = LinearSegmentedColormap.from_list('', ['white', 'darkorange'])
        cmap1 = LinearSegmentedColormap.from_list('', ['white', 'C1'])

        print(np.min(np.where(bkg_hist_selected>self.areacutoff, 1., 0.).T))
        im = ax.imshow(np.where(bkg_hist_selected>self.areacutoff, 1., 0.).T, origin='lower', extent=[xrange[0], xrange[-1], yrange[0],yrange[-1]], cmap=cmap0,aspect='auto', alpha=0.7, interpolation='none' )
        patches.append(Patch(color=im.cmap(im.norm(1.)), label = "Selected Background", alpha=0.6) )
        im = ax.imshow(np.where(bkg_hist_all>self.areacutoff, 1., 0.).T, origin='lower', extent=[xrange[0], xrange[-1], yrange[0],yrange[-1]], cmap=cmap0,aspect='auto', alpha=0.2, label = 'Total Background', interpolation='none')
        patches.append(Patch(color=im.cmap(im.norm(1.)), label = "Total Background", alpha=0.3))
        ax.legend(handles=patches)

    def get_area_tpr_fpr(self):
        area_tpr = []
        area_fpr = []
        xrange = self.xrange
        yrange = self.yrange
        for threshold in self.tpr_thresholds:
            sig_selected = np.where(self.sigloss>self.get_threshold_at_tpr(threshold))
            bkg_selected = np.where(self.bkgloss>self.get_threshold_at_tpr(threshold))
            
            sig_hist_selected, xedges, yedges = np.histogram2d(self.sigembedded[sig_selected,0][0], self.sigembedded[sig_selected,1][0], (xrange, yrange))
            sig_hist_all, xedges, yedges = np.histogram2d(self.sigembedded[:,0], self.sigembedded[:,1], (xrange, yrange))

            bkg_hist_selected, xedges, yedges = np.histogram2d(self.bkgembedded[bkg_selected,0][0], self.bkgembedded[bkg_selected,1][0], (xrange, yrange))
            bkg_hist_all, xedges, yedges = np.histogram2d(self.bkgembedded[:,0], self.bkgembedded[:,1], (xrange, yrange))

            area_tpr.append(np.where(np.where(sig_hist_selected>self.areacutoff, 1, 0).T == 1)[0].shape[0]/np.where(np.where(sig_hist_all>self.areacutoff, 1, 0).T == 1)[0].shape[0])
            area_fpr.append(np.where(np.where(bkg_hist_selected>self.areacutoff, 1, 0).T == 1)[0].shape[0]/np.where(np.where(bkg_hist_all>self.areacutoff, 1, 0).T == 1)[0].shape[0])

        self.area_tpr = area_tpr
        self.area_fpr = area_fpr
        return area_tpr, area_fpr


    def get_precision_recall(self):
        tpr = []
        fpr = []
        precision = []
        for cut in self.bins:
            if self.aetype == 'sig':
                tpr.append(np.where(self.sigloss<cut)[0].shape[0]/len(self.sigloss))
                precision.append((np.where(self.sigloss<cut)[0].shape[0])/(np.where(self.bkgloss<cut)[0].shape[0]+np.where(self.sigloss<cut)[0].shape[0]))
            
            if self.aetype == 'bkg':
                tpr.append(np.where(self.sigloss>cut)[0].shape[0]/len(self.sigloss))
                precision.append((np.where(self.sigloss>cut)[0].shape[0])/(np.where(self.bkgloss>cut)[0].shape[0]+np.where(self.sigloss>cut)[0].shape[0]))
        

        return precision,tpr  


    def get_area_tpr_fpr_at_TPR(self, tpr_val):
        sig_selected = np.where(self.sigloss>self.get_threshold_at_tpr(tpr_val))
        bkg_selected = np.where(self.bkgloss>self.get_threshold_at_tpr(tpr_val))
        sig_hist_selected, xedges, yedges = np.histogram2d(self.sigembedded[sig_selected,0][0], self.sigembedded[sig_selected,1][0], (np.arange(-0.6,0.3,.01), np.arange(0,0.35,.01)))
        sig_hist_all, xedges, yedges = np.histogram2d(self.sigembedded[:,0], self.sigembedded[:,1], (np.arange(-0.6,0.3,.01), np.arange(0,0.35,.01)))

        bkg_hist_selected, xedges, yedges = np.histogram2d(self.bkgembedded[bkg_selected,0][0], self.bkgembedded[bkg_selected,1][0], (np.arange(-0.6,0.3,.01), np.arange(0,0.35,.01)))
        bkg_hist_all, xedges, yedges = np.histogram2d(self.bkgembedded[:,0], self.bkgembedded[:,1], (np.arange(-0.6,0.3,.01), np.arange(0,0.35,.01)))

        area_tpr = np.where(np.where(sig_hist_selected>self.areacutoff, 1, 0).T == 1)[0].shape[0]/np.where(np.where(sig_hist_all>self.areacutoff, 1, 0).T == 1)[0].shape[0]
        area_fpr = np.where(np.where(bkg_hist_selected>self.areacutoff, 1, 0).T == 1)[0].shape[0]/np.where(np.where(bkg_hist_all>self.areacutoff, 1, 0).T == 1)[0].shape[0]
        return area_tpr, area_fpr



    def plot_area_ROC_with_point(self, ax, tpr_val, label):
        #print(self.area_fpr, self.area_tpr)
        ax.plot(self.area_fpr,self.area_tpr,label=f"{label} Area Adjusted ROC")
        tpr, fpr = self.get_area_tpr_fpr_at_TPR(tpr_val)
        #print(fpr, tpr)
        ax.plot(fpr, tpr, marker="*", markersize=20, markeredgecolor="red", markerfacecolor="red")

        
        ax.set_xlabel(r'$1-\epsilon_{area, bkg}$', fontsize=20)
        ax.set_ylabel(r'$\epsilon_{area, sig}$', fontsize=20)

        
        legend_elements = [Line2D([0], [0], color='C0', lw=4, label=f"{label} Area Adjusted ROC"),
                           Line2D([0], [0],  marker="*", markersize=20, markeredgecolor="red", markerfacecolor="red", linestyle='None', label='Translated Point'),
                           ]
        ax.legend(handles=legend_elements, loc='lower right')
        ax.set_title('Area Adjusted ROC')
        ax.set_xlim([0.0,1.0])
        ax.set_ylim([0.0,1.0])
        return tpr, fpr

    def plot_everything(self, tpr_val, label):
        fig = plt.figure(constrained_layout=False, facecolor='1.',  figsize=(28,12))
        gs = fig.add_gridspec(nrows=4, ncols=4, left=0.1, right=.9,
                              hspace=0.4, wspace=0.4)
        ax0 = fig.add_subplot(gs[1:3, 0])
        self.get_tpr_fpr()
        self.plot_ROC_with_point(ax0, tpr_val, label)

        ax1 = fig.add_subplot(gs[0:2, 1])
        self.plot_TPR_selected(ax1, tpr_val)

        ax2 = fig.add_subplot(gs[2:4, 1])
        self.plot_FPR_selected(ax2, tpr_val)

        ax3 = fig.add_subplot(gs[0:2, 2])
        self.plot_area_tpr(ax3, tpr_val)
        #ax3.text(1.5, .7, r'$\epsilon_{area,sig}=\frac{\mathrm{Selected\ Signal\ Area}}{\mathrm{Total\ Signal\ Area}}$',
        #        horizontalalignment='center',
        #        verticalalignment='center',
        #        fontsize=16,color='darkblue',
        #        transform=ax3.transAxes)
        ax3.annotate(r'$\epsilon_{area,sig}=\frac{\mathrm{Selected\ Signal\ Mixture\ Area}}{\mathrm{Total\ Signal\ Mixture\ Area}}$', (1.2, .4),
            xytext=(2.3, 0.8), xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',color='darkblue'),
            fontsize=28,color='darkblue',horizontalalignment='right', verticalalignment='top')
                         

        
        

        ax4 = fig.add_subplot(gs[2:4, 2])
        self.plot_area_fpr(ax4, tpr_val)
        #ax4.text(1.6, .2, r'$1-\epsilon_{area,bkg}=\frac{\mathrm{Selected\ Background\ Area}}{\mathrm{Total\ Background\ Area}}$',
        #        horizontalalignment='center',
        #        verticalalignment='center',
        #        fontsize=16,color='chocolate',
        #        transform=ax4.transAxes)
        ax4.annotate( r'$1-\epsilon_{area,bkg}=\frac{\mathrm{Selected\ Background(QCD)\ Area}}{\mathrm{Total\ Background(QCD)\ Area}}$', (1.2, .35),
            xytext=(2.6, .18), xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.5',color='chocolate'),
            fontsize=28,color='chocolate',horizontalalignment='right', verticalalignment='top')

        self.get_area_tpr_fpr()
        ax5 = fig.add_subplot(gs[1:3, 3])
        tpr, fpr = self.plot_area_ROC_with_point(ax5, tpr_val, label)

        con = ConnectionPatch(xyA=(1,0.6), xyB=(0,0.5), coordsA="axes fraction", coordsB="axes fraction",
                              axesA=ax0, axesB=ax1, color="crimson", lw=2, arrowstyle='->')
        fig.add_artist(con)

        con = ConnectionPatch(xyA=(1,0.5), xyB=(0,0.5), coordsA="axes fraction", coordsB="axes fraction",
                              axesA=ax1, axesB=ax3, color="crimson", lw=2, arrowstyle='->')
        fig.add_artist(con)

        con = ConnectionPatch(xyA=(0.8,0), xyB=(0,0.5), coordsA="axes fraction", coordsB="axes fraction",
                              axesA=ax0, axesB=ax2, color="crimson", lw=2, arrowstyle='->')
        fig.add_artist(con)

        con = ConnectionPatch(xyA=(1,0.5), xyB=(0,0.5), coordsA="axes fraction", coordsB="axes fraction",
                              axesA=ax2, axesB=ax4, color="crimson", lw=2, arrowstyle='->')
        fig.add_artist(con)

        con = ConnectionPatch(xyA=(1,0.5), xyB=(0,tpr), coordsA="axes fraction", coordsB="axes fraction",
                              axesA=ax3, axesB=ax5, color="darkblue", lw=2, arrowstyle='->')
        fig.add_artist(con)

        con = ConnectionPatch(xyA=(1,0.3), xyB=(fpr,0), coordsA="axes fraction", coordsB="axes fraction",
                              axesA=ax4, axesB=ax5, color="darkorange", lw=2, arrowstyle='->')
        fig.add_artist(con)

        fig.suptitle('Area Adjusted ROC Curve', horizontalalignment='center')
        return fig

    def FPRat95TPR(self):
        tprs, fprs = self.tpr, self.fpr
        for i in range(len(tprs)-1):
            if (tprs[i] < 0.95) and (tprs[i+1] >= 0.95):
                return fprs[i+1]

    def FPRat99TPR(self):
        tprs, fprs = self.tpr, self.fpr
        for i in range(len(tprs) - 1):
            if (tprs[i] < 0.99) and (tprs[i + 1] >= 0.99):
                return fprs[i+1]

