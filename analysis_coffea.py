#%%
import time
import numpy as np
from numba import njit
from scipy.special import voigt_profile
import boost_histogram as bh
import awkward as ak 
import uproot 
import matplotlib.pyplot as plt
import mplhep
import hist
mplhep.style.use("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)

import vector
vector.register_awkward()

# NanoEvents
from coffea.nanoevents import NanoEventsFactory, BaseSchema 
from coffea.nanoevents import NanoAODSchema
# Processors
import coffea.processor as processor

num_files_bparking18 = 100
num_files_bparking23 = 49
num_files_scouting = 145

n_cores = 8

class MyProcessorBParking18(processor.ProcessorABC):
    def __init__(self):
        
        jpsi_axis = hist.axis.Regular(name="jpsi", label="SV mass(GeV)", bins=50, start=2.5, stop=3.5)
        lowmass_axis = hist.axis.Regular(name="lowmass", label="SV mass(GeV)", bins=101, start=0, stop=22)
        dimu_mass_axis = hist.axis.Regular(name="mass", label="Dimuon mass(GeV)", bins=101, start=0, stop=100)
        mudxysig_axis = hist.axis.Regular(name="mudxysig", label="Muon dxysig", bins=101, start=0, stop=100)
        svdxysig_axis = hist.axis.Regular(name="svdxysig", label="SV dxysig", bins=101, start=0, stop=100)
        mudxy_axis = hist.axis.Regular(name="mudxy", label="Muon dxy (cm)", bins=101, start=0, stop=100)
        svdxy_axis = hist.axis.Regular(name="svdxy", label="SV dxy (cm)", bins=101, start=0, stop=100)

        self.output = processor.dict_accumulator({
            'num_events': 0,
            'jpsi_hist': hist.Hist(jpsi_axis),
            'lowmass_hist': hist.Hist(lowmass_axis),
            'dimu_mass_hist': hist.Hist(dimu_mass_axis),
            'dimu_lowmass_hist': hist.Hist(lowmass_axis),
            'mudxysig_hist': hist.Hist(mudxysig_axis),
            'svdxysig_hist': hist.Hist(svdxysig_axis),
            'mudxy_hist': hist.Hist(mudxy_axis),
            'svdxy_hist': hist.Hist(svdxy_axis)
        })
        
    def process(self, events):
        
        MuonSV = events.muonSV
        Muons = events.Muon

        
        
        #Cut on Muons
        mu_cut = (np.abs(Muons.dxy/Muons.dxyErr) > 5)
        Muons = Muons[mu_cut]

        #Sort SVs to get only min chi2/ndof one
        min_args = ak.argsort(np.abs(MuonSV['chi2']/MuonSV['ndof']), ascending=True)
        MuonSV = MuonSV[min_args]

        mu_pairs = ak.combinations(Muons, 2)
        mu1, mu2 = ak.unzip(mu_pairs)
        mu_mass = np.sqrt(2*mu1.pt*mu2.pt*(np.cosh(mu1.eta - mu2.eta) - np.cos(mu1.phi - mu2.phi)))
        #Charge cut (OS)
        mu_mass = mu_mass[(mu1.charge!=mu2.charge)]
        #dxy_cut = (np.abs(mu1.dxy) > 5) & (np.abs(mu2.dxy) > 5)

        #Cut on Muon SV's
        sv_cut = MuonSV["dxySig"] > 5
        MuonSV = MuonSV[sv_cut]
        
        
        #Get the first sorted MuonSV (remove nones) => MuonSV is now flattened
        MuonSV = ak.firsts(MuonSV)
        MuonSV = MuonSV[~ak.is_none(MuonSV)]
        

        self.output['num_events'] += len(events)
        self.output['jpsi_hist'].fill(MuonSV.mass)
        self.output['lowmass_hist'].fill(MuonSV.mass)
        self.output['dimu_mass_hist'].fill(ak.flatten(mu_mass))
        self.output['dimu_lowmass_hist'].fill(ak.flatten(mu_mass))
        self.output['mudxysig_hist'].fill(ak.flatten(np.abs(Muons.dxy/Muons.dxyErr)))
        self.output['svdxysig_hist'].fill(MuonSV['dxySig'])
        self.output['mudxy_hist'].fill(ak.flatten(np.abs(Muons.dxy)))
        self.output['svdxy_hist'].fill(MuonSV['dxy'])

        return self.output
    
    def postprocess(self, accumulator):
        pass

class MyProcessorBParking23(processor.ProcessorABC):
    def __init__(self):

        jpsi_axis = hist.axis.Regular(name="jpsi", label="SV mass(GeV)", bins=50, start=2.5, stop=3.5)
        #jpsi_disp_axis = hist.axis.Regular(name="jpsi_disp", label="SV mass(GeV)", bins=50, start=2.5, stop=3.5)
        lowmass_axis = hist.axis.Regular(name="lowmass", label="SV mass(GeV)", bins=101, start=0, stop=22)
        dimu_mass_axis = hist.axis.Regular(name="mass", label="Dimuon mass(GeV)", bins=101, start=0, stop=100)
        mudxysig_axis = hist.axis.Regular(name="mudxysig", label="Muon dxysig", bins=101, start=0, stop=100)
        svdxysig_axis = hist.axis.Regular(name="svdxysig", label="SV dxysig", bins=101, start=0, stop=100)
        mudxy_axis = hist.axis.Regular(name="mudxy", label="Muon dxy (cm)", bins=101, start=0, stop=100)
        svdxy_axis = hist.axis.Regular(name="svdxy", label="SV dxy (cm)", bins=101, start=0, stop=100)

        self.output = processor.dict_accumulator({
            'num_events': 0,
            'jpsi_hist': hist.Hist(jpsi_axis),
            'jpsi_disp_hist': hist.Hist(jpsi_axis),
            'jpsi_hist_orig': hist.Hist(jpsi_axis),
            'lowmass_hist': hist.Hist(lowmass_axis),
            'lowmass_hist_orig': hist.Hist(lowmass_axis),
            'dimu_mass_hist': hist.Hist(dimu_mass_axis),
            'dimu_lowmass_hist': hist.Hist(lowmass_axis),
            'mudxysig_hist': hist.Hist(mudxysig_axis),
            'svdxysig_hist': hist.Hist(svdxysig_axis),
            'mudxy_hist': hist.Hist(mudxy_axis),
            'svdxy_hist': hist.Hist(svdxy_axis)
        })

    def process(self, events):
        HLTDecision = events.HLT
        MuonSV = events.muonSV
        MuonSVOrig = events.SV
        Muons = events.Muon

        #Cut on displaced trigger
        #DispCut = HLTDecision['DoubleMu4_LowMass_Displaced'] == True

        #Cut on Muons
        mu_cut = (np.abs(Muons.dxy/Muons.dxyErr) > 5)
        Muons = Muons[mu_cut]

        #Sort SVs to get only min chi2/ndof one
        min_args = ak.argsort(np.abs(MuonSV['chi2']/MuonSV['ndof']), ascending=True)
        MuonSV = MuonSV[min_args]

        min_args_orig = ak.argsort(np.abs(MuonSVOrig['chi2']/MuonSVOrig['ndof']), ascending=True)
        MuonSVOrig = MuonSVOrig[min_args_orig]

        mu_pairs = ak.combinations(Muons, 2)
        mu1, mu2 = ak.unzip(mu_pairs)
        mu_mass = np.sqrt(2*mu1.pt*mu2.pt*(np.cosh(mu1.eta - mu2.eta) - np.cos(mu1.phi - mu2.phi)))
        #Charge cut (OS)
        mu_mass = mu_mass[(mu1.charge!=mu2.charge)]
        #dxy_cut = (np.abs(mu1.dxy) > 5) & (np.abs(mu2.dxy) > 5)

        #Cut on Muon SV's
        sv_cut = MuonSV["dxySig"] > 5
        MuonSV = MuonSV[sv_cut]
        MuonSVOrig = MuonSVOrig[(MuonSVOrig["dxySig"] > 5)]

        #Get trigger cut SVs
        #MuonSVDisp = MuonSV[DispCut]

        #Get the first sorted MuonSV (remove nones)
        MuonSV = ak.firsts(MuonSV)
        MuonSV = MuonSV[~ak.is_none(MuonSV)]
        #MuonSVDisp = ak.firsts(MuonSVDisp)
        #MuonSVDisp = MuonSVDisp[~ak.is_none(MuonSVDisp)]
        MuonSVOrig = ak.firsts(MuonSVOrig)
        MuonSVOrig = MuonSVOrig[~ak.is_none(MuonSVOrig)]

        self.output['num_events'] += len(events)
        self.output['jpsi_hist'].fill(MuonSV.mass)
        self.output['jpsi_hist_orig'].fill(MuonSVOrig.mass)
        #self.output['jpsi_disp_hist'].fill(MuonSVDisp.mass)
        self.output['lowmass_hist'].fill(MuonSV.mass)
        self.output['lowmass_hist_orig'].fill(MuonSVOrig.mass)
        self.output['dimu_mass_hist'].fill(ak.flatten(mu_mass))
        self.output['dimu_lowmass_hist'].fill(ak.flatten(mu_mass))
        self.output['mudxysig_hist'].fill(ak.flatten(np.abs(Muons.dxy/Muons.dxyErr)))
        self.output['svdxysig_hist'].fill(MuonSV['dxySig'])
        self.output['mudxy_hist'].fill(ak.flatten(np.abs(Muons.dxy)))
        self.output['svdxy_hist'].fill(MuonSV['dxy'])

        return self.output
    
    def postprocess(self, accumulator):
        pass

#96 file version
"""
class MyProcessorScouting(processor.ProcessorABC):
    def __init__(self):
        jpsi_axis = hist.axis.Regular(name="jpsi", label="SV mass(GeV)", bins=50, start=2.5, stop=3.5)
        lowmass_axis = hist.axis.Regular(name="lowmass", label="SV mass(GeV)", bins=101, start=0, stop=22)
        dimu_mass_axis = hist.axis.Regular(name="mass", label="Dimuon mass(GeV)", bins=101, start=0, stop=100)
        mudxysig_axis = hist.axis.Regular(name="mudxysig", label="Muon dxysig", bins=101, start=0, stop=100)
        svdxysig_axis = hist.axis.Regular(name="svdxysig", label="SV dxysig", bins=101, start=0, stop=100)
        mudxy_axis = hist.axis.Regular(name="mudxy", label="Muon dxy (cm)", bins=101, start=0, stop=100)
        svdxy_axis = hist.axis.Regular(name="svdxy", label="SV dxy (cm)", bins=101, start=0, stop=100)

        self.output = processor.dict_accumulator({
            'num_events': 0,
            'jpsi_hist': hist.Hist(jpsi_axis),
            'lowmass_hist': hist.Hist(lowmass_axis),
            'dimu_mass_hist': hist.Hist(dimu_mass_axis),
            'dimu_lowmass_hist': hist.Hist(lowmass_axis),
            'mudxysig_hist': hist.Hist(mudxysig_axis),
            'svdxysig_hist': hist.Hist(svdxysig_axis),
            'mudxy_hist': hist.Hist(mudxy_axis),
            'svdxy_hist': hist.Hist(svdxy_axis)
        })

    def process(self, events):

        EventPV = events.pVtx
        EventPV = ak.firsts(EventPV)
        MuonSV = events.sVtx
        Muons = events.Muon

        #Cut on Muons
        mu_cut = (np.abs(Muons.dxy/Muons.dxyerror) > 5)
        Muons = Muons[mu_cut]

        #Sort SVs to get only min chi2/ndof one
        min_args = ak.argsort(np.abs(MuonSV['chi2']/MuonSV['ndof']), ascending=True)
        MuonSV = MuonSV[min_args]

        mu_pairs = ak.combinations(Muons, 2)
        mu1, mu2 = ak.unzip(mu_pairs)
        mu_mass = np.sqrt(2*mu1.pt*mu2.pt*(np.cosh(mu1.eta - mu2.eta) - np.cos(mu1.phi - mu2.phi)))
        #Charge cut (OS)
        mu_mass = mu_mass[(mu1.charge!=mu2.charge)]

        #Modification where no extra calcs needed
        #Cut on Muon SV's
        sv_cut = MuonSV["dxySig"] > 5
        MuonSV = MuonSV[sv_cut]
        
        #Get the first sorted MuonSV (remove nones)
        MuonSV = ak.firsts(MuonSV)
        MuonSV = MuonSV[~ak.is_none(MuonSV)]

        self.output['num_events'] += len(events)
        self.output['jpsi_hist'].fill(MuonSV.mass)
        self.output['lowmass_hist'].fill(MuonSV.mass)
        self.output['dimu_mass_hist'].fill(ak.flatten(mu_mass))
        self.output['dimu_lowmass_hist'].fill(ak.flatten(mu_mass))
        self.output['mudxysig_hist'].fill(ak.flatten(np.abs(Muons.dxy/Muons.dxyerror)))
        #self.output['svdxysig_hist'].fill(dxysig)
        self.output['svdxysig_hist'].fill(MuonSV['dxySig'])
        self.output['mudxy_hist'].fill(ak.flatten(np.abs(Muons.dxy)))
        #self.output['svdxy_hist'].fill(dxy)
        self.output['svdxy_hist'].fill(MuonSV['dxy'])


        return self.output
    
    def postprocess(self, accumulator):
        pass
"""

#Only muonSV variables are relevant here
class MyProcessorScouting(processor.ProcessorABC):
    def __init__(self):
        jpsi_axis = hist.axis.Regular(name="jpsi", label="SV mass(GeV)", bins=50, start=2.5, stop=3.5)
        lowmass_axis = hist.axis.Regular(name="lowmass", label="SV mass(GeV)", bins=101, start=0, stop=22)
        svdxysig_axis = hist.axis.Regular(name="svdxysig", label="SV dxysig", bins=101, start=0, stop=100)
        svdxy_axis = hist.axis.Regular(name="svdxy", label="SV dxy (cm)", bins=101, start=0, stop=100)

        self.output = processor.dict_accumulator({
            'num_events': 0,
            'jpsi_hist': hist.Hist(jpsi_axis),
            'lowmass_hist': hist.Hist(lowmass_axis),
            'svdxysig_hist': hist.Hist(svdxysig_axis),
            'svdxy_hist': hist.Hist(svdxy_axis)
        })

    def process(self, events):

        MuonSV = events.muonSV


        #Sort SVs to get only min chi2/ndof one
        min_args = ak.argsort(np.abs(MuonSV['chi2']/MuonSV['ndof']), ascending=True)
        MuonSV = MuonSV[min_args]

        #Modification where no extra calcs needed
        #Cut on Muon SV's 

        #dxySig cut kills all the J/Psi yield
        sv_cut = MuonSV["dxySig"] > 5
        MuonSV = MuonSV[sv_cut]

        #Get the first sorted MuonSV (remove nones)
        MuonSV = ak.firsts(MuonSV)
        MuonSV = MuonSV[~ak.is_none(MuonSV)]

        self.output['num_events'] += len(events)
        self.output['jpsi_hist'].fill(MuonSV.mass)
        self.output['lowmass_hist'].fill(MuonSV.mass)
        self.output['svdxysig_hist'].fill(MuonSV['dxySig'])
        self.output['svdxy_hist'].fill(MuonSV['dxy'])


        return self.output
    
    def postprocess(self, accumulator):
        pass



#Process b parking 2018
start_bparking18 = time.time()

#Make the list
#bparking_list_file = open("fileset_bparking.txt", "r")
#bparking_list = bparking_list_file.read()
#bparking_list = bparking_list.split("\n")

#Manual method
bparking18_list = []

with open('fileset_bparking18_proc.txt', 'w') as file:
    for i in range(num_files_bparking18):
        fname = "/vols/cms/mc3909/bparkProductionAll_V1p0/ParkingBPH1_Run2018B-05May2019-v2_MINIAOD_v1p0_generationSync/output_{:d}.root".format((i+1))
        bparking18_list.append(fname)
        file.write(fname+"\n")


fileset_bparking18 = {"b-parking" : bparking18_list}


futures_run_bparking18 = processor.Runner(
    executor = processor.FuturesExecutor(compression=None, workers=n_cores),
    schema=NanoAODSchema
)

out_bparking18 = futures_run_bparking18(
    fileset_bparking18,
    treename='Events',
    processor_instance= MyProcessorBParking18()
)

end_bparking18 = time.time()
time_bparking18 = (end_bparking18 - start_bparking18)

print("\nB parking 2018 processed")

#Process b parking 2023
start_bparking23 = time.time()

bparking23_list = []

with open('fileset_bparking23_proc.txt', 'w') as file:
    for i in range(num_files_bparking23):
        fname = "/vols/cms/pb4918/StoreNTuple/BParking23C/output_{:d}.root".format((i+1))
        bparking23_list.append(fname)
        file.write(fname+"\n")

fileset_bparking23 = {"b-parking" : bparking23_list}


futures_run_bparking23 = processor.Runner(
    executor = processor.FuturesExecutor(compression=None, workers=n_cores),
    schema=NanoAODSchema
)

out_bparking23 = futures_run_bparking23(
    fileset_bparking23,
    treename='Events',
    processor_instance= MyProcessorBParking23()
)

end_bparking23 = time.time()
time_bparking23 = (end_bparking23 - start_bparking23)

print("\nB parking 2023 processed")


#Process scouting
start_scouting = time.time()

#scouting_list_file = open("fileset_scouting.txt", "r")
#scouting_list = scouting_list_file.read()
#scouting_list = scouting_list.split("\n")

scouting_list = []

"""
#List of failed CRAB jobs (96 file version)
skip_files = [17, 24, 40, 43]

with open('fileset_scouting_proc.txt', 'w') as file:
    for i in range(num_files_scouting):
        if (i+1) in skip_files: continue

        else:
            fname = "/vols/cms/pb4918/StoreNTuple/Scouting/2022FStoreExpanded/output_{:d}.root".format((i+1))
            scouting_list.append(fname)
            file.write(fname+"\n")
"""

#145 file version
with open('fileset_scouting_proc.txt', 'w') as file:
    for i in range(num_files_scouting):
        fname = "/vols/cms/pb4918/StoreNTuple/Scouting/2022FNanotron/output_{:d}.root".format((i+1))
        scouting_list.append(fname)
        file.write(fname+"\n")

fileset_scouting = {"scouting" : scouting_list}



futures_run_scouting = processor.Runner(
    executor = processor.FuturesExecutor(compression=None, workers=n_cores),
    schema=NanoAODSchema
)

out_scouting = futures_run_scouting(
    fileset_scouting,
    #treename='mmtree/tree',
    treename='Events',
    processor_instance= MyProcessorScouting()
)

end_scouting = time.time()
time_scouting = (end_scouting - start_scouting)

print("\nScouting processed")

event_num_bparking18 = out_bparking18['num_events']
event_num_bparking23 = out_bparking23['num_events']
event_num_scouting = out_scouting['num_events']


print("\nSummary:")
print("Number of b-parking events 2018 processed:", event_num_bparking18)
print("Processed in %.3f seconds"%(time_bparking18))
print("Number of b-parking events 2023 processed:", event_num_bparking23)
print("Processed in %.3f seconds"%(time_bparking23))
print("\nNumber of scouting events processed:", event_num_scouting)
print("Processed in %.3f seconds"%(time_scouting))

#%%
#Plotting stuff 
fig, ax = plt.subplots()

jpsi_hist_bparking18 = out_bparking18['jpsi_hist']
jpsi_hist_bparking23 = out_bparking23['jpsi_hist']
jpsi_hist_bparking23_orig = out_bparking23['jpsi_hist_orig']
jpsi_hist_scouting = out_scouting['jpsi_hist']


#Scale up to 1/fb
jpsi_hist_bparking18 = jpsi_hist_bparking18/(0.070226436/6)
jpsi_hist_bparking23 = jpsi_hist_bparking23/(0.205478142/8)
jpsi_hist_bparking23_orig = jpsi_hist_bparking23_orig/(0.205478142/8)
#96 file ver
#jpsi_hist_scouting = jpsi_hist_scouting/0.034031063
#145 file ver
jpsi_hist_scouting = jpsi_hist_scouting/0.053647820

mplhep.histplot(jpsi_hist_bparking18, color='blue', label='B-parking 2018', density=False)
mplhep.histplot(jpsi_hist_bparking23, color='orange', label='B-parking 2023 (muonSV)', density=False)
mplhep.histplot(jpsi_hist_bparking23_orig, color='green', label='B-parking 2023 (SV)', density=False)
mplhep.histplot(jpsi_hist_scouting, color='red', label='Scouting', density=False)
ax.set_xlabel('Dimuon invariant mass for SV (GeV)')
ax.set_ylabel('Secondary vertices')
ax.set_xlim(2.5,3.5)
ax.text(0.70,0.70, "SV dxysig > 5", color='black', fontsize=18, ha='left', transform=ax.transAxes)
ax.text(0.70,0.65, "min $\\chi^{2}$/ndof", color='black', fontsize=18, ha='left', transform=ax.transAxes)
ax.legend(loc='upper left')
ax.set_yscale('log')
mplhep.cms.label(data=True, label='Work In Progress', rlabel=r'Scaled to 1 $\mathrm{fb}^{-1}$')
#mplhep.cms.label(data=True, label='Work In Progress', rlabel=r'Normalised')
fig.savefig("plots/sv_jpsi_comparison.png")


fig, ax = plt.subplots()

lowmass_hist_bparking18 = out_bparking18['lowmass_hist']
lowmass_hist_bparking23 = out_bparking23['lowmass_hist']
lowmass_hist_bparking23_orig = out_bparking23['lowmass_hist_orig']
lowmass_hist_scouting = out_scouting['lowmass_hist']

#Scale up to 1/fb
lowmass_hist_bparking18 = lowmass_hist_bparking18/(0.070226436/6)
lowmass_hist_bparking23 = lowmass_hist_bparking23/(0.205478142/8)
lowmass_hist_bparking23_orig = lowmass_hist_bparking23_orig/(0.205478142/8)
#96 file ver
#lowmass_hist_scouting = lowmass_hist_scouting/0.034031063
#145 file ver
lowmass_hist_scouting = lowmass_hist_scouting/0.053647820

mplhep.histplot(lowmass_hist_bparking18, color='blue', label='B-parking 2018', density=False)
mplhep.histplot(lowmass_hist_bparking23, color='orange', label='B-parking 2023 (muonSV)', density=False)
mplhep.histplot(lowmass_hist_bparking23_orig, color='green', label='B-parking 2023 (SV)', density=False)
mplhep.histplot(lowmass_hist_scouting, color='red', label='Scouting', density=False)
ax.set_xlabel('Dimuon invariant mass for SV (GeV)')
ax.set_ylabel('Secondary vertices')
ax.text(0.80,0.75, "SV dxysig > 5", color='black', fontsize=18, ha='left', transform=ax.transAxes)
ax.text(0.80,0.70, "min $\\chi^{2}$/ndof", color='black', fontsize=18, ha='left', transform=ax.transAxes)
ax.legend(loc='upper right')
ax.set_yscale('log')
mplhep.cms.label(data=True, label='Work In Progress', rlabel=r'Scaled to 1 $\mathrm{fb}^{-1}$')
#mplhep.cms.label(data=True, label='Work In Progress', rlabel=r'Normalised')
fig.savefig("plots/sv_low_comparison.png")
#%%
#Writing histograms to root
outfile = uproot.recreate("histograms.root")


outfile["hist_jpsi_bparking18"] = jpsi_hist_bparking18.to_numpy()
outfile["hist_jpsi_bparking23"] = jpsi_hist_bparking23.to_numpy()
outfile["hist_jpsi_bparking23_orig"] = jpsi_hist_bparking23_orig.to_numpy()
outfile["hist_jpsi_scouting"] = jpsi_hist_scouting.to_numpy()