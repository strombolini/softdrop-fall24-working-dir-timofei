import time

import coffea.processor as processor
from coffea.analysis_tools import Weights
from coffea.nanoevents import NanoAODSchema
import awkward as ak
import numpy as np
import fastjet
import hist.dask as dah
import hist
import math

NanoAODSchema.warn_missing_crossrefs = False


class msdProcessor(processor.ProcessorABC):
    def __init__(self, isMC=False):
        ################################
        # INITIALIZE COFFEA PROCESSOR
        ################################

        # Histogram Axes
        msoftdrop_axis = hist.axis.Regular(40, 0, 400, name="msoftdrop", label=r"Jet $m_\mathrm{softdrop}$ [GeV]")

        self.msoftdrop_axis = msoftdrop_axis
        self.make_output = lambda: self.create_histogram_matrix(self.n, self.msoftdrop_axis)

    def create_histogram_matrix(self, n, msoftdrop_axis):
        histograms = {}
        histograms["sumw"] = 0
        
        for i in range(n):
            for j in range(n):

                #Makes a dictionary of b00, b01, etc for beta and z values
                histograms[f"b{i}{j}"] = dah.Hist(
                    msoftdrop_axis,
                    storage=hist.storage.Weight()
                )
        # Returns a list so output[0] is the dictionary:
        return [histograms]

    def normalize(self, val, cut=None):

        #Replaces None values with nan (not a number)
        
        if cut is None:
            return ak.fill_none(val, np.nan)
        else:
            return ak.fill_none(val[cut], np.nan)

    def process(self, events, beta, z_cut, n):
        #Main function call

        #Set global variable self.n to be n from .process(n)
        self.n = n
        output = self.make_output()
        ##################
        # OBJECT SELECTION
        ##################

        # For soft drop studies we care about the AK8 jets

        #Make sure the selection fj_cut equally applies across all events/jets
        
        fatjets = events.FatJet
        fj_cut = (fatjets.pt > 450) & (abs(fatjets.eta) < 2.5)

        print (fatjets.fields)
        
        # define DeepDoubleB tagger threshold
        ddb_score_threshold = 0.8  # Example threshold
        ddb_cut = fatjets.btagDDBvLV2 > ddb_score_threshold

        # Apply the DDB and Softdrop Cuts
        
        candidatejet = fatjets[ddb_cut & fj_cut]


        leadingjets = candidatejet[:, 0:1]
        one_leading_jet = (ak.num(leadingjets, axis=1) == 1)
        leadingjets = leadingjets[one_leading_jet]

        
        select_events = (one_leading_jet) & (ak.any(fj_cut, axis=1))
        leadingjets = ak.flatten(leadingjets, axis=0)

        jetpt = ak.firsts(leadingjets.pt)
        jeteta = ak.firsts(leadingjets.eta)
        jetmsoftdrop = ak.firsts(leadingjets.msoftdrop)

        pf = ak.flatten(leadingjets.constituents.pf, axis=1)
        jetdef = fastjet.JetDefinition(fastjet.cambridge_algorithm, 0.8)
        cluster = fastjet.ClusterSequence(pf, jetdef)

        #N2 energy correlator
        
        n2 = cluster.exclusive_jets_energy_correlator(func="nseries", npoint = 2)
        jetn2= n2
        
        def n2(beta, zcut):
            softdrop = fastjet.ClusterSequence(pf, jetdef).exclusive_jets_softdrop_grooming(beta=beta, symmetry_cut=zcut)
            softdrop_cluster = fastjet.ClusterSequence(softdrop.constituents, jetdef)
            n2=softdrop_cluster.exclusive_jets_energy_correlator(func="nseries", npoint=2)
            return n2

        beta_softdrop_matrix = [[None for _ in range(n)] for _ in range(n)]
        calc_jetmsoftdrop = [[None for _ in range(n)] for _ in range(n)]

        # ^^^ Three dimensional matrix, beta (i) and z_cut (j), but you have to take the first element [0] to find them.

        #Now we use a nested for loop - Start with a beta value (0, 1/n, 2n) and for each one make a histogram with all the z_cut values (0,1/n,2/n...)
        #(n)^2 determines how many total histograms are made.
        for i in range(n):
            for j in range(n):

                #Makes n number of partitions of i and j which determine how many beta's and zcuts we have. The provided beta and z_cut serve as a max
                #And 0 is the minimum
                
                beta_softdrop_matrix[i][j] = cluster.exclusive_jets_softdrop_grooming(
                    beta=beta*(i/n), symmetry_cut=z_cut*(j/n)
                )
                calc_jetmsoftdrop[i][j] = ak.flatten(beta_softdrop_matrix[i][j].msoftdrop, axis=0)

        ################
        # EVENT WEIGHTS
        ################
        weights = Weights(size=None, storeIndividual=True)
        output[0]["sumw"] = ak.sum(events.genWeight[select_events])
        weights.add("genweight", events.genWeight[select_events])

        ###################
        # FILL HISTOGRAMS
        ###################
        self.fill_histograms(output[0], calc_jetmsoftdrop, weights)

        return output

    def fill_histograms(self, output, calc_jetmsoftdrop, weights):
        for i in range(self.n):
            for j in range(self.n):
                output[f"b{i}{j}"].fill(
                msoftdrop=self.normalize(calc_jetmsoftdrop[i][j]),
                weight=weights.weight()
                )

    def postprocess(self, accumulator):
        return accumulator
