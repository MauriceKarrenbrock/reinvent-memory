# coding=utf-8

import abc
import json
import logging

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.Scaffolds import MurckoScaffold

from scaffold.ScaffoldMemory import ScaffoldMemory


class ScaffoldFilter(ScaffoldMemory):

    def __init__(self, nbmax=25, minscore=0.6, generic=False):
        super(ScaffoldFilter, self).__init__()
        self.nbmax = nbmax  # number of smiles for one scaffold to score until the penalizer starts
        self.minscore = minscore  # only add smiles with a minimum score into the memory
        self.generic = generic  # store generic scaffolds or normal murcko scaffolds?
        self._scaffoldfunc = self.getGenericScaffold if generic else self.getScaffold

    @abc.abstractmethod
    def score(self, smiles, scores_dict: dict) -> np.array:
        raise NotImplemented

    def validScores(self, smiles, scores) -> bool:
        if not len(smiles) == len(scores):
            logging.error("SMILES and score vector are not the same length. Do nothing")
            logging.debug(smiles)
            logging.debug(scores)
            return False
        else:
            return True

    def savetojson(self, file):
        savedict = {'nbmax':      self.nbmax, 'minscore': self.minscore, 'generic': self.generic,
                    "_scaffolds": self._scaffolds}
        jsonstr = json.dumps(savedict, sort_keys=True, indent=4, separators=(',', ': '))
        with open(file, 'w') as f:
            f.write(jsonstr)

    def savetocsv(self, file):
        df = {"Cluster": [], "Scaffold": [], "SMILES": []}
        for i, scaffold in enumerate(self._scaffolds):
            for smi, score in self._scaffolds[scaffold].items():
                df["Cluster"].append(i)
                df["Scaffold"].append(scaffold)
                df["SMILES"].append(smi)
                for item in score.keys():
                    if item in df:
                        df[item].append(score[item])
                    else:
                        df[item] = [score[item]]

        df = pd.DataFrame(df)
        df.to_csv(file, index=False)


class ScaffoldMatcher(ScaffoldFilter):
    def __init__(self, nbmax=25, minscore=0.6, generic=False):
        super().__init__(nbmax=nbmax, minscore=minscore, generic=generic)

    def score(self, smiles, scores_dict: dict) -> np.array:
        scores = scores_dict.pop("total_score")
        if not self.validScores(smiles, scores): return scores

        for i, smile in enumerate(smiles):
            score = scores[i]
            try:
                scaffold = self._scaffoldfunc(smile)
            except Exception:
                scaffold = ''
                scores[i] = 0
            if self.has(scaffold, smile):
                scores[i] = 0
            elif score >= self.minscore:
                save_score = {"total_score": float(score)}
                for k in scores_dict:
                    save_score[k] = float(scores_dict[k][i])
                self._update_memory([smile], [scaffold], [save_score])
                if len(self[scaffold]) > self.nbmax:
                    scores[i] = 0
        return scores

    def savetojson(self, file):
        savedict = {'nbmax':      self.nbmax, 'minscore': self.minscore, 'generic': self.generic,
                    "_scaffolds": self._scaffolds}
        jsonstr = json.dumps(savedict, sort_keys=True, indent=4, separators=(',', ': '))
        with open(file, 'w') as f:
            f.write(jsonstr)


class IdenticalMurckoScaffold(ScaffoldMatcher):
    """Penalizes compounds based on exact Murcko Scaffolds previously generated. 'minsimilarity' is ignored."""

    def __init__(self, nbmax=25, minscore=0.6, minsimilarity=0.6):
        super().__init__(nbmax=nbmax, minscore=minscore, generic=False)


class IdenticalTopologicalScaffold(ScaffoldMatcher):
    """Penalizes compounds based on exact Topological Scaffolds previously generated. 'minsimilarity' is ignored."""

    def __init__(self, nbmax=25, minscore=0.6, minsimilarity=0.6):
        super().__init__(nbmax=nbmax, minscore=minscore, generic=True)


class CompoundSimilarity(ScaffoldFilter):
    """Penalizes compounds based on the ECFP or FCFP Tanimoto similarity to previously generated compounds."""

    def __init__(self, nbmax=25, minscore=0.6, minsimilarity=0.6, radius=2, useFeatures=False, bits=2048):
        super().__init__(nbmax=nbmax, minscore=minscore, generic=False)
        self.minsimilarity = minsimilarity
        self.radius = radius
        self.useFeatures = useFeatures
        self.bits = bits

    def score(self, smiles, scores_dict: dict) -> np.array:
        scores = scores_dict.pop("total_score")
        if not self.validScores(smiles, scores): return scores

        for i, smile in enumerate(smiles):
            score = scores[i]
            if score >= self.minscore:
                cluster, fingerprint, isnewcluster = self.findCluster(smile)
                if self.has(cluster, smile):
                    scores[i] = 0
                    continue
                save_score = {"total_score": float(score)}
                for k in scores_dict:
                    save_score[k] = float(scores_dict[k][i])
                if isnewcluster:
                    self._update_memory([smile], [cluster], [save_score], [fingerprint])
                else:
                    self._update_memory([smile], [cluster], [save_score])
                if len(self[cluster]) > self.nbmax:
                    scores[i] = 0

        return scores

    def findCluster(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return "", "", False
        if self.bits > 0:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.bits, useFeatures=self.useFeatures)
        else:
            fp = AllChem.GetMorganFingerprint(mol, self.radius, useFeatures=self.useFeatures)

        if smiles in self.getFingerprints():
            return smiles, fp, False

        fps = list(self.getFingerprints().values())
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
        if len(sims) == 0:
            return smiles, fp, True
        closest = np.argmax(sims)
        if sims[closest] >= self.minsimilarity:
            return list(self.getFingerprints().keys())[closest], fp, False
        else:
            return smiles, fp, True


class ScaffoldSimilarity(CompoundSimilarity):
    """Penalizes compounds based on atompair Tanimoto similarity to previously generated Murcko Scaffolds."""

    def __init__(self, nbmax=25, minscore=0.6, minsimilarity=0.6):
        super().__init__(nbmax=nbmax, minscore=minscore, minsimilarity=minsimilarity)

    def score(self, smiles, scores_dict: dict) -> np.array:
        scores = scores_dict.pop("total_score")
        if not self.validScores(smiles, scores): return scores

        for i, smile in enumerate(smiles):
            score = scores[i]
            if score >= self.minscore:
                cluster, fingerprint, isnewcluster = self.findCluster(smile)
                if self.has(cluster, smile):
                    scores[i] = 0
                    continue
                save_score = {"total_score": float(score)}
                for k in scores_dict:
                    save_score[k] = float(scores_dict[k][i])
                if isnewcluster:
                    self._update_memory([smile], [cluster], [save_score], [fingerprint])
                else:
                    self._update_memory([smile], [cluster], [save_score])
                if len(self[cluster]) > self.nbmax:
                    scores[i] = 0

        return scores

    def findCluster(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold:
                cluster = Chem.MolToSmiles(scaffold, isomericSmiles=False)
            else:
                return "", "", False
        else:
            return "", "", False

        fp = Pairs.GetAtomPairFingerprint(scaffold)
        if cluster in self.getFingerprints():
            return cluster, fp, False

        fps = list(self.getFingerprints().values())
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
        if len(sims) == 0:
            return cluster, fp, True
        closest = np.argmax(sims)
        if sims[closest] >= self.minsimilarity:
            return list(self.getFingerprints().keys())[closest], fp, False
        else:
            return cluster, fp, True


class NoScaffoldFilter(ScaffoldFilter):
    """Don't penalize compounds. Only save them with more than 'minscore'. 'minsimilarity' is ignored."""
    def __init__(self, minscore=0.6, minsimilarity=0.6):
        super().__init__(minscore=minscore)

    def score(self, smiles, scores_dict: dict) -> np.array:
        """
        we only log the compounds
        """
        scores = scores_dict.pop("total_score")
        if not self.validScores(smiles, scores): return scores

        for i, smile in enumerate(smiles):
            score = scores[i]
            try:
                scaffold = self._scaffoldfunc(smile)
            except Exception:
                scaffold = ''
            if score >= self.minscore:
                save_score = {"total_score": float(score)}
                for k in scores_dict:
                    save_score[k] = float(scores_dict[k][i])
                self._update_memory([smile], [scaffold], [save_score])
        return scores
