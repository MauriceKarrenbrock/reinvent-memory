# coding=utf-8

from typing import List

import numpy as np
import openeye.oechem as oechem
import openeye.oedocking as oedocking
import openeye.oeomega as oeomega

import utils


class _Docking(object):
    """Docking superclass
    
    Methods
    --------
    make_docking(smile)
        takes a string (SMILE) and will return a docking score
        must be implemented by the sub-classes
    __call__(smiles)
        given a list of smiles strings will return a numpy array of scores
    """

    def __init__(self, receptor: utils.FilePath):
        self.receptor_file = receptor

    def __call__(self, smiles: List[str]) -> dict:
        score = np.full(len(smiles), 0, dtype=np.float32)
        for idx, smi in enumerate(smiles):
            score[idx] = self.make_docking(smi)
        return {"total_score": np.array(score, dtype=np.float32)}

    def make_docking(self, smile):
        pass

class DockingOedocking(_Docking):
    """Scores based on the omega docking."""
    def __init__(self, receptor: utils.FilePath):
        super().__init__(receptor)
        omegaOpts = oeomega.OEOmegaOptions()
        omegaOpts.SetStrictStereo(False)
        self.omega = oeomega.OEOmega(omegaOpts)
        oechem.OEThrow.SetLevel(10000)
        oereceptor = oechem.OEGraphMol()
        oedocking.OEReadReceptorFile(oereceptor, self.receptor_file)
        self.dock = oedocking.OEDock()
        self.dock.Initialize(oereceptor)

    def make_docking(self, smile):
        mol = oechem.OEMol()
        if not oechem.OESmilesToMol(mol, smile):
            return 0.0
        if self.omega(mol):
            dockedMol = oechem.OEGraphMol()
            self.dock.DockMultiConformerMolecule(dockedMol, mol)
            score = dockedMol.GetEnergy()
            score = max(0.0, -(score + 8) / 10)
            return score

    def __reduce__(self):
        """
        :return: A tuple with the constructor and its arguments. Used to reinitialize the object for pickling
        """
        return DockingOedocking, (self.receptor_file,)


class DockingPyscreen(_Docking):

    def make_docking(self, smile):
        raise NotImplementedError('Work in progress')

    def __reduce__(self):
        """
        :return: A tuple with the constructor and its arguments. Used to reinitialize the object for pickling
        """
        return DockingPyscreen, (self.receptor_file,)