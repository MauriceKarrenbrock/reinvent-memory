# coding=utf-8

from typing import List
from pathlib import Path
import math
import os

import numpy as np
import openeye.oechem as oechem
import openeye.oedocking as oedocking
import openeye.oeomega as oeomega
import pyscreener.preprocessing.tautomers as _tautomers
import pyscreener.docking.vina as _vina
from rdkit import Chem
from rdkit.Chem import AllChem

import utils

OMP_NUM_THREADS = int(os.environ.get('OMP_NUM_THREADS', 1))

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


class _DockingVinaPyscreenerSuperclass(_Docking):
    """The superclass to dock with Vina like software using pyscreener
    """
    def __init__(self, receptor, box_center=(8.882, 6.272, -9.486), box_size=(10,10,10), output_path='./docking_stuff', docking_program='qvina',
    all_smiles_csv='all_smiles.csv'):
        super().__init__(receptor)
        #vina class wants a list of strings
        if isinstance(self.receptor_file, str) or (
            not hasattr(self.receptor_file, '__iter__')):
            self.receptor_file = [self.receptor_file]
        for i, j in enumerate(self.receptor_file):
            self.receptor_file[i] = str(j)
        
        self.box_center = box_center
        self.box_size = box_size
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True, parents=True)

        # subclass dependent
        # but in this way you can use this superclass
        # as a jack of all trades class
        self.docking_program = docking_program

        self.vina = _vina.Vina(self.docking_program, center=self.box_center, size=self.box_size,
        path=str(self.output_path.resolve()), receptors=self.receptor_file)

        # A CSV file where I will save all the smiles
        # and the docking scores
        self.all_smiles_csv = Path(all_smiles_csv)
        with self.all_smiles_csv.open('w') as f:
            f.write('SMILES,DockingScore\n')

    def get_conformers(self, smi, max_conformers=4):
        
        mol = Chem.MolFromSmiles(smi)
        Chem.SanitizeMol(mol)
        mol = Chem.AddHs(mol)
        mol.SetProp("_Name",'Ligand')

        cids = AllChem.EmbedMultipleConfs(mol, max_conformers, AllChem.ETKDG())
        output_pdb = self.output_path / f'{smi}.pdb'
        with output_pdb.open('w') as f:
            #pdbwriter = Chem.PDBWriter(f)
            for conf in cids: 
                AllChem.UFFOptimizeMolecule(mol,confId=conf)
                #pdbwriter.write(mol, conf)
            f.write(Chem.MolToPDBBlock(mol))

        return output_pdb

    def score_transformation(score, a=0.54931, b=2.19722):
        """Inverted sigmoid

        it gets to 1 for X going to -inf and to 0 for
        X going to +inf
        With the defaut values it will be 0.5 for x=-4 and 0.9 fo x=-8

        1/(1+exp(a*X + b))
        """
        return 1/(1+math.exp(a*score + b))

    def make_docking(self, smi):
        #check if it is a valid SMILES
        if Chem.MolFromSmiles(smi) is None:
            return 0.0

        conformers = self.get_conformers(smi)
        conformers = self.vina.prepare_from_file(conformers)
        best_value = float('inf')

        for conformer in conformers:
            docking_output = self.vina.dock_ligand(conformer, software=self.docking_program,
                        receptors=self.receptor_file,
                        center=self.box_center,
                        size=self.box_size, ncpu=OMP_NUM_THREADS, 
                        path=str(self.output_path))

            for receptor in docking_output:
                for docking_run in receptor:
                    if docking_run['score'] is not None:
                        best_value = min(best_value, docking_run['score'])

        if best_value == float('inf'): #all docking failed
            with self.all_smiles_csv.open('a') as f:
                f.write(f'{smi},nan\n')
            return 10 #the score resulting from the sigmoid will be 0 (no risk of overflow)

        with self.all_smiles_csv.open('a') as f:
            f.write(f'{smi},{best_value:.18e}\n')

        return self.score_transformation(best_value)

    def __reduce__(self):
        """
        :return: A tuple with the constructor and its arguments. Used to reinitialize the object for pickling
        """
        return _DockingVinaPyscreenerSuperclass, (self.receptor_file,)

class DockingQvina(_DockingVinaPyscreenerSuperclass):
    def __init__(self, receptor, box_center=(8.882, 6.272, -9.486), box_size=(10,10,10), output_path='./docking_stuff', docking_program
            ='qvina',
                all_smiles_csv='all_smiles.csv'):
        super().__init__(receptor, box_center=(8.882, 6.272, -9.486), box_size=(10,10,10), output_path='./docking_stuff', docking_program
                ='qvina',
                    all_smiles_csv='all_smiles.csv')
        self.docking_program = 'qvina'
