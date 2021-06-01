# coding=utf-8

from typing import List
from pathlib import Path
import math
import os
import shutil
import subprocess
from itertools import takewhile

import numpy as np
import openeye.oechem as oechem
import openeye.oedocking as oedocking
import openeye.oeomega as oeomega
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
        score = np.empty(len(smiles), dtype=np.float32)
        for idx, smi in enumerate(smiles):
            try:
                score[idx] = self.make_docking(smi)
            except Exception:
                score[idx] = 0
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


class DockingQvina(_Docking):
    """Docking with qvina

    Extra parameters

    --receptor
        the path to the receptor file(s) comma separated list if more than one
    --box_center=(8.882, 6.272, -9.486)
        the center of the docking box
    --box_size=(10,10,10)
        the x y z radii of the box in Angstrom comma separated values (default 10,10,10)
    --qvina_path
        path to the qvina executable, default will search in PATH
    --all_smiles_csv
        a csv file where all the smiles and docking scores are saved default='all_smiles.csv'
    --ADFRsuite_bin
        the path to the adrfsuite bin default ~/ADFRsuite-1.0/bin
    --output_path
        where to save all the temporary docking files default ./docking_stuff
    --num_cpu
        the number of cpu that qvina shall use, default os.cpu_count()
    --max_conformers
        default 4
    """
    def __init__(self,
    receptor,
    box_center=[8.882, 6.272, -9.486],
    box_size=[10,10,10],
    output_path='./docking_stuff',
    qvina_path=None,
    all_smiles_csv='all_smiles.csv',
    ADFRsuite_bin='~/ADFRsuite-1.0/bin',
    num_cpu=os.cpu_count(),
    max_conformers=4):

        super().__init__(receptor)

        # I want to have a list of Path
        if isinstance(self.receptor_file, str):
            self.receptor_file = self.receptor_file.split(',')
        elif not hasattr(self.receptor_file, '__iter__'):
            self.receptor_file = [self.receptor_file]

        for i, j in enumerate(self.receptor_file):
            self.receptor_file[i] = Path(j).expanduser().resolve()
        
        if isinstance(box_center, str):
            box_center = box_center.split(',')
            for i in range(len((box_center))):
                box_center[i] = float(box_center[i])
        if len(box_center) != 3:
            raise ValueError('Box center needs the x y z values')
        self.box_center = box_center

        if isinstance(box_size, str):
            box_size = box_size.split(',')
            for i in range(len((box_size))):
                box_size[i]=float(box_size[i])
        if len(box_size) != 3:
            raise ValueError('Box size needs the x y z values')
        self.box_size = box_size

        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True, parents=True)

        if qvina_path is None:
            qvina_path = shutil.which('qvina')
            if qvina_path is None:
                raise RuntimeError('qvina is not in $PATH')
        self.qvina_path = Path(qvina_path).expanduser().resolve()

        self.ADFRsuite_bin = Path(ADFRsuite_bin).expanduser().resolve()

        self.num_cpu = int(num_cpu)

        self.max_conformers = int(max_conformers)

        # A CSV file where I will save all the smiles
        # and the docking scores
        self.all_smiles_csv = Path(all_smiles_csv)
        with self.all_smiles_csv.open('w') as f:
            f.write('SMILES,DockingScore\n')


        self.qvina_log_file = self.output_path / 'tmp_qvina_log.log'

        # Prepare the receptor(s) for docking
        self.prepare_receptors()

    def get_conformers(self, smi):
        
        mol = Chem.MolFromSmiles(smi)
        Chem.SanitizeMol(mol)
        mol = Chem.AddHs(mol)
        mol.SetProp("_Name",'Ligand')

        cids = AllChem.EmbedMultipleConfs(mol, self.max_conformers, AllChem.ETKDG())
        output_pdb = []
        for i, conf in enumerate(cids):
            output_pdb.append(self.output_path / f'conformer{i}.pdb')
            pdbwriter = Chem.PDBWriter(str(output_pdb[-1]))
            try:
                AllChem.UFFOptimizeMolecule(mol,confId=conf)
            except Exception:
                pass
            pdbwriter.write(mol, conf)

        return output_pdb

    @staticmethod
    def score_transformation(score, a=0.3427775704, b=2.944438979):#a=0.54931, b=2.19722):
        """Inverted sigmoid

        it gets to 1 for X going to -inf and to 0 for
        X going to +inf
        With the defaut values it will be 0.5 for x=-4 and 0.9 fo x=-8

        1/(1+exp(a*X + b))
        """
        return 1/(1+math.exp(a*score + b))

    def prepare_receptors(self):
        for i, receptor in enumerate(self.receptor_file):
            commands = [str(self.ADFRsuite_bin / 'prepare_receptor'),
            '-r', str(receptor.name),
            '-o', str(receptor.with_suffix('.pdbqt').name),
            '-A', 'bonds_hydrogens']

            # As a default will try to repair enything possible
            # if it fails won't repair anything
            try:
                self.use_subprocess(commands,
                    cwd=str(receptor.parent),
                    shell=False,
                    error_string='error during receptor preparation')
            except RuntimeError:
                commands.pop(-1)
                commands.pop(-1)

                self.use_subprocess(commands,
                    cwd=str(receptor.parent),
                    shell=False,
                    error_string='error during receptor preparation')
            
            self.receptor_file[i] = receptor.with_suffix('.pdbqt')

    def prepare_ligands_from_files(self, files):

        output_pdbqt = []

        for ligand in files:
            ligand = ligand.expanduser().resolve()
            commands = [str(self.ADFRsuite_bin / 'prepare_ligand'),
            '-l', str(ligand.name),
            '-o', str(ligand.with_suffix('.pdbqt').name),
            '-A', 'bonds_hydrogens']

            # As a default will try to repair enything possible
            # if it fails won't repair anything
            try:
                self.use_subprocess(commands,
                    cwd=str(ligand.parent),
                shell=False,
                error_string=f'error during ligand preparation\n{str(ligand)}')
            except RuntimeError:
                commands.pop(-1)
                commands.pop(-1)

                self.use_subprocess(commands,
                    cwd=str(ligand.parent),
                    shell=False,
                    error_string=f'error during ligand preparation\n{str(ligand)}')
            
            output_pdbqt.append(ligand.with_suffix('.pdbqt'))

        return output_pdbqt

    def parse_qvina_log_file(self, score_mode='best'):
        """Parse the log file generated from a run of Vina-type docking software
        and return the appropriate score.
        Parameters
        ----------
        score_mode : str (Default = 'best')
            The method used to calculate the docking score from the log file.
            See also pyscreener.utils.calc_score for more details
        Returns
        -------
        score : Optional[float]
            the parsed score given the input scoring mode or None if the log
            file was unparsable 
        """
        # This code comes from pyscreener and is under the 
        # MIT license https://github.com/coleygroup/pyscreener


        # HELPER FUNCTION
        #-------------------------------------------------------------
        def calc_score(scores, score_mode='best'):
            """Calculate an overall score from a sequence of scores
            Parameters
            ----------
            scores : Sequence[float]
            score_mode : str, default='best'
                the method used to calculate the overall score. Choices include:
                * 'best': return the top score
                * 'avg': return the average of the scores
                * 'boltzmann': return the boltzmann average of the scores
            Returns
            -------
            score : float
            """
            scores = sorted(scores)
            if score_mode in ('best', 'top'):
                score = scores[0]
            elif score_mode in ('avg', 'mean'):
                score = sum(score for score in scores) / len(scores)
            elif score_mode == 'boltzmann':
                Z = sum(exp(-score) for score in scores)
                score = sum(score * exp(-score) / Z for score in scores)
            else:
                score = scores[0]
                
            return score
        #-----------------------------------------------------------------------

        # vina-type log files have scoring information between this table border
        # and the line containing "Writing output ... done."
        TABLE_BORDER = '-----+------------+----------+----------'

        with self.qvina_log_file.open('r') as fid:
            for line in fid:
                if TABLE_BORDER in line:
                    break

            score_lines = takewhile(lambda line: 'Writing' not in line, fid)
            scores = [float(line.split()[1]) for line in score_lines]

        if len(scores) == 0:
            return None

        return calc_score(scores, score_mode)


    def execute_qvina(self, ligand):

        scores = []
        for receptor in self.receptor_file:
            commands = [
            str(self.qvina_path),
            f'--receptor={receptor}',
            f'--ligand={ligand}',
            f'--center_x={self.box_center[0]}',
            f'--center_y={self.box_center[1]}',
            f'--center_z={self.box_center[2]}',
            f'--size_x={self.box_size[0]}',
            f'--size_y={self.box_size[1]}',
            f'--size_z={self.box_size[2]}',
            f'--cpu={self.num_cpu}',
            f'--out={self.output_path / "tmp_docked.pdbqt"}',
            f'--log={self.qvina_log_file}'
            ]

            try:
                self.use_subprocess(commands,
                    shell=False,
                    error_string='docking was not succesful')
            
            except RuntimeError:
                print(f'\n\nDocking of {ligand} on {receptor} was not succesful\n\n')

                scores.append(None)
            
            else:
                scores.append(self.parse_qvina_log_file())

        return scores

    def make_docking(self, smi):
        #check if it is a valid SMILES
        if Chem.MolFromSmiles(smi) is None:
            return 0.0

        conformers = self.get_conformers(smi)
        conformers = self.prepare_ligands_from_files(conformers)
        best_value = float('inf')

        for conformer in conformers:
            docking_output = self.execute_qvina(conformer)

            for receptor in docking_output:
                if receptor is not None:
                    best_value = min(best_value, receptor)


        if best_value == float('inf'): #all docking failed
            with self.all_smiles_csv.open('a') as f:
                f.write(f'{smi},nan\n')
        
        else:
            with self.all_smiles_csv.open('a') as f:
                f.write(f'{smi},{best_value:.18e}\n')

        return self.score_transformation(best_value)

    @staticmethod
    def use_subprocess(commands, shell=False, error_string='error during the call of an external program', cwd=None):
        
        if cwd is None:
            cwd = str(Path('.').resolve())

        r = subprocess.run(commands,
                        shell=shell,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False,
                        cwd=cwd)

        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            raise RuntimeError(error_string)

    def __reduce__(self):
        """
        :return: A tuple with the constructor and its arguments. Used to reinitialize the object for pickling
        """
        return DockingQvina, (self.receptor_file,
            self.box_center,
            self.box_size,
            self.output_path,
            self.qvina_path,
            self.all_smiles_csv,
            self.ADFRsuite_bin,
            self.num_cpu,
            self.max_conformers)
