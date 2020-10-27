# ECIF
Extended Connectivity Interaction Features

CONTENTS

Notebooks

    01_ECIF_Calculation / ecif.py: Functions for atom definition and ECIF calculation

    02_Examples(Descriptors): How to compute ECIF, ELEMENTS and RDKit descriptors

    03_Examples(ModelTraining): Machine-learning algorithms used to derive the models

    04_Scoring: Use of the best obtained models for re-scoring of protein-ligand complexes
   
Folders

    Descriptors: csv files with ECIF, ELEMENTS and RDKit descriptors for all the protein-ligand complexes employed

    Example_Structures: five pairs of protein-ligand complexes
    
    RawData(Results): csv files needed to reproduce all results shown in the manuscript
    
Notes:

    1. Receptor PDB files are assumed to contain coordinates for all heavy atoms
    
    2. To avoid errors and ensure correct evaluation of the protein-ligand complexes, it is recommended to prepare the ligands using X-tool and ChemAxon's Standardizer
