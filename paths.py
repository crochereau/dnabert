
import os
# local

#PATH_TO_FASTA = '../../Documents/Wang lab/dnabert/ecoli_genome.fasta'
#PATH_TO_GENOMES = 'genomes'
#PATH_TO_TAR = '../../Documents/Wang lab/dnabert/genome_assemblies_3_files.tar'

#CHECKPOINTS_DIR = 'checkpoints'
#LOGS_DIR = 'logs'

# server

DATA_PATH = '../../data/dnabert/'
PATH_TO_FASTA = os.path.join(DATA_PATH, 'ecoli_genome.fasta')
PATH_TO_GENOMES = os.path.join(DATA_PATH, 'genomes')
PATH_TO_TAR = os.path.join(DATA_PATH, 'genome_assemblies_100_files.tar')

CHECKPOINTS_DIR = os.path.join(DATA_PATH, 'checkpoints')
LOGS_DIR = os.path.join(DATA_PATH, 'logs')

