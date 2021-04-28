from argparsers import data_parser
import numpy as np

from paths import *

# vocabulary
DNA_TOKS = ['a','c','g','t']
SPECIAL_TOKS = ['[cls]', '[mask]', '[sep]', '[pad]']
VOCAB = SPECIAL_TOKS + DNA_TOKS

MAX_SEQ_LEN = 510

args = data_parser.parse_args()

if args.large_dataset == True:
	# extract files from archive
	pathlib.Path(PATH_TO_GENOMES).mkdir(parents=True, exist_ok=True)
	# make genome chunks from multiple genomes
	all_chunks = process_data_from_tar(PATH_TO_GENOMES, PATH_TO_TAR, MAX_SEQ_LEN)
else:
	# make genome chunks from single genome
	all_chunks = process_data_from_fasta(PATH_TO_FASTA, MAX_SEQ_LEN)

# check that chunks have expected size
for chunk in all_chunks:
	assert len(chunk) == MAX_SEQ_LEN

dataset_size = len(all_chunks)*MAX_SEQ_LEN
print('{:,}'.format(len(all_chunks)), 'chunks ready')
print('size of dataset:', '{:,}'.format(dataset_size), 'letters')

# Tokenize and vectorize sequences
toks_to_ids = map_toks_to_ids(VOCAB)
ids_to_toks = map_ids_to_toks(VOCAB)
print(toks_to_ids)
all_ids = process_seq_chunks(all_chunks=all_chunks, toks_to_ids=toks_to_ids)
print(all_ids.shape)

# Save vectorized sequences to np.array
if args.large_dataset == True:
	path_to_ids = os.path.join(DATA_PATH, 'ids_large')
else:
	path_to_ids = os.path.join(DATA_PATH, 'ids_small')

np.save(path_to_ids, all_ids)
