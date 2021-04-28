import pathlib

from sklearn.model_selection import train_test_split
from transformers import TFBertForMaskedLM, BertConfig

from data import *
from paths import *
from utils import *

LARGE_DATASET = True

# vocabulary
DNA_TOKS = ['a','c','g','t']
SPECIAL_TOKS = ['[cls]', '[mask]', '[sep]', '[pad]']
VOCAB = SPECIAL_TOKS + DNA_TOKS

MAX_SEQ_LEN = 510
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 100
EPOCHS = 20
NUM_HIDDEN_LAYERS = 6
NUM_ATTENTION_HEADS = 4
HIDDEN_SIZE = 384
INTERMEDIATE_SIZE = 1536


if LARGE_DATASET == True:
	# extract files from archive
	pathlib.Path(PATH_TO_GENOMES).mkdir(parents=True, exist_ok=True)
	extract_tar_file(PATH_TO_TAR, PATH_TO_GENOMES)
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

# Create TF datasets (train-val-test split: 60-20-20)
train_ids, test_ids = train_test_split(all_ids, test_size=0.2, random_state=1)
train_ids, val_ids = train_test_split(train_ids, test_size = 0.25, random_state=1)
print('train:', train_ids.shape, 'val:', val_ids.shape, 'test:', test_ids.shape)

train_ds = create_dataset(train_ids, SHUFFLE_BUFFER_SIZE, BATCH_SIZE)
val_ds = create_dataset(val_ids, SHUFFLE_BUFFER_SIZE, BATCH_SIZE)
test_ds = create_dataset(test_ids, SHUFFLE_BUFFER_SIZE, BATCH_SIZE)

# Example with one batch
for inputs_batch, labels_batch in train_ds.take(1):
	print(inputs_batch)
	print(labels_batch)

# Import model
#DEVICE = "/gpu:0"

# divide config params by 2, to run model quickly
config = BertConfig(vocab_size=len(VOCAB), num_hidden_layers=NUM_HIDDEN_LAYERS, num_attention_heads=NUM_ATTENTION_HEADS,
                    hidden_size=HIDDEN_SIZE, intermediate_size=INTERMEDIATE_SIZE)
model = TFBertForMaskedLM(config)

configuration = model.config
print(configuration)