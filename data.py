import gzip
import numpy as np
import os
import pathlib
import tarfile
import tensorflow as tf



def convert_id_to_tok(id:int, ids_to_toks:dict):
    '''Converts id (int) to token (str) using vocab'''
    return ids_to_toks[id]


def convert_tok_to_id(token:str, toks_to_ids:dict):
    '''Converts token (str) to id (int) using vocab'''
    return toks_to_ids[token]


def create_dataset(ids, shuffle_buffer_size, batch_size):
    '''Create TF dataset in BERT format from input ids'''

    masked_ids, labels = mask_mlm(ids)
    type_ids = get_type_ids(ids)
    attention_mask = get_attention_mask(ids)

    dataset = tf.data.Dataset.from_tensor_slices((masked_ids, attention_mask, type_ids, labels))
    dataset = dataset.map(map_example_to_dict)

    # Shuffle, batch, cache and prefetch
    dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def cut_genome(genome_strs, max_len):
	'''Cut genome string(s) in chunks of size MAX_SEQ_LEN'''

	all_chunks = []

	# split genome sequences into chunks of size MAX_SEQ_LEN
	# (or less if no letter left)
	for seq in genome_strs:
		genome_chunks = [seq[i:i + max_len] for i in range(0, len(seq), max_len)]
	all_chunks.append(genome_chunks)

	# remove last chunk if shorter than expected size
	if len(all_chunks[-1][-1]) < max_len:
		del (all_chunks[-1][-1])

	# flatten chunk list
	all_chunks = [chunk for genome in all_chunks for chunk in genome]

	return all_chunks


def extract_tar_file(path_to_tar:str, path_to_untar_files:str):
	'''Extract files from tar file to genome folder.
        Takes 1 min for a 2GB tar file.'''
	tar = tarfile.open(path_to_tar)
	tar.extractall(path=path_to_untar_files)
	tar.close()


def get_attention_mask(X:np.ndarray):

    '''Indicate to the model which tokens should be attended to, and which should not.
      1 indicates a value that should be attended to, and 0 indicates a padded value.'''

    # no padding, so attention mask is only ones
    attention_mask = np.ones(X.shape, dtype=int)
    return attention_mask


def get_type_ids(X:np.ndarray):

    '''Delimits sequence A from sequence B for inputs of pairs of sequences.
       0 indicates a seq A token, 1 indicates a seq B token.'''

    # only single sequences, so token_type_ids is only zeros
    token_type_ids = np.zeros(X.shape, dtype=int)
    return token_type_ids


def load_sequence(path_to_file):
    '''Read gz or fasta file and save genome sequences and names separately.'''

    if path_to_file.endswith((".gz")):
        with gzip.open(path_to_file, 'rt') as infile:
            sequences, names = read_fasta_file(infile)

    else:
        with open(path_to_file, 'r') as infile:
            sequences, names = read_fasta_file(infile)

    return sequences, names


def map_ids_to_toks(vocab:list):
  '''Maps ids to vocab tokens'''
  return dict((i,j) for i,j in enumerate(vocab))


def map_toks_to_ids(vocab:list):
  '''Maps vocab tokens to ids'''
  return dict((j,i) for i,j in enumerate(vocab))


def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
	return {"input_ids": input_ids,
			"token_type_ids": token_type_ids,
			"attention_mask": attention_masks,}, label


def mask_mlm(X:np.ndarray, prob_mask=0.15, mask_token=1):

    '''Mask 15% of tokens at random for BERT masked language modeling, and set labels
      to -100 for visible tokens. Visible tokens are ignored when computing the loss.
      The loss will only be computed for masked tokens, which have labels in [0, ..., vocab_size]'''

    # original:
    # Prepare masked tokens inputs/labels for masked language modeling:
    # 80% MASK, 10% random, 10% original.
    # see: https://github.com/huggingface/transformers/blob/master/src/transformers/data/data_collator.py#L70
    # in data collator for language modeling, def mask_tokens

    # mask 15% of tokens at random
    inp_mask = np.random.rand(*X.shape)<0.15

    # set labels to -100 for visible tokens (ignored during loss calculation)
    labels =  -100 * np.ones(X.shape, dtype=int)
    labels[inp_mask] = X[inp_mask]
    X[inp_mask] = mask_token

    return X, labels


def process_seq_chunks(all_chunks: list, toks_to_ids: dict):
	'''Build model inputs from a sequence by tokenizing and adding special tokens.
	A single BERT sequence has the following format: ``[CLS] X [SEP]``.
	'''

	all_ids = []

	# for chunk in all_chunks: #comment while debugging
	for i, chunk in enumerate(all_chunks):
		try:
			# tokenize
			tokens = tokenize(chunk)
			ids = []

			# add [cls] tok at the beginning of sequence
			ids.append(toks_to_ids['[cls]'])

			# convert toks to ids
			for tok in tokens:
				ids.append(convert_tok_to_id(tok, toks_to_ids))

			# add [sep] tok at the end of sequence
			ids.append(toks_to_ids['[sep]'])

			all_ids.append(ids)

		except:
			# chunks with out-of-vocab letters
			print(f'chunk {i} OOV')
			pass

	all_ids = np.array([np.array(ids) for ids in all_ids])
	return all_ids


def process_data_from_fasta(PATH_TO_FASTA, MAX_SEQ_LEN):

	genome_str, genome_names = load_sequence(PATH_TO_FASTA)
	genome_chunks = cut_genome(genome_str, MAX_SEQ_LEN)

	return genome_chunks #all_chunks


def process_data_from_tar(PATH_TO_GENOMES, PATH_TO_TAR, MAX_SEQ_LEN):
	# extract files from archive
	pathlib.Path(PATH_TO_GENOMES).mkdir(parents=True, exist_ok=True)
	extract_tar_file(PATH_TO_TAR, PATH_TO_GENOMES)

	# make sequence chunks fr mmultiple genomes
	all_genomes_chunks, all_genomes_names = [], []

	for root, dirs, files in os.walk(PATH_TO_GENOMES):
		for file in files:
			if file.endswith((".gz")):
				try:
					file_path = os.path.join(root, file)
					genome_str, genome_name = load_sequence(file_path)
					genome_chunks = cut_genome(genome_str, MAX_SEQ_LEN)

					all_genomes_chunks.append(genome_chunks)
					all_genomes_names.append(genome_name)
				except:
					pass

	print('Processed', len(all_genomes_chunks), 'genomes')
	all_chunks = [chunk for genome in all_genomes_chunks for chunk in genome]

	return all_chunks


def read_fasta_file(infile):
	'''Save genome names and sequences from opened fasta file.'''

	genome_strs, genome_names = [], []
	sequence_str = ''

	for line_idx, line in enumerate(infile):
		if line.startswith('>'):  # label line
			if sequence_str != '':
				genome_strs.append(sequence_str)
			sequence_str = ''

			line = line[1:].strip()
			genome_names.append(line)

		else:  # sequence line
			line = line.strip()
			sequence_str += line

	genome_strs.append(sequence_str)
	return genome_strs, genome_names


def tokenize(chunk:str):
	'''Tokenize strings'''
	chunk = chunk.lower()
	tokens = []
	for letter in chunk:
		tokens.append(letter)
	return tokens


