import logging
import numpy as np
import os
import tensorflow as tf

from datetime import datetime
from sklearn.model_selection import train_test_split
from transformers import TFBertForMaskedLM, BertConfig

from argparsers import train_parser
from data_utils import *
from paths import *
from metrics import *

# vocabulary
DNA_TOKS = ['a','c','g','t']
SPECIAL_TOKS = ['[cls]', '[mask]', '[sep]', '[pad]']
VOCAB = SPECIAL_TOKS + DNA_TOKS

MAX_SEQ_LEN = 510
SHUFFLE_BUFFER_SIZE = 100


logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S"
)

args = train_parser.parse_args()

# Load vectorized sequences
if args.which_dataset == 'large':
	path_to_ids = os.path.join(DATA_PATH, 'ids_large.npy')
elif args.which_dataset == 'small':
	path_to_ids = os.path.join(DATA_PATH, 'ids_small.npy')
all_ids = np.load(path_to_ids)

# Create TF datasets (train-val-test split: 60-20-20)
train_ids, test_ids = train_test_split(all_ids, test_size=0.2, random_state=1)
train_ids, val_ids = train_test_split(train_ids, test_size = 0.25, random_state=1)
print('train:', train_ids.shape, 'val:', val_ids.shape, 'test:', test_ids.shape)

train_ds = create_dataset(train_ids, SHUFFLE_BUFFER_SIZE, args.batch_size)
val_ds = create_dataset(val_ids, SHUFFLE_BUFFER_SIZE, args.batch_size)
test_ds = create_dataset(test_ids, SHUFFLE_BUFFER_SIZE, args.batch_size)

# Example with one batch
for inputs_batch, labels_batch in train_ds.take(1):
	print(inputs_batch)
	print(labels_batch)


# Create checkpoint directory
checkpoint_dir = os.path.join(CHECKPOINTS_DIR, args.exp_name)
checkpoint_path = os.path.join(checkpoint_dir, "cp-{epoch:08d}.ckpt")

print("Checkpoints directory is", checkpoint_dir)
if os.path.exists(checkpoint_dir):
	print("Checkpoints folder already exists")
else:
	print("Creating a checkpoints directory")
	os.makedirs(checkpoint_dir)

# Restore latest checkpoint (if it exists)
latest = tf.train.latest_checkpoint(checkpoint_dir)
if latest != None:
	print("Loading weights from", latest)
	model.load_weights(latest)
else:
	print("Checkpoint not found. Starting from scratch")

# Create log directory
logs = os.path.join(LOGS_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))


# Import model
DEVICE = "/gpu:1"

config = BertConfig(
	vocab_size=len(VOCAB),
	num_hidden_layers=args.n_hidden_layers,
	num_attention_heads=args.n_attention_heads,
	hidden_size=args.hidden_size,
	intermediate_size=args.intermediate_size)

model = TFBertForMaskedLM(config)

configuration = model.config
print(configuration)

# training
lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
	initial_learning_rate=2e-5,
	decay_steps=15,
	end_learning_rate=0)

optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

my_callbacks = [
	tf.keras.callbacks.ModelCheckpoint(
		filepath=checkpoint_path,
		save_weights_only=True,
		monitor='val_accuracy',
		save_best_only=True),

	tf.keras.callbacks.TensorBoard(
		log_dir=logs,
		histogram_freq=1,
		profile_batch='500,520')
]

model.compile(
	optimizer=optimizer,
	loss=masked_sparse_cce,
	metrics=[masked_sparse_ca])

history = model.fit(
	train_ds,
	validation_data=val_ds,
	epochs=args.epochs,
	callbacks=my_callbacks)
