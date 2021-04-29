import argparse


data_parser = argparse.ArgumentParser()

data_parser.add_argument(
	"--which_dataset",
	type=str,
	default='small',
	required=True,
	help="Which dataset to use. Can be 'small' or 'large'."
)


train_parser = argparse.ArgumentParser()

train_parser.add_argument(
	"--which_dataset",
	type=str,
	default='small',
	required=True,
	help="Which dataset to use. Can be 'small' or 'large'."
)

train_parser.add_argument(
	"--exp_name",
	type=str,
	required=True,
	help="Name of experiment."
)

train_parser.add_argument(
	"--gpu",
	type=int,
	help='GPU device to use')

train_parser.add_argument(
	"--batch_size",
	type=int,
	default=256,
	required=False,
	help='Batch size.')

train_parser.add_argument(
	"--epochs",
	type=int,
	default=20,
	required=False,
	help="Number of epochs.")

train_parser.add_argument(
	"--learning_rate",
	type = float,
	default=5e-5,
	required=False,
	help="Learning rate.")

train_parser.add_argument(
	"--n_hidden_layers",
	type=int,
	default=6,
	required=False,
	help="Number of hidden layers.",
	)

train_parser.add_argument(
	"--n_attention_heads",
	type=int,
	default=4,
	required=False,
	help="Number of attention heads.",
	)

train_parser.add_argument(
	"--hidden_size",
	type=int,
	default=384,
	required=False,
	help="bert hidden size.",
	)

train_parser.add_argument(
	"--intermediate_size",
	type=int,
	default=1536,
	required=False,
	help="Bert intermediate size.",
	)

