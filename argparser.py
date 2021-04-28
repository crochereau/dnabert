import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
	"--large_dataset",
	default=False,
	type=bool,
	required=False,
	help="Large or small training dataset."
)

parser.add_argument(
	"--exp_name",
	type=str,
	required=True,
	help="Name of experiment."
)

parser.add_argument(
	"--batch_size",
	type=int,
	default=256,
	required=False,
	help='Batch size.')

parser.add_argument(
	"--epochs",
	type=int,
	default=20,
	required=False,
	help="Number of epochs.")

parser.add_argument(
	"--learning_rate",
	type = float,
	default=5e-5,
	required=False,
	help="Learning rate.")

parser.add_argument(
	"--output_dir",
	type=str,
	required=False,
	help="The output directory where the model predictions and checkpoints will be written.",
	)

parser.add_argument(
	"--n_hidden_layers",
	type=int,
	default=6,
	required=False,
	help="Number of hidden layers.",
	)

parser.add_argument(
	"--n_attention_heads",
	type=int,
	default=4,
	required=False,
	help="Number of attention heads.",
	)

parser.add_argument(
	"--hidden_size",
	type=int,
	default=384,
	required=False,
	help="bert hidden size.",
	)

parser.add_argument(
	"--intermediate_size",
	type=int,
	default=1536,
	required=False,
	help="Bert intermediate size.",
	)

