This repo contains code for pre-training a Bert model for DNA sequences.
The datasets used are fasta files downloaded from NCBI Assembly.
Models were trained on *Escherichia coli* complete genomes.

To preprocess the data, add the path to your directory in ```paths.py```, then run the following command: 
<br> ```python preprocess_data.py```.

To pre-train your model, run ```run_pretrain.py```. You can specify the experiment's name and the model's hyperparameters by passing arguments defined in ```argparser.py```.
For instance, to specify the experiment's name and the batch size, use the following command:
<br>```python run_pretrain.py --exp_name exp1 --batch_size 32```.
