# CS22S027_DL_Assignment_3
Implementation of CS6910: Deep Learning Assignmnet 3, Jan-May 2024


### Details
Name: Ishaan Taneja </br>
Roll No.: CS22S027 </br>
</br>

This project is an implementation of a neural network from scratch using Python. It is designed to be flexible, allowing adjustments to various parameters such as the dimension of the input character embeddings, the hidden states of the encoders and decoders, the cell (RNN, LSTM, GRU), the number of layers in the encoder and decoder, batch size, epochs, etc. can be changed.


### Dependencies
 - python
 - numpy library
 - wandb library
 - torch library
 - tqdm library (for fast, extensible progress bar for loops and iterations in python)
 - matplotlib (Optional: if you want to plot heatmaps)

To download all the necessary dependencies, you can run: `pip install -r requirements.txt`


### Clone and Download Instructions
Clone the repository or download the project files. Ensure that python and other required packages are installed in the project directory.</br>
To clone the repository directly to you local machine, ensure git is installed, run the command: 
</br>
`git clone https://github.com/IT527/CS22S027-DL_Assignment_3.git`
</br>
</br>
Alternatively, you can download the entire repository as a .zip file from the Download ZIP option provided by github.


### Usage
To run the python script, navigate to the project directory and run: `python train.py [OPTIONS]`
</br>
The 'OPTIONS' can take different values for parameters to select dataset, modify network architecture, select activation function and many more.</br>
The possible arguments and respective values for 'OPTIONS' are shown in the table below:</br>

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | 'CS6910_Assignment_3' | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | cs22s027  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-ehd`, `--encoder_hidden_dimension` | 256| Encoder hidden dimension size |
| `-dhd`, `--decoder_hidden_dimension` | 256 |  Decoder hidden dimension size|
| `-eed`, `--encoder_embed_dimension` | 256 | Encoder embedding dimension size | 
| `-ded`, `--decoder_embed_dimension` | 256 | Decoder embedding dimension size |
| `-b`, `--batch_size` | 128 | Batch size used to train the model | 
| `-bd`, `--bidirectional` | 'True' | choices=['True','False'] | 
| `-enl`, `--encoder_num_layers` | 2 | Number of layers in the encoder |
| `-dnl`, `--decoder_num_layers` | 3 | Number of layers in the decoder | 
| `-ct`, `--cell_type` | 'lstm' | choices=['lstm', 'gru', 'rnn'] | 
| `dp`, `--dropout` | 0 | Dropout rate |
| `-bw`, `--beam_width` | 1 | Beam width for beam search |
| `-e`, `--epochs` | 15 | Number of training epochs |
| `-bm`, `--beam` | 'True' | choices=['True','False'] | 
| `-d`, `--device` | 'cuda' | if torch.cuda.is_available() else 'cpu', help='Device to use for training | 
| `-a`, `--attention` | 'False' | ['True','False'] |


An example run with cell type "rnn" and number of epochs as 10: `python train.py --epochs 10 --cell_type 'rnn'`

</br>

On execution of the file as shown above, loss and accuracies for the train, validation and test dataset will be printed on the terminal. Along with it, the plots highlighting the loss and accuracies for each epochs, for both train and validation dataset, will be logged onto the wandb project. At the end, it will also print the test loss and test accuracy by evaluating model on test dataset.</br>
To access plots in wandb, ensure to replace the given key with your wandb API key.</br>
Look for line 22 in train.py file and enter your API key in the key variable.


### Additional Resources and help
Included in the project is DL_Assignment_3.ipynb, compatible with Jupyter Notebook or Google Colab. It encompasses all the classes/functions, sweep operations, and logging utilities like heatmaps. For tailored runs, you may need to adjust configurations and uncomment sections in the notebook to log specific metrics or plots. The notebook serves as a practical reference for understanding the project's workflow. </br>
All the plots are generated and logged to wandb using this file only, while for a new configuration one can run the train.py file as shown above.
</br>
</br>
The sweep details for choosing the hyperparameters, runs, sample images, and related plots can be viewed at: ``


