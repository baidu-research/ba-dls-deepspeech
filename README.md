# ba-dls-deepspeech
Train your own CTC model!

# Table of Contents
1. [Dependencies](#dependencies)
2. [Data](#data)
3. [Running an example](#running-an-example)

## Dependencies
You will need the following packages installed before you can train a model using this code. You may have to change `PYTHONPATH` to include the directories
of your new packages.  
  
**theano**  
The underlying deep learning Python library. We suggest downloading the latest release from https://github.com/Theano/Theano/releases.    
```bash
$tar xf <downloaded_tar_file>
$cd theano-*
$python setup.py install --user
```  

**keras**  
This is a wrapper over Theano that provides nice functions for building networks. Download the latest release from https://github.com/fchollet/keras/releases  
Make sure you install it with support for `hdf5` - we make use of that to save models.  
```bash
$tar xf <downloaded_tar_file>
$cd keras-*
$python setup.py install --user
```  

**lasagne**  
```
$pip install lasagne <--user>
```

**warp-ctc**  
This contains the main implementation of the CTC cost function.  
<code>git clone https://github.com/baidu-research/warp-ctc</code>  
To install it, follow the instructions on https://github.com/baidu-research/warp-ctc

**theano-warp-ctc**  
This is a theano wrapper over warp-ctc.  
<code>git clone https://github.com/sherjilozair/ctc</code>  
Follow the instructions on https://github.com/sherjilozair/ctc for installation.

**Others**  
You may require some additional packages. Install Python requirements through `pip` as:  
<code>pip install soundfile</code>  
On Ubuntu, `avconv` (used here for audio format conversions) requires `libav-tools`.  
<code>sudo apt-get install libav-tools</code>  
## Data
We will make use of the LibriSpeech ASR corpus to train our models. Use the `download.sh` script to download this corpus (~65GB). Use `flac_to_wav.sh` to convert any `flac` files to `wav`.  
We make use of a JSON file that aggregates all data for training, validation and testing. Once you have a corpus, create a description file that is a json-line file in the following format:
<pre>
{"duration": 15.685, "text": "spoken text label", "key": "/home/username/LibriSpeech/train-clean-360/5672/88367/5672-88367-0031.wav"}
{"duration": 14.32, "text": "ground truth text", "key": "/home/username/LibriSpeech/train-other-500/8678/280914/8678-280914-0009.wav"}
</pre>  
You can create such a file using `create_desc_file.py`. Each line is a JSON. We will make use of the durations to construct a curriculum in the first epoch (shorter utterances are easier).  
You can query the duration of a file using: <code>soxi -D filename</code>.
## Running an example
Finally, let's train a model!  
<code>python train.py train_corpus.json validation_corpus.json ./save_my_model_here</code>  
This will checkpoint a model every few iterations into the directory you specify.
