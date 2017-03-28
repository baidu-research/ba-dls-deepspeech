# ba-dls-deepspeech
Train your own CTC model!  This code was released with the lecture from the [Bay Area DL School](http://www.bayareadlschool.org/).  PDF slides are available [here](http://cs.stanford.edu/~acoates/ba_dls_speech2016.pdf).

# Table of Contents
1. [Dependencies](#dependencies)
2. [Data](#data)
3. [Running an example](#running-an-example)

## Dependencies
You will need the following packages installed before you can train a model using this code. You may have to change `PYTHONPATH` to include the directories
of your new packages.  
  
**theano**  
The underlying deep learning Python library. We suggest downloading version 0.8.2 from https://github.com/Theano/Theano/releases.    
```bash
$tar xf <downloaded_tar_file>
$cd theano-*
$python setup.py install --user
```  
or
```
pip install 'theano==0.8.2'
``` 

**keras**  
This is a wrapper over Theano that provides nice functions for building networks. Download version 1.1.2 from https://github.com/fchollet/keras/releases  
Make sure you install it with support for `hdf5` - we make use of that to save models.  
```bash
$tar xf <downloaded_tar_file>
$cd keras-*
$python setup.py install --user
```  
or
```
pip install 'keras==1.1.2'
``` 

Update the keras.json to use Theano backend:

```bash
vim ~/.keras/keras.json
```
Update the backend property
```
"backend": "theano"
```

**lasagne**  
```
$pip install lasagne <--user>
```

**scipy**
Scipy needs to be version 0.18.1
```
pip install 'scipy==0.18.1'
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
We will make use of the LibriSpeech ASR corpus to train our models. While you can start off by using the 'clean' LibriSpeech datasets, you can use the `download.sh` script to download the entire corpus (~65GB).  Use `flac_to_wav.sh` to convert any `flac` files to `wav`.  
We make use of a JSON file that aggregates all data for training, validation and testing. Once you have a corpus, create a description file that is a json-line file in the following format:
<pre>
{"duration": 15.685, "text": "spoken text label", "key": "/home/username/LibriSpeech/train-clean-360/5672/88367/5672-88367-0031.wav"}
{"duration": 14.32, "text": "ground truth text", "key": "/home/username/LibriSpeech/train-other-500/8678/280914/8678-280914-0009.wav"}
</pre>  
You can create such a file using `create_desc_file.py`.  
```bash
$python create_desc_file.py /path/to/LibriSpeech/train-clean-100 train_corpus.json
$python create_desc_file.py /path/to/LibriSpeech/dev-clean validation_corpus.json
$python create_desc_file.py /path/to/LibriSpeech/test-clean test_corpus.json
```
You can query the duration of a file using: <code>soxi -D filename</code>.
## Running an example
**Training**  
Finally, let's train a model!  
```bash
$python train.py train_corpus.json validation_corpus.json /path/to/model
```
This will checkpoint a model every few iterations into the directory you specify. You can monitor how your model is doing, using `plot.py`.
```bash
$python plot.py -d /path/to/model1 /path/to/model2 -s plot.png
```
This will save a plot comparing two models' training and validation performance over iterations. This helps you gauge hyperparameter settings and their effects. Eg: You can change learning rate passed to `compile_train_fn` in `train.py`, and see how that affects training curves.
Note that the model and costs are checkpointed only once in 500 iterations or once every epoch, so it may take a while before you can see updates plots.

**Testing**  
Once you've trained your model for a sufficient number of iterations, you can test its performance on a different dataset:
```bash
$python test.py test_corpus.json train_corpus.json /path/to/model
```
This will output the average loss over the test set, and the predictions compared to their ground truth. We make use of the training corpus here, to compute feature means and variance.

**Visualization/Debugging**  
You can also visualize your model's outputs for an audio clip using:
```bash
$python visualize.py audio_clip.wav train_corpus.json /path/to/model
```
This outputs: `softmax.png` and `softmax.npy`. These will tell you how confident your model is about the ground truth, across all the timesteps.
