# ba-dls-deepspeech
Train your own CTC model!

# Table of Contents
1. [Dependencies](#dependencies)
2. [Data](#data)
3. [Running an example](#running-an-example)

## Dependencies
You will need the following packages installed before you can train a model using this code. You may have to change `PYTHONPATH` to include the directories
of your new packages.  
  
1. **theano**  
The underlying deep learning Python library. We suggest using the bleeding edge version through  
<code>git clone https://github.com/Theano/Theano</code>  
Follow the instructions on: http://deeplearning.net/software/theano/install.html#bleeding-edge-install-instructions  
Or, simply: <code>cd Theano; python setup.py install</code>

2. **keras**  
This is a wrapper over Theano that provides nice functions for building networks. Once again, we suggest using the bleeding edge version.
Make sure you install it with support for `hdf5` - we make use of that to save models.  
<code>git clone https://github.com/fchollet/keras</code>  
Follow the installation instructions on https://github.com/fchollet/keras  
Or, simply: <code>cd keras; python setup.py install</code>

3. **warp-ctc**  
This contains the main implementation of the CTC cost function.  
<code>git clone https://github.com/baidu-research/warp-ctc</code>  
To install it, follow the instructions on https://github.com/baidu-research/warp-ctc

4. **theano-warp-ctc**  
This is a theano wrapper over warp-ctc.  
<code>git clone https://github.com/sherjilozair/ctc</code>  
Follow the instructions on https://github.com/sherjilozair/ctc for installation.

5. **Others**  
You will need other common packages. Install them through `pip` as:  
<code>pip install json soundfile</code>  
## Data
## Running an example
