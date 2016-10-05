# Use this script to download the entire LibriSpeech dataset

#!/bin/bash

base="http://www.openslr.org/resources/12/"



for s in 'dev-clean' 'dev-other' 'test-clean' 'test-other' 'train-clean-100' 'train-clean-360' 'train-other-500'
do
    linkname="${base}/${s}.tar.gz"
    echo $linkname
    wget -c $linkname
done

for s in 'dev-clean' 'dev-other' 'test-clean' 'test-other' 'train-clean-100' 'train-clean-360' 'train-other-500'
do
    tar -xzvf $s.tar.gz
done

