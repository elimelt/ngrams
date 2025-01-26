#!/bin/bash

mkdir -p data && pushd data

wget https://www.anc.org/MASC/download/masc_500k_texts.tgz

tar -xf masc_500k_texts.tgz

rm masc_500k_texts.tgz

find masc_500k_texts -type f -name "*.txt" -exec cat {} + > combined.txt

rm -rf masc_500k_texts