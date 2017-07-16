## Syllable-aware Neural Language Models
Code for the Syl-Concat, Syl-Sum, and Syl-CNN models from the paper "Syllable-aware Neural Language Models: A Failure to Beat Character-aware Ones" (EMNLP 2017)

### Requirements
Code is written in Python 3 and requires TensorFlow 1.1+. It also requires the following Python modules: `numpy`, `pyphen`, `argparse`. You can install them via:
```
sudo pip3 install numpy pyphen argparse
```

### Data
Data should be put into the `data/` directory, split into `train.txt`, `valid.txt`, and `test.txt`. Each line of the .txt file should be a sentence. The English Penn Treebank (PTB) data is given as the default.

The non-English data (Czech, French, German, Russian, and Spanish) can be downloaded from [Jan Botha's website](https://bothameister.github.io). For ease of use you can use the [Yoon Kim's script](https://github.com/yoonkim/lstm-char-cnn/blob/master/get_data.sh), which downloads these data and saves them into the relevant folders.

#### Note on non-English data
The PTB data above does not have end-of-sentence tokens for each sentence, and by default these are
appended. The non-English data already have end-of-sentence tokens for each line so, you want to add
`--eos " "` to the command line. 

For non-English data you also need to specify the appropriate hyphenation dictionary which is supported by [Pyphen](http://pyphen.org) via the `--dict` option. The complete list of dictionaries is available at [LibreOffice's repository](https://cgit.freedesktop.org/libreoffice/dictionaries/tree/).

### Model
To reproduce the Syl-Concat result on English PTB from Table 1
```
python3 LSTM-Syl.py
```
To reproduce the Syl-Sum result on English PTB from Table 3 use
```
python3 LSTM-Syl.py --model sum --size medium
```
To reproduce the Syl-CNN result on Czech DATA-S from Table 3 use
```
python3 LSTM-Syl.py --model cnn --size medium --data_dir data/cs --dict cs_CZ --eos " "
```

### Sampled Softmax
Training on a larger vocabulary will require sampled softmax (SSM) to train at a reasonable speed. You can use the `--ssm 1` option to do this.

### Other options
To see the full list of options run
```
python3 LSTM-Syl.py -h
```
