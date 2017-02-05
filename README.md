# Goal
The main goal of the project is extending [Ukrainian tonal dictionary](https://github.com/lang-uk/tone-dict-uk). At first, we tried to achieve it by looking at words similar to ones with known tonality. **Word2vec** and **LexVec** models are used to find similar words. Then we built NN classifier and used word embeddings and existing tonal dictionary to train it.

### General
`split_to_chunks/subsample.py` - is used to take a piece of files so it can be read with notepad:  
`utils.py` - has some useful methods and folders paths

### Split to sentences:
We have different sources of text: csv, txt and wiki. There are different files to preprocess them.

##### for csv: (each item is a news or article)

1. split raw csv data to chunks and save as chunks: `split_to_chunks/csv_to_pd_chunks.py`  
Result items are saved in data\chunks  
2. read chunk items, tokenize text, save list of sentences: `split_to_chunks/pd_to_sentences.py`  
Result items are saved in data\sents

##### for txt:
run `split_to_chunks/txt_to_sentences` to tokenize text and save chunks with lists of sentences   
Result is saved to data\sents
 
##### for wiki
run `split_to_chunks/wiki_to_sentences` to tokenize text and save chunks with lists of sentences  

### Cleanup words:
`train/clean.py` - will read all existing sentences files and clean words

### Building word2vec model
`train/build_w2v_dict.py` - will build **word2vec** model  
`train/learn.py` - will train **word2vec** model

### Building LexVec model
[https://github.com/alexandres/lexvec](LexVec) was used on the same data with settings identical to Word2Vec to calculate embeddings.

### Used word embeddings models
If you don't want to calculate word vectors for yourself, you can obtain them from http://lang.org.ua/models website or download from Google Drive ([https://drive.google.com/file/d/0B9adEr6qDus4TjVVUW9CcEkzSjQ/view](LexVec), [https://drive.google.com/open?id=0B9adEr6qDus4dkRpaDZ4bWZCc2M](Word2Vec))

### Using LexVec and word2vec models to predict the tone of the word
`predict/build_joined_vect_dict.py` - is used to concatenate two models: LexVec and word2vec  
`predict/predict.py` - predict, save the whole set, save subsample  
`predict/save_best.py` - take best negative and positive candidates  

### Credits
**Oleksandr Marykovskyi**, **Vyacheslav Tykhonov** provided the seed dictionary  
**Serhiy Shehovtsov** wrote the code and ran numerous experiments  
**Oles Petriv** created and trained neural network model  
**Vsevolod Dyomkin** proof-read the result and prepared it for publishing  
**Dmitry Chaplinsky** led the project :)
