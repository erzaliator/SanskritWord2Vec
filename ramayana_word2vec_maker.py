#-*- coding: UTF-8 -*-
'''
Helper functions for word2vec functions
TODO: Implement for other corpora
TODO: Use logging
TODO: get_sims()
TODO: make assemble_sanskrit_author_filepaths
'''
import os
import time
from multiprocessing import cpu_count

#corpus requirements
from cltk.corpus.utils.importer import CorpusImporter

#preprocessing
from cltk.tokenize.indian_tokenizer import *
from cltk.stop.sanskrit.stops import STOPS_LIST

#word2vec embeddings
try:
	from gensim.models import Word2Vec
	from gensim.models.word2vec import LineSentence
except ImportError:
    print('Gensim not installed.')
    raise

from numpy import ndarray

def pre_process(line,rm_stops=False):
	'''Preprocess input line and convert into words separated by spaces'''

	#tokenize
	line = indian_punctuation_tokenize_regex(line)

	#strip punctutations
	indic_punkt_list = [u',', u'।', u'\n', u'\t']
	for del_punkt in indic_punkt_list:
		line[:] = (value for value in line if value != del_punkt)

	#stopword removal
	if rm_stops:
		line = [w for w in line if w not in STOPS_LIST]

	#no lemmatizer in sanskrit

	return line


def read_doc(infile, outfile, rm_stops=False):
	'''Reads the file specified in filepath and makes word2vec model in outfile'''

	for line in infile:
		#preprocessing
		line = pre_process(line,rm_stops)

		#convert to a string
		line = u' '.join(line)

		#write line-by-line into file
		outfile.write(line + os.linesep)


def gen_docs(corpus, lemmatize, rm_stops=False):
	'''Opens and processes file. Stores in processed .txt files to be used for making the models.'''

	language = 'sanskrit'
	assert corpus in ['ramayana']

	#assert if gita corpus exists
	path = os.path.join(os.path.expanduser('~'),
		'cltk_data/sanskrit/parallel/sanskrit_parallel_gitasupersite/')
	if(not os.path.exists(path)):
		print('Importing \'sanskrit_parallel_gitasupersite\'...')
		c = CorpusImporter('sanskrit')
		c.import_corpus('sanskrit_parallel_gitasupersite')
	else:
		print('Not importing corpora...')
	
	#make path in ramayana for word2vec_models
	path = os.path.join(os.path.expanduser('~'),
		'cltk_data/sanskrit/parallel/sanskrit_parallel_gitasupersite/ramayana/word2vec_models')
	if(not os.path.exists(path)):	os.makedirs(path)	
	
	#make preprocessed text and make the models
	path = os.path.join(os.path.expanduser('~'),
		'cltk_data/sanskrit/parallel/sanskrit_parallel_gitasupersite/ramayana/')

	for name in [file for file in os.listdir(path) if file.endswith(".txt")]:
		#for each file in ramayana folder
		if name.endswith('_sanskrit.txt'):
			
			#txt file to read data from
			f = open(path + name)
			print('Reading ' + path + name)

			#make raw data file to store preprocessed text
			if os.path.exists(path + 'word2vec_models/ramayana_sanskrit.txt'):
				append_write = 'a'
			else:
				append_write = 'w'
			f2 = open(path + 'word2vec_models/ramayana_sanskrit.txt', append_write)
			print( 'and storing in ' + path + 'word2vec_models/ramayana_sanskrit.txt\n')
			read_doc(f, f2, rm_stops)

			f.close()
			f2.close()
	return True

def assert_models(save_path=None):
	'''Assert if models have been made in save_path'''

	try:
		#set the default path
		if not save_path:
			save_path = os.path.join(os.path.expanduser('~'),
				'cltk_data/sanskrit/parallel/sanskrit_parallel_gitasupersite/ramayana/word2vec_models/')

		if not os.path.exists(save_path):
			return False

		if not os.path.exists(save_path+'ramayana_sanskrit.model'):
			return False

		try:
			model = Word2Vec.load(save_path+'ramayana_sanskrit.model')
			if not model:
				return False

			#a common sanskrit word for test; eed a better test
			test_string = 'तु'
			if not isinstance(model[test_string],ndarray):
				return False
		except:
			return False
	except:
		return False
	return True


def make_model(corpus, lemmatize=False, rm_stops=False, size=100, window=10, min_count=5, workers=1, sg=1, save_path=None):
	'''Train word2vec model'''

	t0 = time.time()

	#make w2v models only if models do not exist
	print("First sanity check:", assert_models(save_path))
	if not assert_models(save_path):
		#Delete the model file if it previously exists
		path = os.path.join(os.path.expanduser('~'),
			'cltk_data/sanskrit/parallel/sanskrit_parallel_gitasupersite/ramayana/word2vec_models/ramayana_sanskrit.txt')
		if os.path.exists(path):	os.remove(path)

		gen_flag = gen_docs(corpus, lemmatize=lemmatize, rm_stops=rm_stops)
		
		if not gen_flag:
			print('Error in making word2vec raw text files')
			return

		#convert preprocessed text and make models
		path = os.path.join(os.path.expanduser('~'),
			'cltk_data/sanskrit/parallel/sanskrit_parallel_gitasupersite/ramayana/word2vec_models/ramayana_sanskrit.txt') 
		#create model from preprocessed file
		model = Word2Vec(LineSentence(path), size=100, window=10, min_count=5, workers=cpu_count(), sg=sg)

		#trim unneeded model memory = use (much) less RAM
		model.init_sims(replace=True)

		#save model
		save_path = path.replace('txt','model')
		print("Model saved at ", save_path)
		model.save(save_path)

	print('\n\nTotal training time for {0}: {1} minutes'.format(save_path, (time.time() - t0) / 60))



if __name__ == '__main__':
	lang = 'sanskrit'
	corpus = 'ramayana'
	rm_stops = True
	default_save_path = os.path.join(os.path.expanduser('~'),
		'cltk_data/sanskrit/parallel/sanskrit_parallel_gitasupersite/ramayana/word2vec_models/')

	make_model(corpus, lemmatize=False, rm_stops=False, size=100, window=10, min_count=5, workers=cpu_count(), sg=1, save_path=default_save_path)

	# final sanity check for models
	if assert_models(save_path=None):
		print('Word2Vec models made successfully in: ~/cltk_data/sanskrit/parallel/sanskrit_parallel_gitasupersite/ramayana/word2vec_models/')
	else:
		print('Something went wrong. Models were not created fully.')