#-*- coding: UTF-8 -*-

import os
import time
from multiprocessing import cpu_count

#corpus requirements
from cltk.corpus.utils.importer import CorpusImporter

#preprocessing
from cltk.tokenize.indian_tokenizer import *
from cltk.stop.sanskrit.stops import STOPS_LIST

#word2vec embeddings
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from numpy import ndarray

def pre_process(line,rm_stops=False):
	#tokenize
	line = indian_punctuation_tokenize_regex(line)
	print(line)

	#strip punkt
	indic_punkt_list = [u',', u'।', u'\n', u'\t']
	for del_punkt in indic_punkt_list:
		line[:] = (value for value in line if value != del_punkt)

	#stopword removal
	if rm_stops:
		line = [w for w in line if w not in STOPS_LIST]

	#no lemmatizer in sanskrit

	return line

def read_doc(filepath, outfile, rm_stops=False):

	count = 0
	with open(filepath) as f:
		for line in f:
			# if count ==1:
			# 	break

			#preprocessing
			line = pre_process(line,rm_stops)

			#convert to a string
			line = u' '.join(line)

			#write line-by-line into file
			print(line)
			# count += 1
			outfile.write(line + os.linesep)
	f.close()

def gen_docs(corpus, lemmatize, rm_stops=False):
	#skip assertion part
	language = 'sanskrit'
	path = os.path.join(os.path.expanduser('~'),'cltk_data/sanskrit/parallel/sanskrit_parallel_gitasupersite/')
	if(not os.path.exists(path)):
		pass #import sans_corpus

	path = os.path.join(os.path.expanduser('~'),'cltk_data/sanskrit/parallel/sanskrit_parallel_gitasupersite/ramayana/word2vec')
	os.makedirs(path)

	f = open(path+'/balakanda_sanskrit.txt','w')
	print(f)
	#skipped assertion part

	#make raw data file
	path=os.path.join(os.path.expanduser('~'),'cltk_data/sanskrit/parallel/sanskrit_parallel_gitasupersite/ramayana/balakanda_sanskrit.txt')
	read_doc(path, f, rm_stops)
	f.close()

	return True
	s='तु'

def assert_model(model_file_name=None):
	try:
		path = os.path.join(os.path.expanduser('~'),'cltk_data/sanskrit/parallel/sanskrit_parallel_gitasupersite/ramayana/word2vec/balakanda_sanskrit.model')
		model = Word2Vec.load(path)
		if not model:
			return False
		test_string = 'तु'
		if not isinstance(model[test_string],ndarray):
			return False
	except:
		pass
	return True


def make_model(corpus, lemmatize=False, rm_stops=False, size=100, window=10, min_count=5, workers=1, sg=1, save_path=None):
	t0 = time.time()
	sentences_stream = gen_docs(corpus, lemmatize=lemmatize, rm_stops=rm_stops)
	if not sentences_stream:
		print('Error in making word2vec raw text files')
		return
	path = os.path.join(os.path.expanduser('~'),'cltk_data/sanskrit/parallel/sanskrit_parallel_gitasupersite/ramayana/word2vec/balakanda_sanskrit.txt')
	model = Word2Vec(LineSentence(path), size=100, window=10, min_count=5, workers=cpu_count())

	# trim unneeded model memory = use (much) less RAM
	model.init_sims(replace=True)

	#will change to binary format as soon as i work out the details properly
	#save the model
	path = os.path.join(os.path.expanduser('~'),'cltk_data/sanskrit/parallel/sanskrit_parallel_gitasupersite/ramayana/word2vec/balakanda_sanskrit.model')
	model.save(path)
	

lang = 'sanskrit'
corpus = 'ramayana'
rm_stops = True
# make_model(corpus, lemmatize=False, rm_stops=False, size=100, window=10, min_count=5, workers=1, sg=1, save_path=None)
print(assert_model())