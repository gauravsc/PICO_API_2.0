import json
import os
import pickle
import numpy as np
import random as rd
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer
from model.Models import  BERTCLassifierModelAspect

# set random seed
rd.seed(9001)
np.random.seed(9001)

# class of the model API
class PICOModel:
	
	def extract_target_vocab(self, data):
		vocab = []
		for sample in data:
			vocab += [triplet[2] for triplet in sample['population condition'] if triplet[2] != "NULL"]
			vocab += [triplet[2] for triplet in sample['intervention applied'] if triplet[2] != "NULL"]
			vocab += [triplet[2] for triplet in sample['outcome condition'] if triplet[2] != "NULL"]
		idx_to_cui = list(set(vocab))
		idx_to_cui = sorted(idx_to_cui)
		# print(idx_to_cui)

		cui_to_idx = {}
		for idx, cui in enumerate(idx_to_cui):
			cui_to_idx[cui] = idx

		return idx_to_cui, cui_to_idx 

	def concept_cui_mapping(self, data):
		cui_to_concept = {}; concept_to_cui = {}
		for sample in data:
			triplets = sample['population condition'] + sample['intervention applied'] + sample['outcome condition']
			for triplet in triplets:
				if triplet[1] != 'NULL' and triplet[2] != 'NULL': 
					concept_to_cui[triplet[1]] = triplet[2]
					cui_to_concept[triplet[2]] = triplet[1]

		return cui_to_concept, concept_to_cui

	def prepare_data(self, data):
		X = []; Mask = []

		for article in data:
			input_text = article['abstract']		
			tokenized_text = self.tokenizer.tokenize('[CLS] ' + input_text.lower())[0:512]
			idx_seq = self.tokenizer.convert_tokens_to_ids(tokenized_text)
			src_seq = np.zeros(self.max_seq_len)
			src_seq[0:len(idx_seq)] = idx_seq
			X.append(src_seq)
			
			# input padding mask 
			mask = np.zeros(self.max_seq_len)
			mask[0:len(idx_seq)] = 1
			Mask.append(mask)

		X = np.vstack(X)
		Mask = np.vstack(Mask)
		
		return X, Mask


	def predict(self, data):
		X, Mask = self.prepare_data(data)
				
		input_idx_seq = torch.tensor(X, dtype=torch.long)
		input_mask = torch.tensor(Mask)
		predict_p, predict_i, predict_o = self.model(input_idx_seq, input_mask)
		
		predict_p = F.sigmoid(predict_p)
		predict_i = F.sigmoid(predict_i)
		predict_o = F.sigmoid(predict_o)
		
		predict_p[predict_p>=self.threshold] = 1
		predict_p[predict_p<self.threshold] = 0
		predict_i[predict_i>=self.threshold] = 1
		predict_i[predict_i<self.threshold] = 0
		predict_o[predict_o>=self.threshold] = 1
		predict_o[predict_o<self.threshold] = 0
		
		predict_p = predict_p.data.numpy()
		predict_i = predict_i.data.numpy()
		predict_o = predict_o.data.numpy()
			
		idx_p = np.nonzero(predict_p[0])[0]
		idx_i = np.nonzero(predict_i[0])[0]
		idx_o = np.nonzero(predict_o[0])[0]
		
		population_concepts = [self.cui_to_concept[self.idx_to_cui[idx]] for idx in idx_p]
		intervention_concepts = [self.cui_to_concept[self.idx_to_cui[idx]] for idx in idx_i]
		outcome_concepts = [self.cui_to_concept[self.idx_to_cui[idx]] for idx in idx_o]

		results = {
		'population': population_concepts,
		'intervention': intervention_concepts,
		'outcome': outcome_concepts
		}

		return results


	def __init__(self):
		# load the dataset
		data = json.load(open('../data/data_with_cuis.json', 'r'))
		# concept to cui mappings
		self.cui_to_concept, self.concept_to_cui = self.concept_cui_mapping(data) 
		# # create the vocabulary for the input 
		self.idx_to_cui, self.cui_to_idx = self.extract_target_vocab(data)
		# load pre-trained model tokenizer (vocabulary)
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', max_len=512)

		# setting different model parameters
		self.n_tgt_vocab = len(self.cui_to_idx)
		self.max_seq_len = 512
		self.d_word_vec = 200
		self.dropout = 0.1
		
		# threshold to use for classification
		self.threshold = 0.4

		self.model = BERTCLassifierModelAspect(self.n_tgt_vocab, dropout=self.dropout)
		# model = nn.DataParallel(model, output_device=device)
		# model.to(device)
		self.model.load_state_dict(torch.load('../saved_model/aspect_full_model.pt', map_location=torch.device('cpu')))
		self.model.eval()
		print ("Done loading the saved model .....")

	# results = predict(model, test_data, idx_to_cui, cui_to_concept, tokenizer, thresholdd)




