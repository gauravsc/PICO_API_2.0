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
from model.Models import CNNModel, BERTCLassifierModelAspect

# Global variables
batch_size = 4
clip_norm = 20.0
max_epochs = 150
device = 'cuda:0'
load_model = False
threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# set random seed
rd.seed(9001)
np.random.seed(9001)

def extract_target_vocab(data):
	vocab = []
	for sample in data:
		vocab += [triplet[2] for triplet in sample['population condition'] if triplet[2] != "NULL"]
		vocab += [triplet[2] for triplet in sample['intervention applied'] if triplet[2] != "NULL"]
		vocab += [triplet[2] for triplet in sample['outcome condition'] if triplet[2] != "NULL"]
	idx_to_cui = list(set(vocab))

	cui_to_idx = {}
	for idx, cui in enumerate(idx_to_cui):
		cui_to_idx[cui] = idx

	return idx_to_cui, cui_to_idx 

def concept_cui_mapping(data):
	cui_to_concept = {}; concept_to_cui = {}
	for sample in data:
		triplets = sample['population condition'] + sample['intervention applied'] + sample['outcome condition']
		for triplet in triplets:
			if triplet[1] != 'NULL' and triplet[2] != 'NULL': 
				concept_to_cui[triplet[1]] = triplet[2]
				cui_to_concept[triplet[2]] = triplet[1]

	return cui_to_concept, concept_to_cui

def extract_synonyms(word):
	synonyms = []

	for syn in wordnet.synsets(word):
		for l in syn.lemmas():
			synonyms.append(l.name())

	return list(set(synonyms))


def display(results):
	print ("F1 Score Micro Population: ", f1_score_micro_p)
	print ("F1 Score Macro Population: ", f1_score_macro_p)
	print ("F1 Score Micro intervention: ", f1_score_micro_i)
	print ("F1 Score Macro intervention: ", f1_score_macro_i)
	print ("F1 Score Micro Outcome: ", f1_score_micro_o)
	print ("F1 Score Macro Outcome: ", f1_score_macro_o)

	print ("F1 Score Macro: ", (f1_score_macro_p+f1_score_macro_i+f1_score_macro_o)/3.0)
	print ("F1 Score Micro: ", (f1_score_micro_p+f1_score_micro_i+f1_score_micro_o)/3.0)



def prepare_data_for_label_training(data, cui_to_idx, tokenizer):
	X = []; Y_p = []; Y_i = []; Y_o = []; Mask = []; concepts = defaultdict(list)
	aspects = ['population condition', 'intervention applied', 'outcome condition']

	sentence_to_prepend_based_on_aspect = {
	'population condition': 'The population of the trials was ',
	'intervention applied': 'The intervention applied was ',
	'outcome condition': 'The outcome of the study was ' 
	}

	for article in data:
		for aspect in aspects:
			concepts[aspect] += [triplet[1] for triplet in article[aspect] if triplet[1] != "NULL"]

	for aspect in aspects:
		concepts[aspect] = list(set(concepts[aspect]))

	for aspect in aspects:
		for concept in concepts[aspect]:
			input_text = concept
			tokenized_text = tokenizer.tokenize('[CLS] ' + sentence_to_prepend_based_on_aspect[aspect].lower() + input_text.lower())[0:512]
			idx_seq = tokenizer.convert_tokens_to_ids(tokenized_text)
			src_seq = np.zeros(max_seq_len)
			src_seq[0:len(idx_seq)] = idx_seq
			X.append(src_seq)

			# input padding mask 
			mask = np.zeros(max_seq_len)
			mask[0:len(idx_seq)] = 1
			Mask.append(mask)

			# population target
			tgt_seq_p = np.zeros(len(cui_to_idx))
			if aspect == 'population condition':
				tgt_idx_p = [cui_to_idx[concept_to_cui[concept]]]
				tgt_seq_p[tgt_idx_p] = 1
			Y_p.append(tgt_seq_p)

			# intervention target
			tgt_seq_i = np.zeros(len(cui_to_idx))
			if aspect == 'intervention applied':
				tgt_idx_i = [cui_to_idx[concept_to_cui[concept]]]
				tgt_seq_i[tgt_idx_i] = 1
			Y_i.append(tgt_seq_i)

			# outcome target
			tgt_seq_o = np.zeros(len(cui_to_idx))
			if aspect == 'outcome condition':
				tgt_idx_o = [cui_to_idx[concept_to_cui[concept]]]
				tgt_seq_o[tgt_idx_o] = 1
			Y_o.append(tgt_seq_o)

	for it in range(4):
		for aspect in aspects:
			for concept in concepts[aspect]:
				input_text = concept.lower()
				input_word_seq = word_tokenize(input_text)
				new_word_seq = []
				for word in input_word_seq:
					synonyms = extract_synonyms(word)
					if len(synonyms) > 0:
						rand_syn = rd.sample(synonyms, 1)[0]
					else:
						rand_syn = word
					new_word_seq.append(rand_syn)

				input_text = ' '.join(new_word_seq)
				tokenized_text = tokenizer.tokenize('[CLS] ' + sentence_to_prepend_based_on_aspect[aspect].lower() + input_text.lower())[0:512]
				idx_seq = tokenizer.convert_tokens_to_ids(tokenized_text)
				src_seq = np.zeros(max_seq_len)
				src_seq[0:len(idx_seq)] = idx_seq
				X.append(src_seq)

				# input padding mask 
				mask = np.zeros(max_seq_len)
				mask[0:len(idx_seq)] = 1
				Mask.append(mask)

				# population target
				tgt_seq_p = np.zeros(len(cui_to_idx))
				if aspect == 'population condition':
					tgt_idx_p = [cui_to_idx[concept_to_cui[concept]]]
					tgt_seq_p[tgt_idx_p] = 1
				Y_p.append(tgt_seq_p)

				# intervention target
				tgt_seq_i = np.zeros(len(cui_to_idx))
				if aspect == 'intervention applied':
					tgt_idx_i = [cui_to_idx[concept_to_cui[concept]]]
					tgt_seq_i[tgt_idx_i] = 1
				Y_i.append(tgt_seq_i)

				# outcome target
				tgt_seq_o = np.zeros(len(cui_to_idx))
				if aspect == 'outcome condition':
					tgt_idx_o = [cui_to_idx[concept_to_cui[concept]]]
					tgt_seq_o[tgt_idx_o] = 1
				Y_o.append(tgt_seq_o)


	X = np.vstack(X); Y_p = np.vstack(Y_p); Y_i = np.vstack(Y_i); Y_o = np.vstack(Y_o); Mask = np.vstack(Mask)
	
	return X, Mask, Y_p, Y_i, Y_o


def prepare_data(data, cui_to_idx, tokenizer):
	X = []; Y_p = []; Y_i = []; Y_o = []; Mask = []

	for article in data:
		input_text = article['population text'] + article['intervention text'] + article['outcome text']
		tokenized_text = tokenizer.tokenize('[CLS] ' + input_text.lower())[0:512]
		idx_seq = tokenizer.convert_tokens_to_ids(tokenized_text)
		src_seq = np.zeros(max_seq_len)
		src_seq[0:len(idx_seq)] = idx_seq
		X.append(src_seq)
		
		# input padding mask 
		mask = np.zeros(max_seq_len)
		mask[0:len(idx_seq)] = 1
		Mask.append(mask)

		# population target
		tgt_seq_p = np.zeros(len(cui_to_idx))
		tgt_idx_p = [cui_to_idx[triplet[2]] for triplet in article['population condition'] if triplet[2] != "NULL" 
		and p_label_cnt[triplet[2]] > 0]
		tgt_seq_p[tgt_idx_p] = 1
		Y_p.append(tgt_seq_p)

		# intervention target
		tgt_seq_i = np.zeros(len(cui_to_idx))
		tgt_idx_i = [cui_to_idx[triplet[2]] for triplet in article['intervention applied'] if triplet[2] != "NULL"
		and i_label_cnt[triplet[2]] > 0]
		tgt_seq_i[tgt_idx_i] = 1
		Y_i.append(tgt_seq_i)

		# outcome target
		tgt_seq_o = np.zeros(len(cui_to_idx))
		tgt_idx_o = [cui_to_idx[triplet[2]] for triplet in article['outcome condition'] if triplet[2] != "NULL"
		and o_label_cnt[triplet[2]] > 0]
		tgt_seq_o[tgt_idx_o] = 1
		Y_o.append(tgt_seq_o)

	X = np.vstack(X); Y_p = np.vstack(Y_p); Y_i = np.vstack(Y_i); Y_o = np.vstack(Y_o); Mask = np.vstack(Mask)
	
	return X, Mask, Y_p, Y_i, Y_o




def train(model, train_data, val_data, all_data, criterion, cui_to_idx, idx_to_cui, tokenizer):
	X_1, Mask_1, Y_p_1, Y_i_1, Y_o_1 = prepare_data(train_data, cui_to_idx, tokenizer)
	X_2, Mask_2, Y_p_2, Y_i_2, Y_o_2 = prepare_data_for_label_training(all_data, cui_to_idx, tokenizer)
	# X = np.vstack((X_1, X_2)); Mask = np.vstack((Mask_1, Mask_2)); Y_p = np.vstack((Y_p_1, Y_p_2)); Y_i = np.vstack((Y_i_1, Y_i_2)); Y_o = np.vstack((Y_o_1, Y_o_2))
	# shfl_idxs = rd.sample(range(X.shape[0]), X.shape[0])
	# X = X[shfl_idxs]; Mask = Mask[shfl_idxs]; Y_p = Y_p[shfl_idxs]; Y_i = Y_i[shfl_idxs]; Y_o = Y_o[shfl_idxs]
	print ('num docs: ', X_1.shape[0], " num of labels: ", X_2.shape[0])

	best_f1_score = -100; list_losses = []
	for ep in range(max_epochs):
		model = model.train()
		i = 0
		while i < X_1.shape[0]:
			if rd.random() < 0.5 or ep > 70:
				X = X_1; Mask = Mask_1; Y_p = Y_p_1; Y_i = Y_i_1; Y_o = Y_o_1
				is_label_training = False
			else:
				X = X_2; Mask = Mask_2; Y_p = Y_p_2; Y_i = Y_i_2; Y_o = Y_o_2
				is_label_training = True

			indices = rd.sample(range(X.shape[0]), batch_size)

			input_idx_seq = torch.tensor(X[indices]).to(device, dtype=torch.long)
			input_mask = torch.tensor(Mask[indices]).to(device, dtype=torch.long)
			target_p = torch.tensor(Y_p[indices]).to(device, dtype=torch.float)
			target_i = torch.tensor(Y_i[indices]).to(device, dtype=torch.float)
			target_o = torch.tensor(Y_o[indices]).to(device, dtype=torch.float)
			output_p, output_i, output_o = model(input_idx_seq, input_mask, is_label_training=is_label_training, noise_weight = 1.0)

			# computing the loss over the prediction
			loss = (criterion(output_p, target_p) + criterion(output_i, target_i) + criterion(output_o, target_o))*1/3.0
			loss = torch.sum(loss, dim=(1))
			loss = torch.mean(loss)
			print ("loss: ", loss)

			# back-propagation
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
			optimizer.step()

			list_losses.append(loss.data.cpu().numpy())
			i += batch_size

		for threshold in threshold_list:
			f1_score_curr, _ = validate(model, val_data, cui_to_idx, tokenizer, threshold)
			print ("F1 score: ", f1_score_curr, " at threshold: ", threshold)
			if f1_score_curr > best_f1_score:
				torch.save(model.state_dict(), '../saved_models/bert_based/english_labels/aspect_full_model.pt')
				# torch.save(model.bert.state_dict(), '../saved_models/bert_based/bert_retrained_mesh_model.pt')
				best_f1_score = f1_score_curr
		
		print("Loss after epochs ", ep, ":  ", np.mean(list_losses))
		list_losses = []
		
	return model 


def validate(model, data, cui_to_idx, tokenizer, threshold):
	model = model.eval()
	X, Mask, Y_p, Y_i, Y_o = prepare_data(data, cui_to_idx, tokenizer)

	pred_labels_mat_p = []; pred_labels_mat_i = []; pred_labels_mat_o = []; i = 0

	while i < X.shape[0]:			
		input_idx_seq = torch.tensor(X[i:i+4]).to(device, dtype=torch.long)
		input_mask = torch.tensor(Mask[i:i+4]).to(device, dtype=torch.long)
		predict_p, predict_i, predict_o = model(input_idx_seq, input_mask)
		
		predict_p = F.sigmoid(predict_p)
		predict_i = F.sigmoid(predict_i)
		predict_o = F.sigmoid(predict_o)
		
		predict_p[predict_p>=threshold] = 1
		predict_p[predict_p<threshold] = 0
		predict_i[predict_i>=threshold] = 1
		predict_i[predict_i<threshold] = 0
		predict_o[predict_o>=threshold] = 1
		predict_o[predict_o<threshold] = 0
		
		predict_p = predict_p.data.to('cpu').numpy()
		predict_i = predict_i.data.to('cpu').numpy()
		predict_o = predict_o.data.to('cpu').numpy()

		# target_p = Y_p[i:i+4]
		# target_i = Y_i[i:i+4]
		# target_o = Y_o[i:i+4]
		
		pred_labels_mat_p.append(predict_p)
		pred_labels_mat_i.append(predict_i)
		pred_labels_mat_o.append(predict_o)

		i += 4

	# true_labels_mat_p = np.vstack(true_labels_mat_p)
	# true_labels_mat_i = np.vstack(true_labels_mat_i)
	# true_labels_mat_o = np.vstack(true_labels_mat_o)

	pred_labels_mat_p = np.vstack(pred_labels_mat_p)
	pred_labels_mat_i = np.vstack(pred_labels_mat_i)
	pred_labels_mat_o = np.vstack(pred_labels_mat_o)

	results = {}
	f1_score_micro_p, f1_score_macro_p = f1_score(Y_p, pred_labels_mat_p)
	pr_score_micro_p, pr_score_macro_p = precision_score(Y_p, pred_labels_mat_p)
	re_score_micro_p, re_score_macro_p = recall_score(Y_p, pred_labels_mat_p)
	results['f1_score_micro_p'] = f1_score_micro_p
	results['f1_score_macro_p'] = f1_score_macro_p
	results['pr_score_micro_p'] = pr_score_micro_p
	results['pr_score_macro_p'] = pr_score_macro_p
	results['re_score_micro_p'] = re_score_micro_p
	results['re_score_macro_p'] = re_score_macro_p

	f1_score_micro_i, f1_score_macro_i = f1_score(Y_i, pred_labels_mat_i)
	pr_score_micro_i, pr_score_macro_i = precision_score(Y_i, pred_labels_mat_i)
	re_score_micro_i, re_score_macro_i = recall_score(Y_i, pred_labels_mat_i) 
	results['f1_score_micro_i'] = f1_score_micro_i
	results['f1_score_macro_i'] = f1_score_macro_i
	results['pr_score_micro_i'] = pr_score_micro_i
	results['pr_score_macro_i'] = pr_score_macro_i
	results['re_score_micro_i'] = re_score_micro_i
	results['re_score_macro_i'] = re_score_macro_i

	f1_score_micro_o, f1_score_macro_o = f1_score(Y_o, pred_labels_mat_o)
	pr_score_micro_o, pr_score_macro_o = precision_score(Y_o, pred_labels_mat_o)
	re_score_micro_o, re_score_macro_o = recall_score(Y_o, pred_labels_mat_o)  
	results['f1_score_micro_o'] = f1_score_micro_o
	results['f1_score_macro_o'] = f1_score_macro_o
	results['pr_score_micro_o'] = pr_score_micro_o
	results['pr_score_macro_o'] = pr_score_macro_o
	results['re_score_micro_o'] = re_score_micro_o
	results['re_score_macro_o'] = re_score_macro_o
	results['avg_micro_f1_score'] = (f1_score_micro_p + f1_score_micro_i + f1_score_micro_o)/3.0

	# display(results)

	return (f1_score_micro_p + f1_score_micro_i + f1_score_micro_o)/3.0, results


def tune_threshold(model, data, cui_to_idx, tokenizer):
	best_threshold = 0.0
	best_f1_score = -100
	for threshold in threshold_list:
		f1_score_curr, _ = validate(model, data, cui_to_idx, tokenizer, threshold)
		print ("F1 score: ", f1_score_curr, " at threshold: ", threshold)
		if f1_score_curr > best_f1_score:
				best_f1_score = f1_score_curr
				best_threshold = threshold
	return best_threshold



if __name__ == '__main__':
	# load the dataset
	data = json.load(open('../data/data_with_cuis.json', 'r'))
	# concept to cui mappings
	cui_to_concept, concept_to_cui = concept_cui_mapping(data) 
	# # create the vocabulary for the input 
	idx_to_cui, cui_to_idx = extract_target_vocab(data)
	# Load pre-trained model tokenizer (vocabulary)
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', max_len=512)

	# load label count 
	label_cnt = json.load(open('../data/label_counts.json', 'r'))
	p_label_cnt = label_cnt['p_label_cnt']
	i_label_cnt = label_cnt['i_label_cnt']
	o_label_cnt = label_cnt['o_label_cnt']

	# Split train and test data
	train_idx = rd.sample(range(len(data)), int(0.9*len(data)))
	test_idx = [i for i in range(len(data)) if i not in train_idx]

	train_data = [data[i] for i in train_idx]
	test_data = [data[i] for i in test_idx]

	val_idx = rd.sample(range(len(train_data)), 300)
	train_idx = [i for i in range(len(train_data)) if i not in val_idx]
	val_data = [train_data[i] for i in val_idx]
	train_data = [train_data[i] for i in train_idx]

	# setting different model parameters
	n_tgt_vocab = len(cui_to_idx)
	max_seq_len = 512
	d_word_vec = 200
	dropout = 0.1
	learning_rate = 0.005

	model = BERTCLassifierModelAspect(n_tgt_vocab, dropout=dropout)
	# model = nn.DataParallel(model, output_device=device)
	model.to(device)

	if load_model:
		model.load_state_dict(torch.load('../saved_models/bert_based/english_labels/aspect_full_model.pt'))
		print ("Done loading the saved model .....")

	criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
	# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9,0.999))
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

	model = train(model, train_data, val_data, data, criterion, cui_to_idx, idx_to_cui, tokenizer)
	# load the best performing model
	model.load_state_dict(torch.load('../saved_models/bert_based/english_labels/aspect_full_model.pt'))
	best_threshold = tune_threshold(model, val_data, cui_to_idx, tokenizer)
	print ("Best threshold was: ", best_threshold)
	_, results = validate(model, test_data, cui_to_idx, tokenizer, best_threshold)

	print (results)



