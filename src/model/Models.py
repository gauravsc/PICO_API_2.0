import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_pretrained_bert import BertModel

class CNNModel(nn.Module):
	def __init__(self, n_tgt_vocab, dropout=0.1):
		super().__init__()
		self.src_word_emb = nn.Embedding(100000, 200, padding_idx=0)
		# self.conv_layer_1 = nn.Conv1d(d_word_vec, 1024, 3, padding=1, stride=1)

		self.conv_layer_1_1 = nn.Conv1d(200, 1024, 1, padding=0, stride=1)
		self.conv_layer_1_3 = nn.Conv1d(200, 1024, 3, padding=1, stride=1)
		self.conv_layer_1_5 = nn.Conv1d(200, 1024, 5, padding=2, stride=1)

		# self.conv_layer_2 = nn.Conv1d(3072, 512, 3, padding=1, stride=1)
		# self.conv_layer_3 = nn.Conv1d(512, 256, 3, padding=1, stride=1)
		# self.conv_layer_4 = nn.Conv1d(256, 256, 3, padding=1, stride=1)
		self.maxpool = nn.MaxPool1d(1000)
		self.dropout = nn.Dropout(p=dropout)
		# self.fc_layer_1 = nn.Linear((len_max_seq//16)*256, 256)
		self.layer_norm_1 = nn.LayerNorm(768)
		self.fc_layer_1 = nn.Linear(3072, 768)
		self.output_layer_1 = nn.Linear(768, n_tgt_vocab)
		self.output_layer_2 = nn.Linear(768, n_tgt_vocab)
		self.output_layer_3 = nn.Linear(768, n_tgt_vocab)
		self.relu_activation = nn.ReLU()


	def forward(self, input_idxs):
		output = self.src_word_emb(input_idxs)
		output = output.permute(0,2,1)

		# output = self.conv_layer_1(output)
		# output = self.relu_activation(output)
		# output = self.maxpool(output)

		output_1 = self.conv_layer_1_1(output)
		output_1 = self.relu_activation(output_1)
		output_1 = self.maxpool(output_1)

		output_3 = self.conv_layer_1_3(output)
		output_3 = self.relu_activation(output_3)
		output_3 = self.maxpool(output_3)

		output_5 = self.conv_layer_1_5(output)
		output_5 = self.relu_activation(output_5)
		output_5 = self.maxpool(output_5)

		output = torch.cat((output_1, output_3, output_5), 1)

		# output = self.conv_layer_2(output)
		# output = self.relu_activation(output)
		
		# output = self.maxpool(output)

		# output = self.conv_layer_3(output)
		# output = self.relu_activation(output)
		# output = self.maxpool(output)

		# output = self.conv_layer_4(output)
		# output = self.relu_activation(output)
		# output = self.maxpool(output)

		output = self.dropout(output.view(output.size()[0], -1))
		output_fc_layer = self.relu_activation(self.fc_layer_1(output))
		# output_fc_layer = self.layer_norm_1(self.fc_layer_1(output))
		target_p = self.output_layer_1(self.dropout(output_fc_layer))
		target_i = self.output_layer_2(self.dropout(output_fc_layer))
		target_o = self.output_layer_3(self.dropout(output_fc_layer))

		return target_p, target_i, target_o


class BERTCLassifierModel(nn.Module):
	def __init__(self, n_tgt_vocab, dropout=0.1):
		super().__init__()
		self.bert = BertModel.from_pretrained('bert-base-uncased')
		# self.bert.load_state_dict(torch.load('../saved_models/bert_based/bert_retrained_mesh_model.pt'))
		self.fc_layer_1 = nn.Linear(768, 768)
		self.fc_layer_2 = nn.Linear(768, 768)
		self.fc_layer_3 = nn.Linear(768, 768)
		self.output_layer_1 = nn.Linear(768, n_tgt_vocab)
		self.output_layer_2 = nn.Linear(768, n_tgt_vocab)
		self.output_layer_3 = nn.Linear(768, n_tgt_vocab)

	def forward(self, input_idxs, input_mask):
		enc_out, _ = self.bert(input_idxs, attention_mask=input_mask, output_all_encoded_layers=False)
		
		# extract encoding for the [CLS] token
		enc_out = enc_out[:,0,:]
		enc_out_1 = self.fc_layer_1(enc_out)
		enc_out_2 = self.fc_layer_2(enc_out)
		enc_out_3 = self.fc_layer_3(enc_out)
		
		# pass the embedding for [CLS] token to the final classification layer
		target_p = self.output_layer_1(enc_out_1)
		target_i = self.output_layer_2(enc_out_2)
		target_o = self.output_layer_3(enc_out_3)
		
		return target_p, target_i, target_o


class BERTCLassifierModelAspect(nn.Module):
	def __init__(self, n_tgt_vocab, dropout=0.1):
		super().__init__()
		self.bert = BertModel.from_pretrained('bert-base-uncased')
		self.bert.load_state_dict(torch.load('../saved_models/bert_based/bert_retrained_mesh_model.pt'))
		self.fc_layer_1 = nn.Linear(768, 768)
		self.fc_layer_2 = nn.Linear(768, 768)
		self.fc_layer_3 = nn.Linear(768, 768)
		self.output_layer_1 = nn.Linear(768, n_tgt_vocab)
		self.output_layer_2 = nn.Linear(768, n_tgt_vocab)
		self.output_layer_3 = nn.Linear(768, n_tgt_vocab)

	def forward(self, input_idxs, input_mask, is_label_training=False, noise_weight = 0.0):
		enc_out, _ = self.bert(input_idxs, attention_mask=input_mask, output_all_encoded_layers=False)
		
		# extract encoding for the [CLS] token
		enc_out = enc_out[:,0,:]

		if is_label_training:
			# add noise to the representation
			rand_out = torch.randn_like(enc_out)
			rand_out = rand_out/(torch.norm(rand_out, dim=1)[:, None])
			rand_out = noise_weight * torch.norm(enc_out, dim=1)[:, None] * rand_out
			enc_out = enc_out + rand_out

		enc_out_1 = self.fc_layer_1(enc_out)
		enc_out_2 = self.fc_layer_2(enc_out)
		enc_out_3 = self.fc_layer_3(enc_out)
		
		# pass the embedding for [CLS] token to the final classification layer
		target_p = self.output_layer_1(enc_out_1)
		target_i = self.output_layer_2(enc_out_2)
		target_o = self.output_layer_3(enc_out_3)
		
		return target_p, target_i, target_o


class BERTClassifierLabelTransfer(nn.Module):
	def __init__(self, dropout=0.1):
		super().__init__()
		self.bert = BertModel.from_pretrained('bert-base-uncased')
		self.bert.load_state_dict(torch.load('../saved_models/bert_based/bert_retrained_mesh_model.pt'))
		self.fc_layer_1 = nn.Linear(768, 768)
		self.fc_layer_2 = nn.Linear(768, 768)
		self.fc_layer_3 = nn.Linear(768*2, 768)
		self.output_layer = nn.Linear(768, 1)


	def forward(self, input_idxs, input_mask, target_idxs, target_mask):
		enc_out, _ = self.bert(input_idxs, attention_mask=input_mask, output_all_encoded_layers=False)
		tgt_out, _ = self.bert(target_idxs, attention_mask=target_mask, output_all_encoded_layers=False)
		
		# extract encoding for the [CLS] token
		enc_out = enc_out[:,0,:]
		tgt_out = tgt_out[:,0,:]

		enc_out = self.fc_layer_1(enc_out)
		tgt_out = self.fc_layer_2(tgt_out)

		out = torch.cat((enc_out, tgt_out), 1)
		out = self.fc_layer_3(out)
		# pass the embedding for [CLS] token to the final classification layer
		target = self.output_layer(out)
		
		return target


