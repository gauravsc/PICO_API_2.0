import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_pretrained_bert import BertModel



class BERTCLassifierModelAspect(nn.Module):
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
