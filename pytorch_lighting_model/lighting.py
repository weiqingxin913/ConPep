import os
import torch
from torch import nn
import pytorch_lightning as pl

d_model = 512  # Embedding Size（token embedding和position编码的维度）
d_ff = 1024  # FeedForward dimension (两次线性层中的隐藏层 512->2048->512，线性层是用来做特征提取的），当然最后会再接一个projection层
d_k = d_v = 64  # dimension of K(=Q), V（Q和K的维度需要相同，这里为了方便让K=V）
n_layers = 1  # number of Encoder of Decoder Layer（Block的个数）
n_heads = 2  # number of heads in Multi-Head Attention（有几套头）
# from

T5Prot = "/mnt/8t/jy/HyperGAT_TextClassification-main/HyperGAT_TextClassification-main/Bert-BiLSTM-CRF-pytorch/Rostlab/prot_t5_xl_uniref50"
from transformers import T5Model, T5Tokenizer
import re


class ContactMapEncoder(pl.LightningModule):
    def __init__(self, t5_model_dim=1024):
        super().__init__()
        self.t5_model = T5Model.from_pretrained(T5Prot)
        self.t5_model.eval()
        self.bi_gru = nn.GRU(input_size=t5_model_dim,
                             hidden_size=t5_model_dim,
                             num_layers=1,
                             bidirectional=True)
        self.projection = nn.Linear(t5_model_dim, 3, bias=False)
        # self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def pdist(self, a, b, p: int = 2):
        return ((a - b).abs().pow(p).sum(-1) + 1e-10).pow(1 / p)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        # print(x)
        # print(x.device)
        input_ids, attention_mask = self.tokenize_module(x)

        with torch.no_grad():
            embedding = self.t5_model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=None)
        enc_outputs = embedding[2]
        enc_outputs = self.bi_gru(enc_outputs)
        # print(enc_outputs.size())
        enc_outputs = self.projection(enc_outputs)
        return enc_outputs

    def training_step(self, batch, batch_idx):
        # training_step defined the para loop.
        # It is independent of forward

        x, y = batch
        # print(x.device)
        x = x.view(x.size(0), -1)
        enc_outputs, attention = self.Transformer(x)
        output_list = []
        for value in enc_outputs:
            # pairwise_distances(value)
            value_ = value.unsqueeze(1)
            output_list.append(self.pdist(value_, value))
        output = torch.stack(output_list, dim=0)

        loss = nn.PairwiseDistance(p=2, eps=1e-06)(output, y).sum()
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        enc_outputs, attention = self.Transformer(x)
        output_list = []
        for value in enc_outputs:
            # pairwise_distances(value)
            # print(value)

            value_ = value.unsqueeze(1)
            output_list.append(self.pdist(value_, value))

        output = torch.stack(output_list, dim=0)

        loss = nn.PairwiseDistance(p=2, eps=1e-06)(output, y).sum()
        self.log('val_loss', loss)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        enc_outputs, attention = self.Transformer(x)
        output_list = []
        for value in enc_outputs:
            # pairwise_distances(value)
            # print(value)

            value_ = value.unsqueeze(1)
            output_list.append(self.pdist(value_, value))

        output = torch.stack(output_list, dim=0)

        loss = nn.PairwiseDistance(p=2, eps=1e-06)(output, y).sum()
        self.log('val_loss', loss)

    def tokenize_module(self, input):
        tokenizer = T5Tokenizer.from_pretrained(T5Prot, do_lower_case=False)
        re_sequence = [re.sub(r"[UZOB]", "X", sequence) for sequence in input]
        ids = tokenizer.batch_encode_plus(re_sequence, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids'])
        attention_mask = torch.tensor(ids['attention_mask'])
        return input_ids, attention_mask
