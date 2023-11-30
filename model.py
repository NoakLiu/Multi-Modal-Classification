# model.py
import torch
from torch import nn
from torchvision import models
from gensim.models import FastText


class Resnext50(nn.Module):
    def __init__(self, n_class):
        super(Resnext50, self).__init__()
        resnet = models.resnext50_32x4d(pretrained=True)
        self.in_features = resnet.fc.in_features
        resnet.fc = nn.Linear(resnet.fc.in_features, n_class)
        self.base_model = resnet
        self.activation = nn.Sigmoid()
        self.out_features = n_class

    def forward(self, x, is_feature_extractor=True):
        if not is_feature_extractor:
            return self.activation(self.base_model(x))
        self.out_features = self.in_features
        self.base_model.fc = nn.Identity()
        features = self.base_model(x)
        return features


class TextEmbedding(nn.Module):
    def __init__(self, word_list, training_data):
        super(TextEmbedding, self).__init__()
        self.emb_table = None
        self.training_data = training_data
        self.model = FastText(sentences=self.training_data, size=150, window=5, min_count=1, workers=4, sg=1)
        self.keys = list(self.model.wv.vocab.keys())
        self.emb_dim = self.model.vector_size

    def get_embedding(self):
        model = self.model
        assert (model is not None)
        emb_table = []
        for i, word in enumerate(self.word_list):
            if word in self.keys:
                word_emb = model.wv[word]
                emb_table.append(word_emb)
            else:
                word_emb = [0] * self.emb_dim
                emb_table.append(word_emb)
        emb_table = np.array(emb_table)
        self.emb_table = emb_table
        return emb_table


class BiGRUAttention(nn.Module):
    def __init__(self, emb_table, n_hidden, n_emb):
        super(BiGRUAttention, self).__init__()
        self.n_hidden = n_hidden
        self.n_emb = n_emb
        self.emb = nn.Embedding(emb_table.shape[0], emb_table.shape[1])
        self.emb.weight.data.copy_(torch.from_numpy(emb_table))
        self.gru = nn.GRU(emb_table.shape[1], n_hidden, batch_first=True, bidirectional=True)
        self.encoder_fc = nn.Linear(2 * n_hidden, n_emb)
        self.activation = nn.ReLU()

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.n_hidden * 2, 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights

    def forward(self, X):
        x = self.emb(X)
        hidden_state = Variable(torch.zeros(1 * 2, x.shape[1], self.n_hidden))
        output, final_hidden_state = self.gru(x)
        attn_output, attention = self.attention_net(output, final_hidden_state)
        features = self.activation(self.encoder_fc(attn_output))
        return features


class ConcatenateEmbedModel(nn.Module):
    def __init__(self, img_emb_model, text_emb_model, emb_dim, n_classes):
        super(ConcatenateEmbedModel, self).__init__()
        self.img_emb_model = img_emb_model
        self.text_emb_model = text_emb_model
        self.n_classes = n_classes
        self.emb_dim = emb_dim
        self.img_fc = nn.Sequential(
            nn.Linear(img_emb_model.out_features, 256),
            nn.ReLU(),
            nn.Linear(256, self.emb_dim))
        self.text_fc = nn.Sequential(
            nn.Linear(text_emb_model.emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.emb_dim))
        self.fc = nn.Sequential(
            nn.Linear(2 * self.emb_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.n_classes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, text, is_attention=False):
        img_emb = self.img_emb_model.forward(img, is_feature_extractor=True)
        text_emb = self.text_emb_model.forward(text)
        img_norm = nn.BatchNorm1d(img_emb.shape[1])
        text_norm = nn.BatchNorm1d(text_emb.shape[1])
        img_emb = img_norm(img_emb)
        text_emb = text_norm(text_emb)
        img_emb = self.img_fc(img_emb)
        text_emb = self.text_fc(text_emb)
        concate_emb = torch.cat((img_emb, text_emb), 1)
        output_before_sigmoid = self.fc(concate_emb)
        output = self.sigmoid(output_before_sigmoid)
        return output_before_sigmoid, output
