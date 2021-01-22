import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


# class EncoderCNN(nn.Module):
#     def __init__(self, embed_size):
#         """Load the pretrained ResNet-152 and replace top fc layer."""
#         super(EncoderCNN, self).__init__()
#         resnet = models.resnet152(pretrained=True)
#         modules = list(resnet.children())[:-1]      # delete the last fc layer.
#         self.resnet = nn.Sequential(*modules)
#         self.linear = nn.Linear(resnet.fc.in_features, embed_size)
#         self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
#
#     def forward(self, images):
#         """Extract feature vectors from input images."""
#         with torch.no_grad():
#             features = self.resnet(images)
#         features = features.reshape(features.size(0), -1)
#         features = self.bn(self.linear(features))
#         return features


# VGG16_bn
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        vgg = models.vgg16_bn(pretrained=True)
        modules = list(vgg.children())[:-1]  # delete the last fc layer.
        self.vgg = nn.Sequential(*modules)
        self.linear = nn.Linear(25088, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.vgg(images)
        # print(features.shape)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids


class EncoderCNN_VGG_Attention(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN_VGG_Attention, self).__init__()
        vgg = models.vgg16_bn(pretrained=True)
        modules = list(vgg.children())[0]
        self.vgg = nn.Sequential(*modules)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.vgg(images)
        # print(features.shape)  # 1, 512, 7, 7
        # batch_size, attention_size, w, h = features.shape
        # a = torch.zeros([batch_size, w * h, attention_size], dtype=features.dtype)
        # for i in range(w):
        #     for j in range(h):
        #         a[:, i*h+j, :] = features[:, :, i, j]
        features = features.permute(0, 2, 3, 1)
        features = features.view(features.size(0), -1, features.size(-1))  # batch_size, 7*7, 512
        return features


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha


class DecoderRNN_Attention(nn.Module):
    def __init__(self, attention_size, embed_size, hidden_size, vocab_size, num_layers, encoder_size=512, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN_Attention, self).__init__()
        self.encoder_size = encoder_size
        self.decoder_size = hidden_size
        self.vocab_size = vocab_size
        self.attention_size = attention_size
        self.attention = Attention(encoder_size, hidden_size, attention_size)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTMCell(embed_size + encoder_size, hidden_size, bias=True)
        # self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.init_h = nn.Linear(encoder_size, hidden_size)
        self.init_c = nn.Linear(encoder_size, hidden_size)
        self.f_beta = nn.Linear(encoder_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=0.5)
        self.max_seg_length = max_seq_length

    def _init_hidden_state(self, features):
        h = self.init_h(features.mean(dim=1))
        c = self.init_c(features.mean(dim=1))
        return h, c

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        batch_size, num_pixels, encoder_size = features.size()
        embeddings = self.embed(captions)
        h, c = self._init_hidden_state(features)
        decode_lengths = [c - 1 for c in lengths]
        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels)
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(features[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.lstm(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
        outputs = pack_padded_sequence(predictions, decode_lengths, batch_first=True)[0]
        return outputs, alphas

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids


class EncoderCNN_prune(nn.Module):
    def __init__(self, embed_size, layer_cfg=None):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN_prune, self).__init__()
        vgg = VGG(layer_cfg=layer_cfg)
        modules = list(vgg.children())
        self.vgg = nn.Sequential(*modules)
        self.linear = nn.Linear(25088, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        features = self.vgg(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class VGG(nn.Module):
    def __init__(self, depth=16, layer_cfg=None):
        super(VGG, self).__init__()
        self.default_layer_cfg = {
            11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        }
        if layer_cfg is None:
            layer_cfg = self.default_layer_cfg[depth]
        self.feature = self.make_layers(layer_cfg, True)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    def make_layers(self, layer_cfg, batch_norm=True):
        layers = []
        in_channels = 3
        for v in layer_cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


if __name__ == '__main__':
    vgg = models.vgg16_bn(pretrained=True)
    modules = list(vgg.children())[:-1]
    print(1)