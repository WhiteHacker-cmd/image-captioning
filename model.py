import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn.functional import tanh,softmax
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F



# Major changes include the ignoring of the last two layers. Author use a lower layer for more dense features.

class EncoderCnn(nn.Module):
    def __init__(self, encoded_size=14, train_CNN = False):
        super(EncoderCnn, self).__init__()
        self.cnn = models.resnet50(pretrained=True) # Pretrained ResNet-50
        
        self.encoded_size = encoded_size
        
        #extracting the layers needed
        #removing last adaptive pool and fc
        
        layers_to_use = list(self.cnn.children())[:-3]
        
        # Unpack and make it the conv_net
        self.cnn = nn.Sequential(*layers_to_use)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_size, encoded_size))
        
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(0.5)
        
        if not train_CNN:
            for param in self.cnn.parameters():
                param.requires_grad = False
        
        
        
    def forward(self, images):
        
        # images.shape (batch_size, 3, image_size, image_size)
        
        # Change the image_size dimensions.
        batch_size = images.shape[0]
        
        with torch.no_grad():
            features = self.cnn(images) # Shape : (batch_size, encoder_dim, image_size/32, image_size/32)
            
            features = self.adaptive_pool(features)# Shape (batch_size, encoder_dim, encoed_size, encoded_size)
            
            features = features.permute(0, 2, 3, 1) # Shape : (batch_size, encoded_size, encoded_size, encoder_dim)
            
            # The above transformation is needed because we are going to do some computation in the 
            # decoder.
            
            encoder_dim = features.shape[-1]
            features = features.view(batch_size, -1, encoder_dim)  # (batch_size, L, D)
            
            return features
        



class BahdanauAttention(nn.Module):

    
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        
        self.attention = nn.Linear(attention_dim, 1)
        
        self.softmax = nn.Softmax(dim=1)
        
        self.relu = nn.ReLU()
        
        self.encoder_to_attention_dim = nn.Linear(encoder_dim, attention_dim)
        
        self.decoder_to_attention_dim = nn.Linear(decoder_dim, attention_dim)
        
        self.dropout = nn.Dropout(0.5)
        
        self.tanh = nn.Tanh()
        
        
    def forward(self, encoder_output, hidden_states):
    
        encoder_attention = self.encoder_to_attention_dim(encoder_output) # (batch_size, L, attention_dim)

        decoder_attention = self.decoder_to_attention_dim(hidden_states) # (batch_size, attention_dim)


        #   (batch_size, L, attention_dim) + (batch_size, 1, attention_dim) 
        encoder_decoder = encoder_attention + decoder_attention.unsqueeze(1)  # (batch_size, L, attention_dim)

        encoder_decoder = self.tanh(encoder_decoder)

        attention_full = (self.attention(encoder_decoder)).squeeze(2) # (batch_size, L)

        alpha = self.softmax(attention_full) # Take the softmax across L(acc to paper)

        z = (encoder_output * alpha.unsqueeze(2) ).sum(dim = 1) # Sum across L (pixels)

        return z, alpha
    



class Decoder(nn.Module):
   
    def __init__(self,encoder_dim, decoder_dim, embed_size, vocab_size, attention_dim, device, dropout = 0.5):
        super(Decoder, self).__init__()
        
        
        
        self.device = device
        
        self.embed = nn.Embedding(vocab_size, embed_size) # Embedding layer courtesy Pytorch
        
        self.encoder_dim = encoder_dim
        
        self.decoder_dim = decoder_dim
        
        self.attention_dim = attention_dim
        
        self.embed_dim = embed_size
        
        self.vocab_size = vocab_size
        
        self.dropout = dropout
        
        
        self.lstm = nn.LSTMCell(self.encoder_dim + self.embed_dim, self.decoder_dim, bias=True)
        
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        
        self.f_beta = nn.Linear(self.decoder_dim, self.encoder_dim)
        
        self.sigmoid = nn.Sigmoid()
        
        # See the paper 
        
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()
        
    # Encoder output is the annotated vector a (L,D) in the paper

    def initialise_hidden_states(self, encoder_output):
       
        # encoder_output : shape (batch_size, L, encoder_dim=D)

        mean = (encoder_output).mean(dim = 1) # Take mean over L

        # Pass through Fully connected

        c_0 = self.init_c(mean)
        c_0 = self.tanh(c_0)

        h_0 = self.init_h(mean)
        h_0 = self.tanh(h_0)
        return h_0, c_0
        
    def init_weights(self):

        

        self.embed.weight.data.uniform_(-0.1,0.1)
        self.fc.weight.data.uniform_(-0.1,0.1)
        self.fc.bias.data.fill_(0)
            
            
    def forward(self, encoder_output, caption, caption_lengths):

        
        device = self.device
        batch_size = encoder_output.size(0)

        # num_pixels 
        L = encoder_output.size(1)

        max_caption_length = caption.shape[-1] # shape : (batch_size, max_caption)

        
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_output = encoder_output[sort_ind]
        caption = caption[sort_ind]

       

        
        lengths = [l - 1 for l in caption_lengths]

        embedding_of_all_captions = self.embed(caption)

        predictions = torch.zeros(batch_size, max_caption_length - 1, self.vocab_size).to(device)

        alphas = torch.zeros(batch_size, max_caption_length - 1, L).to(device)

       

        h, c = self.initialise_hidden_states(encoder_output)

       
        for t in range(max_caption_length - 1):

            batch_size_t = sum([l > t for l in lengths])
            context_vector, alpha = self.attention(encoder_output[:batch_size_t], h[:batch_size_t])

            
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))

            gated_context = gate * context_vector

            # context_vector : torch.Size([32, 1024]), embedded_caption_t : torch.Size([32, 256])
            h, c = self.lstm(torch.cat([embedding_of_all_captions[:batch_size_t,t,:], gated_context], dim=1),(h[:batch_size_t], c[:batch_size_t]))
            predict_deep = self.deep_output_layer(embedding_of_all_captions[:batch_size_t,t,:], h, context_vector)

            predictions[:batch_size_t, t, :] = predict_deep 
            alphas[:batch_size_t, t, :] = alpha
            
        return predictions, alphas, caption, lengths
    
        
    def deep_output_layer(self, embedded_caption, h, context_vector):
      
        dropout = nn.Dropout(0.2)
        scores = dropout(self.fc(h))
        return scores
        
    def predict_caption(self, encoder_output, captions):
        # "<SOS>" 1
        caption_list = [1]
        alphas = [] 
        h, c = self.initialise_hidden_states(encoder_output)
        # 2 is <EOS>
        while len(caption_list) < 40 :
            word = caption_list[-1]
            embedded_caption = self.embed(  torch.LongTensor([word]).to(self.device)  )  # (1, embed_dim)

            context_vector, alpha = self.attention(encoder_output, h)

            gate = self.sigmoid(self.f_beta(h))

            gated_context = gate * context_vector

            h, c = self.lstm(torch.cat([embedded_caption, gated_context], dim=1), (h, c))

            predictions = self.deep_output_layer(embedded_caption, h, context_vector)  # (1, vocab_size)

           

            next_word = (torch.argmax(predictions, dim=1, keepdim=True).squeeze()).item()

            caption_list.append(next_word)

            alphas.append(alpha)

            if(caption_list[-1] == 2):
                break
        return caption_list, alphas
    def beam_search(self, encoder_output, beam_size = 3):

        device = self.device

        k = beam_size

        vocab_size = self.vocab_size

        encoder_size = encoder_output.size(-1)

        encoder_output = encoder_output.view(1, -1, encoder_size)

        num_pixels = encoder_output.size(1)

        
        encoder_output = encoder_output.expand(k, num_pixels, encoder_size)  # (k, num_pixels, encoder_dim)

        # Vocab.stoi(SOS)
        k_prev_words = torch.LongTensor([[1]] * k).to(device) 
        seqs = k_prev_words

        # To store the scores at each stage
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Store sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        step = 1


        h, c = self.initialise_hidden_states(encoder_output)

        while True:

            # Get embedding
            embedded_caption = self.embed(k_prev_words).squeeze(1)

            # Pass through the attention network
            context_vector, alpha = self.attention(encoder_output, h);

            gate = self.sigmoid(self.f_beta(h))

            gated_context = gate * context_vector

            h, c = self.lstm(torch.cat([embedded_caption, gated_context], dim=1), (h, c))

            # Used normal Fully connected layer instead with little dropout.
            scores = self.deep_output_layer(embedded_caption, h, context_vector)

            scores = F.log_softmax(scores, dim=1)

            scores = top_k_scores.expand_as(scores) + scores

            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, dim=0)  # (s)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, dim=0)  # (s)

            prev_word_inds = torch.true_divide(top_k_words , vocab_size).long().cpu()  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

           
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != 2 ] #vocab.itos['<EOS>']
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])

            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break

            seqs = seqs[incomplete_inds]

            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]

            encoder_output = encoder_output[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

           
            if step > 50:
                break
            step += 1


        if len(complete_seqs_scores) > 0:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
            return seq
        else:
            return [1,2] # If cannot predict <EOS> in the beginning of training because I was checking an untrained network, very dumb of me.
        return complete_seqs
    




