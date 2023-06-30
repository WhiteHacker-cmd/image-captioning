import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from nltk.translate.bleu_score import sentence_bleu


from model import (Decoder, EncoderCnn)
from vocabulary import Vocabulary
from utils import load_captions


captions = load_captions()


# Initialize vocabulary and build vocab
vocab = Vocabulary(5)
vocab.build_vocabulary(captions)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





embed_size = 256
encoder_dim = 1024
decoder_dim = 400
attention_dim = 400
vocab_size = 2548




transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])



encoder = EncoderCnn()
encoder.load_state_dict(torch.load("weights/encoder-2_state_dict.pt", map_location=device))
encoder = encoder.to(device)
encoder.eval()




decoder = Decoder(encoder_dim=encoder_dim, decoder_dim=decoder_dim, embed_size=embed_size, vocab_size=vocab_size,attention_dim=attention_dim, device=device)
decoder.load_state_dict(torch.load('weights/decoder-2_state_dict.pt', map_location=device))
decoder = decoder.to(device)
decoder.eval()



max_length = 40
attention_features_shape = 196

def evaluate(img):
    attention_plot = np.zeros((max_length, attention_features_shape))
    img = Image.open(img).convert("RGB")
    img = transform(img).to(device)
    img = img.view(1, 3, 224, 224)

    encder_output = encoder(img)
    dec_input = torch.unsqueeze(torch.tensor([vocab.stoi['<SOS>']]), 0)
    result = []
    caption_lengths = torch.tensor([1,]).long().to(device)

    h, c = decoder.initialise_hidden_states(encder_output)

    for i in range(max_length):
        dec_input = dec_input.to(device)
        embedded_caption = decoder.embed(dec_input).squeeze(1)
        context_vector, alpha = decoder.attention(encder_output, h)

        gate = decoder.sigmoid(decoder.f_beta(h))
        gated_context = gate * context_vector
        h, c = decoder.lstm(torch.cat([embedded_caption, gated_context], dim=1), (h, c))
        predictions = decoder.deep_output_layer(embedded_caption, h, context_vector)
        
        attention_plot[i] = alpha.view(-1, ).cpu().detach().numpy()
        predicted_id = (torch.argmax(predictions, dim=1, keepdim=True).squeeze()).item()
        result.append(vocab.itos[predicted_id])

        if vocab.itos[predicted_id] == '<EOS>':
            return result, attention_plot

        dec_input = torch.unsqueeze(torch.tensor([predicted_id]), 0)
    attention_plot = attention_plot[:len(result), :]

    return result, attention_plot




def plot_attention(image, result, attention_plot):
   temp_image = np.array(Image.open(image))
   fig = plt.figure(figsize=(10, 10))
   len_result = len(result)
   for l in range(len_result):
       temp_att = np.resize(attention_plot[l], (8, 8))
       ax = fig.add_subplot(len_result//2, len_result//2, l+1)
       ax.set_title(result[l])
       img = ax.imshow(temp_image)
       ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

   plt.tight_layout()
   plt.show()




# captions on the validation set
image = 'test_images/1000268201_693b08cb0e.jpg'

# real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
result, attention_plot = evaluate(image)

real_caption = 'A child in a pink dress is climbing up a set of stairs in an entry way .'

#remove "<unk>" in result
for i in result:
   if i=="<UNK>":
       result.remove(i)

for i in real_caption:
   if i=="<UNK>":
       real_caption.remove(i)

#remove <end> from result        
result_join = ' '.join(result)
result_final = result_join.rsplit(' ', 1)[0]

real_appn = []
real_appn.append(real_caption.split())
reference = real_appn
candidate = result

score = sentence_bleu(reference, candidate)
print(f"BELU score: {score*100}")

print ('Real Caption:', real_caption)
print ('Prediction Caption:', result_final)
plot_attention(image, result, attention_plot)