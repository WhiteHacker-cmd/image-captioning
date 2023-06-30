import torch
from PIL import Image
from torchvision import transforms
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


class ImgCModel:
    def __init__(self):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.encoder = EncoderCnn()
        # self.encoder.load_state_dict(torch.load("encoder-2_state_dict.pt", map_location=device))
        # self.encoder = encoder.to(device)
        # self.encoder.eval()


        # self.decoder = Decoder(encoder_dim=encoder_dim, decoder_dim=decoder_dim, embed_size=embed_size, vocab_size=vocab_size,attention_dim=attention_dim, device=device)
        # self.decoder.load_state_dict(torch.load('decoder-2_state_dict.pt', map_location=device))
        # self.decoder = decoder.to(device)
        # self.decoder.eval()

        # self.transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


        # self.vocab = Vocabulary(5)
        # self.vocab.build_vocabulary(captions)
        pass





    def predict(self, img):
        img = Image.open(img).convert("RGB")
        img = transform(img).to(device)
        img = img.view(1, 3, 224, 224)

        encdoer_output = encoder(img)
        c = decoder.beam_search(encdoer_output)
        return " ".join([vocab.itos[x] for x in c][1:-1])


