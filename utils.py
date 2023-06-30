def load_fp(filename):
      #open the file to read
      with open(filename, 'r') as file:
        text = file.read()
      return text



def load_captions():

    captions = []

    
    img_names = load_fp('caption_files/Flickr_8k.trainImages.txt')

    text_data = load_fp("caption_files/Flickr8k.token.txt")
    captions_list = text_data.split('\n')
    

    for caption in captions_list[:-1]:
        img, caption = caption.split('\t')
        if img[:-2] in img_names:
            captions.append(caption)

    return captions
