import sys
sys.path.append('/home/administrator/PycharmProjects/clip_sg_decoder/')

from argparse import ArgumentParser, Namespace
from glob import glob
from typing import List, Tuple, Dict
import os

import clip
import torch
from torch import nn, Tensor
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer


from models import clip_txt_encoder, get_txt_encoder
from sentence_transformers import SentenceTransformer



def RDM(txt_encoder: nn.Module, strings: List[str]) -> Tuple[Tensor, Tensor]:
    all_emb = []
    for s in strings:
        s = s.replace('_', ' ')
        # s = '<|endoftext|>' + s
        # s = s + '<|endoftext|>'
        emb = txt_encoder.encode_text([s]).squeeze(0)
        all_emb.append(emb)
    all_emb = torch.stack(all_emb)
    all_emb = all_emb / all_emb.norm(dim=1, p=2)[:, None]
    cos_dist = 1 - torch.mm(all_emb, all_emb.transpose(0, 1))
    l2_dist = torch.cdist(all_emb, all_emb)
    return cos_dist, l2_dist


def heatmap(txt_encoder: nn.Module, strings: List[str], Title: str, colors: List[str], num_names: int, num_places: int, num_objects: int, fig, idx: int) -> None:
    cos_dist, l2_dist = RDM(txt_encoder, txt_input)
    # cos_df = pd.DataFrame(cos_dist.cpu().detach().numpy(), columns=txt_input, index=txt_input)
    l2_df = pd.DataFrame(l2_dist.cpu().detach().numpy(), columns=txt_input, index=txt_input)
    # px.imshow(cos_df, title=f'{Title} cos-distance').show()
    # px.imshow(l2_df, title=f'{Title} L2-distance').show()
    x_embedded = TSNE(n_components=2, metric='precomputed', learning_rate='auto').fit_transform(l2_df)
    x_embedded = pd.DataFrame(x_embedded, index = strings, columns=['x', 'y'])
    imagenet = x_embedded.iloc[:num_objects]
    places = x_embedded.iloc[num_objects:num_objects+num_places]
    names = x_embedded.iloc[num_objects+num_places:num_objects+num_places + num_names]
    traits = x_embedded.iloc[num_objects+num_places + num_names:]
    fig.add_trace(go.Scatter(x=imagenet['x'], y=imagenet['y'], mode="markers+text", textposition="bottom center", name='ImageNet', marker={"color": 'purple'}), row=idx, col=2)
    fig.add_trace(go.Scatter(x=places['x'], y=places['y'], mode="markers+text", textposition="bottom center", name='Places365', marker={"color": 'red'}), row=idx, col=2)
    fig.add_trace(go.Scatter(x=names['x'], y=names['y'], mode="markers+text", textposition="bottom center", name='Names', marker={"color": 'blue'}), row=idx, col=2)
    fig.add_trace(go.Scatter(x=traits['x'], y=traits['y'], mode="markers+text", textposition="bottom center", name='Traits', marker={"color": 'green'}), row=idx, col=2)
    fig.add_trace(go.Heatmap(z=l2_df, x=strings, y=strings, showscale=False), row=idx, col=1)
    # fig.update_layout(title_text=f'{Title} L2-distance')
    
    # px.scatter(x_embedded, x='x', y='y', color=colors).show()



def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('model', type=str, help='The architecture to use to encode the text')
    parser.add_argument('--reduction', default=None, type=str, help='How to reduce the representations')
    return parser.parse_args()



if __name__ == '__main__':
    args = get_args()
    with torch.no_grad():
        places365 = pd.read_csv('/home/ssd_storage/experiments/clip_decoder/places365_cls_names.csv', index_col='class')
        imagenet = pd.read_csv('/home/administrator/Documents/imagenet_chosen_classes.csv', index_col='cls')
        people = ['Ray Romano', 'Ray_Romano', 'Tom Hardy', 'Tom_Hardy', 'AKON', 'akon', 'Akon']
        hair = ['Blonde', 'Red hair', 'Ginger', 'Black hair', 'Bald', 'Grey hair', 'Beard']
        ethnicity = ['African american', 'Indian', 'Jewish', 'Caucasian', 'Asian']
        gender = ['Man', 'Woman']
        joined = ['African american woman', 'African man', 'Blonde woman', 'Brunette man']
        eyes = ['Blue eyes', 'Brown eyes', 'Green eyes', 'Sunglasses', 'Glasses']
        behavioral = ['Smiling', 'Laughing', 'Frowning', 'crying', 'Happy', 'Sad', 'Angry', 'Mad']
        personality = ['Nice', 'Mean', 'Smart', 'Dumb', 'Good', 'Bad']
        celebA = [os.path.basename(pth) for pth in glob('/home/ssd_storage/datasets/celebA/*')]
        celebA.remove('celebA_names_list.txt')
        uri = ['Man with red hair and brown eyes', 'Man with brown eyes and red hair', 'Woman with red hair and brown eyes', 'Woman with brown eyes and red hair']
        characteristics = ['big nose', 'short forehead', 'fat', 'big ears', 'big eyes', 'thick eyebrows']
        names = ['Donald_Trump', 'Barack Obama', 'angela_rayner', 'angelina_jolie', 'anthony_hopkins', 'bill_clinton', 'boris_johnson', 'david_cameron', 'donald_trump', 'esther_mcvey', 'george_W_bush', 'hillary_clinton', 'Hugh_Grant','jennifer_aniston','Judi_Dench','Kate_Winslet','keira_knightley','liam_neeson','Martin_Freeman','michael_caine','nicolas_cage','nicola_sturgeon','priti_patel','robert_de_niro','sandra_bullock','theresa_may','tom_hanks']

        all_names = names + celebA + people
        all_traits = eyes + hair + ethnicity + gender + joined + uri + characteristics + personality + behavioral
        places365 = list(places365['name'])
        imagenet = list(imagenet.sample(300)['content'])
        txt_input = imagenet + places365 + all_names + all_traits

        colors = ['name' for name in all_names] + ['trait' for trait in all_traits]
        fig = make_subplots(rows=1, cols=2)
        fig.update_layout(height=600*2, width=1400, title_text=f'Text space TSNE L2-distance')
        if args.model in clip.available_models():
            clip_model, _ = clip.load(args.model)
            clip_txt = clip_txt_encoder(clip_model)
            heatmap(clip_txt, txt_input, f'CLIP {args.model}', colors, len(all_names), len(places365), len(imagenet), fig, 1)
        else:
            model = get_txt_encoder(args.model, args.reduction)
            # model = SentenceTransformer('all-mpnet-base-v2')
            heatmap(model, txt_input, args.model, colors, len(all_names), len(places365), len(imagenet), fig, 1)

        # model = SentenceTransformer('all-mpnet-base-v2')
        # heatmap(model, txt_input, args.model, colors, len(all_names), len(places365), len(imagenet), fig, 1)
        
        # roberta_mean = get_txt_encoder('roberta-large', 'mean')
        # roberta_first = get_txt_encoder('roberta-large', 'first')
        # heatmap(roberta_mean, txt_input, 'Roberta mean', colors, len(all_names), len(places365), len(imagenet), fig, 3)
        # heatmap(roberta_first, txt_input, 'Roberta first', colors, len(all_names), len(places365), len(imagenet), fig, 4)
        
        # bert_mean = get_txt_encoder('bert-large-uncased', 'mean')
        # bert_first = get_txt_encoder('bert-large-uncased', 'first')
        # heatmap(bert_mean, txt_input, 'BERT mean', colors, len(all_names), len(places365), len(imagenet), fig, 5)
        # heatmap(bert_first, txt_input, 'BERT first', colors, len(all_names), len(places365), len(imagenet), fig, 2)
        fig.update_layout(height=600*1, width=1200, title_text=f'{args.model} L2-distance')
        fig.show()
        # fig.write_html(f'/home/ssd_storage/experiments/clip_decoder/txt_distributions/L2_txt_tsne_{args.model.replace("/", "")}_{args.reduction}.html')
        
        
        

        