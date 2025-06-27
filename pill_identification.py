import torch
from torch import nn
import timm
from torchvision import transforms
import pickle
from medication_matching import match_with_rx

import torch.nn.functional as F

from dis_bg_remover import remove_background
import cv2
import numpy as np
from PIL import Image

import pandas as pd

import mongo_conn

fda_df = mongo_conn.fda_df
drug_list = mongo_conn.drug_list
drugdata_df = mongo_conn.drugdata_df

# fda_df = pd.read_csv('FDA_ALL.csv')
# drug_list = pd.read_csv("Medications List Clean.csv")
# drugdata_df = pd.read_csv('drug_data.csv')

with open('pill_model/traintestsplit.pkl', 'rb') as splits:
    traintestsplit = pickle.load(splits)

class_to_label = traintestsplit['class_to_label']
label_to_class = {class_to_label[c]: c for c in class_to_label}

base_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

class ContrastiveClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super().__init__()

        self.encoder = timm.create_model('convnextv2_nano', pretrained=False)
    
        self.encoder.head.fc = nn.Identity()
        
        self.classifier_head = nn.Sequential(
            nn.Linear(640, embedding_dim),
            nn.LayerNorm(embedding_dim, eps=1e-05, elementwise_affine=True),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, num_classes, bias=True)  # Output contrastive embedding
        )
        self.projection_head = nn.Sequential(
            nn.Linear(640, embedding_dim),
            nn.LayerNorm(embedding_dim, eps=1e-05, elementwise_affine=True),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim, bias=True)  # Output contrastive embedding
        )

    def forward(self, x):
        features = self.encoder(x)
        embeddings = self.projection_head(features)
        logits = self.classifier_head(features)
        return embeddings, logits
    
OUT_FEATURES = 512
num_classes = 28
model = ContrastiveClassifier(OUT_FEATURES, num_classes)

PATH = 'pill_model/model_last.pt'
state_dict = torch.load(PATH, weights_only=False, map_location='cpu')
model.load_state_dict(state_dict['model_state_dict'])
model.eval()
print("Model Loadded.")

support_set = torch.load('pill_model/support_embeddings.pt', map_location='cpu')
mean_embeddings = support_set['mean_embeddings']
support_embeddings = torch.vstack(list(mean_embeddings.values()))
labels = list(mean_embeddings.keys())

bg_model_path = "pill_model/isnet_dis.onnx"

bg_model_path = "pill_model/isnet_dis.onnx"
input_img_path = 'test_pills\LoratadineClaritin.jpg'

def preprocess_image(input_img_path):
    img, mask = remove_background(bg_model_path, input_img_path)
                    
    # Handle potential differences in dimensions
    if img.shape[2] == 4:  # BGRA image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to BGR

    # Ensure mask is 2D
    if len(mask.shape) > 2:
        mask = mask[:,:,0]  # Take first channel if mask has multiple channels

    bg_color = 128
    # Create gray background
    h, w = img.shape[:2]
    gray_bg = np.ones((h, w, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)

    # Convert binary mask to 3-channel mask for multiplication
    # Ensure mask has values between 0 and 1
    mask = mask.astype(np.float32) / 255.0 if np.max(mask) > 1 else mask.astype(np.float32)
    mask_3channel = np.stack([mask, mask, mask], axis=2)
    # Blend foreground with gray background using the mask
    result = (img * mask_3channel + gray_bg * (1 - mask_3channel)).astype(np.uint8)
    result = Image.fromarray(result)

    return result

@torch.no_grad()
def predict(image):
    preprocessed_image_tensor = base_transform(image).unsqueeze(0)
    pred_embedding, _ = model(preprocessed_image_tensor)

    scores = F.cosine_similarity(support_embeddings, pred_embedding)
    top_values, top_indices = scores.topk(5)
    top_labels = [labels[i] for i in top_indices]
    top_predictions = [label_to_class[i] for i in top_labels]
    return top_predictions, [round(i, 2) for i in top_values.numpy().tolist()]

def get_pill_info(drug_name):
    list_entry = drug_list[drug_list['Name'] == drug_name].iloc[0]
    if 'drug_products' not in list_entry['FDA Link']:
        return 'Not Drug'
    registration_num = list_entry['FDA Link'].split("=")[-1]
    pill_entry = fda_df[fda_df['Registration Number'] == registration_num].iloc[0]
    pill_entry = dict(pill_entry)
    return pill_entry

def get_info(image_path):
    image = preprocess_image(image_path)
    predictions, scores = predict(image)
    pill_pred_info = {'matches': [], 'images': []}
    for p, s in zip(predictions, scores):
        pill_info = get_pill_info(p)
        if pill_info == 'Not Drug':
            continue
        pill_info['Score'] = s
        
        matches, _ = match_with_rx(pill_info['Generic Name'])
        rx_info = dict(drugdata_df.iloc[matches[0]])
        pill_info['rx_info'] = rx_info
        pill_info['image_url'] = f'/pill-images/{p}.jpg'

        pill_pred_info['matches'].append(pill_info)

        #pill_pred_info['images'].append(f'/pill-images/{p}.jpg')

    return pill_pred_info
    