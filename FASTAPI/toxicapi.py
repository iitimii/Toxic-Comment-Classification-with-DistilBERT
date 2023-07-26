from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
import pickle
import torch
import torch.nn as nn
import transformers
from transformers import DistilBertTokenizer, DistilBertModel

app = FastAPI()


class DistilBERTClass(nn.Module):
    
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768, 6)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


device ='cpu'
MAX_LEN = 512

toxic_model = DistilBERTClass().to(device)
toxic_model.load_state_dict(torch.load('FASTAPI/models/model_0_018.pth', map_location=torch.device('cpu')))
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)


class UserInput(BaseModel):
    prompt: str 


@app.get("/")
def index():
    return {"message": "Hello"}

@app.get('/{name}')
def get_name(name:str):
    return {'Hello': f'{name}'}

@app.post('/predict')
def predict_comment(data:UserInput):
    data = data.dict()
    comment = data['prompt']
    with torch.no_grad():
        inputs = tokenizer.encode_plus(comment, None, add_special_tokens=True, max_length=MAX_LEN,
                                    pad_to_max_length=True, return_token_type_ids=True,)
        output = {
        'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
        'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
        'token_type_ids': torch.tensor(inputs["token_type_ids"], dtype=torch.long)
         }
        y_pred = nn.Sigmoid()(toxic_model(output['ids'], output['mask'], output['token_type_ids']))

    y_pred = (y_pred > 0.3)
    y_pred = y_pred.tolist()
    cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    return {"classes":cols, "prediction": y_pred}



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1" , port=8000)
                
#uvicorn toxicapi:app --reload

