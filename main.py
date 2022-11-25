#---- 0. Setup

from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel

#---- 1. Iniciando o app

app = FastAPI()

#---- 2. Carregando o modelo

model = load('models/hr_tuning_model_lgbm.joblib') # Modelo carregado

#---- 3. Carregando as variáveis/features: Request Body

class request_body(BaseModel):
    enrollee_id : int
    city : str
    city_development_index : float
    gender : str
    relevent_experience : str
    enrolled_university : str
    education_level : str
    major_discipline : str
    experience : str
    company_size : str
    company_type : str
    last_new_job : str
    training_hours : int

#---- 4. Endpoint: Predição do modelo de ML 

@app.post('/predict')
def predict(data : request_body):
    test_data = [[
            data.enrollee_id, 
            data.city, 
            data.city_development_index, 
            data.gender,
            data.relevent_experience,
            data.enrolled_university,
            data.education_level,
            data.major_discipline,
            data.experience,
            data.company_size,
            data.company_type,
            data.last_new_job,
            data.training_hours

    ]]
    class_idx = model.predict(test_data)[0]
    return { 'class' : iris.target_names[class_idx]}



@app.get("/")
def hello_world_root():
    return 'Bem-vind@ a página de predições do modelo'


# @app.post("/predict_hr")


# Carregar as features
# Predição
# Mudança de label na resposta: 0 -> Não quer trocar de emprego e 1: Quer trocar de emprego

