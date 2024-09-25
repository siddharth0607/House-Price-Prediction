from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import os
import numpy as np
from fastapi import Request

app = FastAPI()
templates = Jinja2Templates(directory="templates")  # Assuming home.html is in a 'templates' folder

# Load the model
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the columns used during training
with open('X_columns.pkl', 'rb') as f:
    X_columns = pickle.load(f)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/predict")
async def predict(
    request: Request,
    location: str = Form(...),
    area: str = Form(...),
    availability: str = Form(...),
    sqft: float = Form(...),
    bath: float = Form(...),
    balcony: float = Form(...),
    bhk: float = Form(...)
):
    input_features = np.zeros(len(X_columns))
    input_features[0] = sqft
    input_features[1] = bath
    input_features[2] = balcony
    input_features[3] = bhk

    for feature in ['location', 'area', 'availability']:
        if feature in [location, area, availability] and feature in X_columns:
            index = np.where(np.array(X_columns) == feature)[0][0]
            input_features[index] = 1

    # Ensure input shape is correct
    input_features = input_features.reshape(1, -1)
    
    prediction = model.predict(input_features)[0]
    formatted_prediction = "₹{:.2f}".format(prediction * 100000)
    
    return templates.TemplateResponse("home.html", {"request": request, "prediction_text": f"The estimated house price is {formatted_prediction}."})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)