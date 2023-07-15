import mlflow.pytorch
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

app = FastAPI(
    title="ML Models as API",
    description="Deploy ML Models as API on Lightsail",
    version="1.0",
)

model = None
device = None
transform = None


@app.on_event("startup")
def load_model():
    global model
    global device
    global transform

    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/final_datasci")
    run_id = "891a0daf33b9454e8f7b4a3efea937b6"
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pytorch.load_model(model_uri, map_location=torch.device("cpu"))
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize((230, 230)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@app.post("/api", tags=["prediction"])
async def get_predictions(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        input = transform(img)
        input = input.unsqueeze_(0)
        input = input.to(device)
        output = model(input)

        road_types = {0: "not bad", 1: "bad", 2: "very bad"}

        percent_predict = output.amax(dim=1).cpu().detach().numpy()[0]

        if percent_predict < 0.7:
            return {"prediction": "undefined"}
        else:
            return {"prediction": road_types[output.argmax().item()]}
    except:
        return {"prediction": "error"}
