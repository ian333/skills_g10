from fastapi import FastAPI,HTTPException,status,File,UploadFile
import pickle
from Schemas.IrisSchema import IrisSchema
from PIL import Image
import numpy as np
import io
from fastapi.responses import FileResponse,StreamingResponse



app = FastAPI()


@app.get("/",status_code=status.HTTP_200_OK)
async def root():
    return {"message": "Hello daniel y henry"}

@app.post("/predict",status_code=status.HTTP_200_OK)
async def predict_flower(data:IrisSchema):
    """
    Esta funcion nos predecira si es una setosa ,virginica o versicolor

    Clase Iris:

    class IrisSchema(BaseModel):
    
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    """
    ModPredLR=pickle.load(open("Modelos/ModPredLR.pkl",'rb'))
    prediccion=ModPredLR.predict([[data.sepal_length,data.sepal_width,data.petal_length,data.petal_width]])
    if int(prediccion[0])==0:
        return {"prediccion":"setosa"}
    elif int(prediccion[0])==1:
        return {"prediccion":"versicolor"}
    else :
        return {"prediccion":"virginica"}




@app.post("/predict/svc",status_code=status.HTTP_200_OK)
async def predict_flower_svc(data:IrisSchema):

    """
    Esta funcion nos predecira si es una setosa ,virginica o versicolor

    Clase Iris:

    class IrisSchema(BaseModel):
    
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    """
    ModPredLR=pickle.load(open("Modelos/ModPredLR.pkl",'rb'))
    #ModPredLR.predict([[3,4,5,10]])
    prediccion=ModPredLR.predict([[data.sepal_length,data.sepal_width,data.petal_length,data.petal_width]])
    if int(prediccion[0])==0:
        return {"prediccion":"setosa"}
    elif int(prediccion[0])==1:
        return {"prediccion":"versicolor"}
    else :
        return {"prediccion":"virginica"}
    

@app.post("/predict/lr",status_code=status.HTTP_200_OK)
async def predict_flower_lr(data:IrisSchema):

    """
    Esta funcion nos predecira si es una setosa ,virginica o versicolor

    Clase Iris:

    class IrisSchema(BaseModel):
    
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    """
    ModPredLR=pickle.load(open("Modelos/ModPredLR.pkl",'rb'))
    #ModPredLR.predict([[3,4,5,10]])
    prediccion=ModPredLR.predict([[data.sepal_length,data.sepal_width,data.petal_length,data.petal_width]])
    if int(prediccion[0])==0:
        return {"prediccion":"setosa"}
    elif int(prediccion[0])==1:
        return {"prediccion":"versicolor"}
    else :
        return {"prediccion":"virginica"}


@app.post("/uploadfile/",status_code=status.HTTP_200_OK)
async def create_upload_file(file:UploadFile=File(...)):
    ModPredMNISTSVC=pickle.load(open("Modelos/ModPredMNISTSVC.pkl",'rb'))

    image = Image.open(file.file)
    image=image.convert('L')
    image=image.resize((8,8))
    image.save("Images/img" + str(2) + ".jpg")

    image=np.array(image)/30
    print(image)
    image_predict=image.reshape(1,-1)
    prediction=ModPredMNISTSVC.predict(image_predict)
    prediction_number=int(prediction[0])
    
    # buffer= io.BytesIO()
    # Image.fromarray((image * 255).astype(np.uint8)).save(buffer, format='png')
    # buffer.seek(0)

    return {"prediction_number":prediction_number}

