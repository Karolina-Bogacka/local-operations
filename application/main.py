from http import HTTPStatus
from logging import INFO

import gridfs
import uvicorn
from flwr.client import start_numpy_client
from flwr.common.logger import log

from application.src.big_client import BigClient
from config import PORT, HOST, DB_PORT
from fastapi import BackgroundTasks
from fastapi import FastAPI, status, UploadFile, File, Response, HTTPException
from pymongo import MongoClient


from application.config import DATABASE_NAME
from application.utils import formulate_id
from pydloc.models import LOTrainingConfiguration, MLModel

app = FastAPI()
big_client: BigClient = None

# Receive configuration for training job
@app.post("/job/big/config/{id}")
def receive_big_client_updated(id, data: LOTrainingConfiguration, background_tasks:
BackgroundTasks):
    try:
        global big_client
        log(INFO, f"Connect big client to main server {data.server_address}:8080")
        big_client = BigClient(data)
        start_numpy_client(server_address=f"{data.server_address}:8080",
                           client=big_client)
    except Exception as e:
        print("An exception occurred ::", e)
        return 500
    return 200

# Receive configuration for training job
@app.post("/job/small/config/{id}")
def receive_middle_updated(id, data: LOTrainingConfiguration, background_tasks:
BackgroundTasks):
    try:
        log(INFO, f"{big_client}")
        log(INFO, f"Connect big client to main server {data.server_address}:8080")
        big_client.start_small_client(data)
    except Exception as e:
        print("An exception occurred ::", e)
        return 500
    return 200


# Receive  new shared model configuration
@app.post("/model/")
def receive_conf(model: MLModel):
    try:
        client = MongoClient(DATABASE_NAME, DB_PORT)
        db = client.local
        db.models.insert_one(model.dict(by_alias=True))
    except Exception as e:
        print("An exception occurred ::", e)
        return 500
    return 200


# Receive new model file
@app.put("/model/{id}/{version}", status_code=status.HTTP_204_NO_CONTENT)
async def update_model(id: int, version: int, file: UploadFile = File(...)):
    client = MongoClient(DATABASE_NAME, DB_PORT)
    db = client.repository
    db_grid = client.repository_grid
    fs = gridfs.GridFS(db_grid)
    if len(list(db.models.find({'id': id, 'version': version}).limit(1))) > 0:
        data = await file.read()
        model_id = fs.put(data, filename=f'model/{id}/{version}')
        db.models.update_one({'id': id, 'version': version}, {"$set": {"model_id": str(model_id)}},
                             upsert=False)
        return Response(status_code=HTTPStatus.NO_CONTENT.value)
    else:
        raise HTTPException(status_code=404, detail="model not found")


# Returns statuses of currently running jobs by returning information
# about the number of model ids and versions being ran
@app.post("/job/status")
def retrieve_status():
    return {}


if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT)
