from http import HTTPStatus
from logging import INFO

import gridfs
import uvicorn
from flwr.common.logger import log

from config import PORT, HOST, DB_PORT
from fastapi import BackgroundTasks
from fastapi import FastAPI, status, UploadFile, File, Response, HTTPException
from pymongo import MongoClient

import src.local_clients
from application.config import DATABASE_NAME
from application.utils import formulate_id
from pydloc.models import LOTrainingConfiguration, MLModel
from multiprocessing import Process, Array, Value, Queue

app = FastAPI()


# Receive configuration for training job
@app.post("/job/config/{id}")
def receive_updated(id, data: LOTrainingConfiguration, background_tasks:BackgroundTasks):
    try:
        losses = Array("f", [0]*50)
        accuracies = Array("f", [0]*50)
        times = Array("f", [0]*50)
        queue = Queue()
        queue_loss = Queue()
        queue_cluster = Queue()
        priority = Value("i", 0)
        processes = []
        placed_id = formulate_id(config=data)
        if placed_id not in src.local_clients.current_jobs:
            src.local_clients.current_jobs[placed_id] = 1
        else:
            src.local_clients.current_jobs[placed_id] += 1
        for i in range(data.num_clusters):
            client_process = Process(target=src.local_clients.start_client, args=(id,
                                                                                  i,
                                                                                  data,
                                                                                  losses,priority,queue,queue_loss,queue_cluster,
                                                                                  accuracies, times))
            client_process.start()
            log(INFO, f"processes on client {i}")
            processes.append(client_process)
        for p in processes:
            p.join()
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
    return src.local_clients.current_jobs


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
