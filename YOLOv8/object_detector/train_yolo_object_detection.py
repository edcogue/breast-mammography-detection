import torch
from hyperopt import fmin, tpe, hp, Trials, space_eval
from ultralytics import YOLO

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

def objective(params):
    # Define el modelo con los parámetros dados
    model = YOLO().load('yolov8n.pt')
    # Entrena el modelo
    model.train(data="/tf/data/Mammographies/yolo_data/data_object_detection.yaml",
                epochs=5,
                dropout=params["dropout"],
                lr0=params["lr"],   
                amp=False,             
                batch=1,
                device=0,
                augment=True,
                verbose=False,
                imgsz=1600)
    
    # Evalúa el modelo
    metrics = model.val()
    
    # Devuelve la métrica a optimizar
    return -metrics.results_dict["metrics/recall(B)"]

# Define los rangos de los hiperparámetros
space = {
    'dropout': hp.choice('dropout',[0.0, 0.2, 0.4]),
    'lr': hp.loguniform('lr', -6, -1),
}

trials = Trials()

# Realiza la búsqueda de hiperparámetros
best = fmin(objective, space, algo=tpe.suggest, max_evals=10, trials=trials)

print(trials.trials)

best = space_eval(space, best)

print(best)


# Define el modelo con los parámetros dados
model = YOLO().load('yolov8n.pt')

# Entrena el modelo
model.train(data="/tf/data/Mammographies/yolo_data/data_object_detection.yaml",
            epochs=100,
            patience=10,
            amp=False,
            dropout=best["dropout"],
            lr0=best["lr"],                
            batch=1,
            augment=True,
            imgsz=1600)    

metrics = model.val()

print(metrics)

model.export()
