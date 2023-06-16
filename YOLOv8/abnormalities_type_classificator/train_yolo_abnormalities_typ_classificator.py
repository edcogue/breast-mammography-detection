import torch
from hyperopt import fmin, tpe, hp, Trials, space_eval
from ultralytics import YOLO

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

def objective(params):
    # Define el modelo con los parámetros dados
    model = YOLO('yolov8n-cls.pt')
    
    # Entrena el modelo
    model.train(data="/tf/data/Mammographies/yolo_data/abnormality_classification",
                task="classify",
                epochs=10,
                amp=False, 
                dropout=params["dropout"],
                lr0=params["lr"],                
                batch=2,
                device=0,
                augment=True,
                imgsz=320)
    
    # Evalúa el modelo
    metrics = model.val()
    
    # Devuelve la métrica a optimizar
    return -metrics.results_dict["metrics/accuracy_top1"]

# Define los rangos de los hiperparámetros
space = {
    'dropout': hp.choice('dropout',[0.0, 0.2, 0.4]),
    'lr': hp.loguniform('lr', -6, -1),
}

trials = Trials()

# Realiza la búsqueda de hiperparámetros
best = fmin(objective, space, algo=tpe.suggest, max_evals=15, trials=trials)

print(trials.trials)

best = space_eval(space, best)

print(best)

# Define el modelo con los parámetros dados
model = YOLO('yolov8n-cls.pt')

# Entrena el modelo
model.train(data="/tf/data/Mammographies/yolo_data/abnormality_classification",
            epochs=100,
            patience=10,
            dropout=best["dropout"],
            lr0=best["lr"],  
            amp=False,              
            batch=2,
            device=0,
            augment=True,
            imgsz=320)

metrics = model.val()

print(metrics)

model.export()