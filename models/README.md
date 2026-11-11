# Custom models

Put custom YOLOv8 weights (`.pt` files) here. They appear in the dashboard's model menus.

* **Emergency-vehicle model:** train one with `notebooks/train_emergency_kaggle_dataset.ipynb` (for example `emergency_yolov8s.pt`) and choose it under **Emergency-vehicle model**. The normal YOLOv8 model keeps counting all vehicles; this model adds ambulance, fire truck and police detection.
* The standard `yolov8n/s/m.pt` and the Faster R-CNN weights download automatically and do not need to be placed here.

Weight files are not committed to Git.
