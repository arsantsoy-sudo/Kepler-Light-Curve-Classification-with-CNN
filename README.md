<img width="712" height="413" alt="image" src="https://github.com/user-attachments/assets/1066a729-8e66-467b-90b0-0271677facdd" />


## Kepler Light Curve Classification

A 1D CNN pipeline for classifying Kepler light curves into three classes:

Planet candidate

Eclipsing binary

Non-transit object

## Dataset

The current dataset contains approximately 600 Kepler objects.

Each processed light curve has the input shape:

(1024, 1)

## Workflow

01_preprocess_fits.py — preprocesses Kepler FITS light curves.

02_build_dataset.py — builds the final dataset and data splits.

03_train_cnn.py — trains the 1D CNN and saves the accuracy and loss curves.

04_evaluate.py — saves the final test metrics and confusion matrix.

Run the scripts from the project root:

python scripts/01_preprocess_fits.py
python scripts/02_build_dataset.py
python scripts/03_train_cnn.py
python scripts/04_evaluate.py

## Pipe line

Kepler FITS files
        ↓
Preprocessing
        ↓
Dataset construction
        ↓
1D CNN training
        ↓
Evaluation

## Model

The neural network uses:

Conv1D layers

Batch normalization

Max pooling

Dropout

Global average pooling

Dense layer

Three-class softmax output

The model is trained for 80 epochs. Early stopping is not used. The checkpoint with the lowest validation loss is saved as models/best_cnn.keras.

## Results

Accuracy curve

<img width="2000" height="1200" alt="accuracy_curve" src="https://github.com/user-attachments/assets/989522a2-530c-4d46-aeb6-bb9b2d825c1b" />


Loss curve

<img width="2000" height="1200" alt="loss_curve" src="https://github.com/user-attachments/assets/7c63928b-fd31-46dc-b746-915c52cf2278" />


Confusion matrix

<img width="1400" height="1200" alt="confusion_matrix" src="https://github.com/user-attachments/assets/ea4e4389-1818-4333-b9a6-be5c6de57c0a" />


Final numerical metrics are stored in results/metrics.json.

## Project Structure

Kepler_project/
├── catalogs/
├── scripts/
│   ├── 01_preprocess_fits.py
│   ├── 02_build_dataset.py
│   ├── 03_train_cnn.py
│   └── 04_evaluate.py
├── results/
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   └── metrics.json
├── .gitignore
├── requirements.txt
├── README.md 
└── LICENSE

Raw FITS files, generated datasets, virtual environments, and trained model files are not stored in the repository.

## Author

Arsan Tsoy

## License

This project is licensed under the MIT License.

License

This project is licensed under the MIT License.
