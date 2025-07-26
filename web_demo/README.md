This is a demonstration of session-based recommendation system via a simple e-commerce web application, using GRU4Rec.

{project-name}/
├── app.py                  ← Flask backend
├── templates/
│   └── index.html          ← Product list + click handler
├── static/
│   └── style.css           ← Optional styling
├── model/
│   ├── gru_model.py        ← GRU4Rec model definition
│   └── gru_model.pt        ← Saved model weights
├── utils/
│   └── preprocess.py       ← Sequence encoder
└── data/
    └── interactions.csv    ← User logs