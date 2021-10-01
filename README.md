# Easy ML
Quick and easy classification modeling with scikit-learn and pandas

## Usage
```
from easyml.modeler import Modeler

modeler = Modeler(df, features, label, model, verbose=True)
modeler.fit()
```

## Key ideas
- Quickly iterate with new features
- Experiments (including features, data, and metrics) are pickleable for easy reproducibility
- Basic metrics out of the box
- Track train, test, and full dataset metrics
