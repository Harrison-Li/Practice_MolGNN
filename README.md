# Practice_GNN4MOL

This repository is my practice project focused on applying Graph Neural Networks (GNNs) to molecular property prediction.



Compared to using SMILES strings or ECFP molecular fingerprints, GNNs offer a more robust way to capture the hidden molecular structure correlation by directly modeling atoms and bonds as nodes and edges in a graph. This graph representation allows GNN to better learn  chemical properties from molecular data.

## How to use the code
You can train it based on your own dataset by adjusting the code line bellow:
```bash
nohup python -m train.train --data_file train_data --props logp --batch_size 100 --max_epochs 10 --learning_rate 1e-5 >> ./logp.log 2>&1 &
```

Evaluation codes is stored in the notebook
## Result

### Gradient boosting

![gb](assets/gb.png)

### MLP

![mlp](assets/mlp.png)

### Random Forest

![rf](assets/rf.png)

### My MolGNN

![gb](assets/MolGNN.png)