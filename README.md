# Het-Net-GNN

Repository Implementing **Heterogeneous Network Graph Neural Network** (HetNetGNN) on Wireless Cell-Free Network.

## Requirements
```commandline
python==3.9.16
matplotlib==3.7.0
numpy==1.23.5
scipy==1.10.0
torch==2.0.0
torch_geometric==2.3.1
```
## Installation

Run the requirments file and the following command:
``` commandline 
pip install -r requirements.txt 
```

For Linux/Google Colab 
- Without GPU:
``` commandline
 pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html 
```
- With GPU:
``` commandline
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```


For Windows
- Without GPU:
``` commandline
 pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html 
```
- With GPU:
``` commandline
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```