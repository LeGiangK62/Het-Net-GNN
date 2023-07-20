# Het-Net-GNN

Repository Implementing **Heterogeneous Network Graph Neural Network** (HetNetGNN) on Wireless Cell-Free Network.

## Installation

Run the requirments file and the following command:
``` 
pip install -r requirements.txt 
```

For Linux/Google Colab 
- Without GPU:
```
 pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html 
```
- With GPU:
```
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```


For Windows
- Without GPU:
```
 pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html 
```
- With GPU:
```
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```