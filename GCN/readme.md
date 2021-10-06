### Requirements

---

- numpy
- matplotlib
- pandas
- torch
- pytorch-lightning>=1.3.0
- torchmetrics>=0.3.0
- python-dotenv
- torch-scatter
- torch-sparse
- torch-geometric

---

### Train Model

```
python main.py --model_name GCN --log_dir runs/GCN --num_layer 2 --seq_len 100 --hidden_dim 64 --learning_rate 1e-3 --weight_decay 1.5e-3 --num_epoch 100
python main.py --model_name TGCN --log_dir runs/TGCN --num_layer 2 --pooling_first True --seq_len 100 --hidden_dim 64 --learning_rate 1e-3 --weight_decay 1.5e-3 --num_epoch 100```

- **--model_name**：模型名称（目前只有GCN）
- **--log_dir**：Tensorboard监控训练的目录
- **--num_layer**: 隐藏层数量
- **--pooling_first**：GCN后先池化（默认先池化（适用于图分类），选择 False 会最后池化）  
- **--seq_len**: 序列长度（单条分类取1）
- **--hidden_dim**: 隐藏层特征维数
- **--learning_rate**：学习率
- **--weight_decay**：学习率权重衰减
- **--num_epoch**：训练轮次

