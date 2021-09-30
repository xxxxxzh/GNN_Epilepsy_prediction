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
python main.py --model_name GCN --num_layer 4 --seq_len 500 --hidden_dim 64 --learning_rate 1e-3 --weight_decay 1.5e-3 --num_epoch 100
```

- **--model_name：**模型名称（目前只有GCN）
- **--num_layer**: 隐藏层数量
- **--seq_len**: 序列长度（单条分类取1）
- **--hidden_dim**: 隐藏层特征维数
- **--learning_rate**：学习率
- **--weight_decay**：学习率权重衰减
- **--num_epoch**：训练轮次
