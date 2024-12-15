import torch
import torch.nn as nn
import torch.optim as optim


# embedding只更新嵌入的行的权重，其他的不更新

class SimpleBinaryClassifier(nn.Module):
    def __init__(self):
        super(SimpleBinaryClassifier, self).__init__()

        self.embedding = nn.Embedding(10, 3)
        nn.init.uniform_(self.embedding.weight)

        # 全连接层：输入 9 维，输出 1 维 (二分类)
        self.fc = nn.Linear(9, 1)

        # Sigmoid 激活函数，用于输出概率
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x).flatten()
        x = self.fc(x)  # 将输入传入全连接层
        x = self.sigmoid(x)  # 输出通过 Sigmoid 函数得到概率
        return x

# 模拟一个输入，假设我们有一个batch，里面包含类别索引
input_indices = torch.tensor([2, 7, 5])  # 每个值代表类别索引

# 目标标签
target = torch.tensor([1])  # 假设这是二分类的目标标签
model = SimpleBinaryClassifier()
optimizer = optim.SGD(model.parameters(), lr=0.1)

print("Before embedding weights:\n", model.embedding.weight)

output = model(input_indices)
print(output)

# 假设我们使用一个简单的损失函数（这里为了简化，用平方误差）
loss_fn = nn.MSELoss()
loss = loss_fn(output, target.float())  # 简单的损失计算

# 反向传播
optimizer.zero_grad()  # 清除之前的梯度
loss.backward()  # 反向传播计算梯度

# 更新嵌入层的权重
optimizer.step()

# 查看嵌入矩阵的更新
print("Updated embedding weights:\n", model.embedding.weight)


if __name__ == '__main__':
    w = 3
    h = 2
    # 表示列位置
    i = torch.arange(w)
    # 表示行位置
    j = torch.arange(h)
    print(i, j)
    embedding = nn.Embedding(10, 3)
    x_emb = embedding(i) # (3, 3)
    # 将行位置 j 输入到 row_embed 嵌入层，得到行方向的位置嵌入（y_emb）。
    y_emb = embedding(j)
    print(x_emb, y_emb)  # (2, 3)

    # concat (2, 3, 6) -> permute(6, 2, 3) -> unsqueeze(0)  (1, 6, 2, 3)
    pos = torch.cat([
        # unsqueeze(0).repeat(h, 1, 1)：对列嵌入进行扩展，以便将列嵌入沿着行维度复制 h 次，从而使每一行的所有列共享相同的列位置嵌入。
        # (2, 3, 3)
        x_emb.unsqueeze(0).repeat(h, 1, 1),
        # unsqueeze(1).repeat(1, w, 1)：对行嵌入进行扩展，以便将行嵌入沿着列维度复制 w 次，从而使每一列的所有行共享相同的行位置嵌入
        # (2, 3, 3)
        y_emb.unsqueeze(1).repeat(1, w, 1),
        # pos = ... .permute(2, 0, 1)：调整位置嵌入的维度顺序，将位置嵌入转换为 (2 * num_pos_feats, h, w) 的形状。
        # unsqueeze(0).repeat(x.shape[0], 1, 1, 1)：将位置嵌入扩展到整个批次（batch size）。最终的输出 pos 形状为 (batch_size, 2 * num_pos_feats, h, w)，它包含了整个批次中每个像素的位置嵌入。
    ], dim=-1).permute(2, 0, 1).unsqueeze(0)
    print(pos.shape)  # (2, 3, 6) -> ()