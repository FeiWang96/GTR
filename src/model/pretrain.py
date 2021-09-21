import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import BertTokenizer, BertModel

from src.table_encoder.gat import GATEncoder


class PretrainMatching(nn.Module):
    def __init__(self, gnn_output_size=300):
        super().__init__()

        self.gnn = GATEncoder(input_dim=300, output_dim=gnn_output_size, hidden_dim=300, layer_num=4,
                              activation=nn.LeakyReLU(0.2))

        self.project_table = nn.Sequential(
            nn.Linear(gnn_output_size, 300),
            nn.LayerNorm(300)
        )

        self.dim_reduction = nn.Sequential(
            nn.Linear(1200, 1200),
            nn.Tanh(),
        )

        self.predictor = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(1200, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
        )

    def forward(self, grahp_a, grahp_b, feature_a, feature_b, caption_a, caption_b, pgtitle_a, pgtitle_b, sectitle_a,
                sectitle_b):
        """pretraining task: caption prediction"""
        taqa = self.text_table_matching(grahp_a, feature_a, caption_a, pgtitle_a, sectitle_a)
        taqa = self.predictor(taqa)
        taqb = self.text_table_matching(grahp_a, feature_a, caption_b, pgtitle_b, sectitle_b)
        taqb = self.predictor(taqb)
        tbqa = self.text_table_matching(grahp_b, feature_b, caption_a, pgtitle_a, sectitle_a)
        tbqa = self.predictor(tbqa)
        tbqb = self.text_table_matching(grahp_b, feature_b, caption_b, pgtitle_b, sectitle_b)
        tbqb = self.predictor(tbqb)

        logits = torch.cat((taqa, taqb, tbqa, tbqb))

        return logits

    def text_table_matching(self, dgl_graph, table_embs, caption_emb, pgtitle_emb, sectitle_emb):
        """text-table matching module"""
        creps = self.gnn(dgl_graph, table_embs)

        tmapping = self.project_table(creps)

        qmapping = torch.cat((caption_emb, pgtitle_emb, sectitle_emb)).view(3, -1)

        qmapping = torch.mean(qmapping, 0)

        qmapping = qmapping.repeat(tmapping.shape[0], 1)

        hidden = torch.cat((tmapping, qmapping, tmapping - qmapping, tmapping * qmapping), 1)

        hidden = self.dim_reduction(hidden)

        hidden = torch.max(hidden, 0)[0]

        return hidden
