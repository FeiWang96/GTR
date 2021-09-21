import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import BertTokenizer, BertModel

from src.table_encoder.gat import GATEncoder


class MatchingModel(nn.Module):
    def __init__(self, bert_dir='bert-base-uncased', do_lower_case=True, bert_size=768, gnn_output_size=300):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(bert_dir, do_lower_case=do_lower_case)
        self.bert = BertModel.from_pretrained(bert_dir)

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

        self.regression = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(bert_size + 1200, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
        )

    def forward(self, table, query, dgl_g, t_feat, q_feat):
        """table retrieval"""
        bert_rep = self.text_matching(table, query)

        gnn_rep = self.text_table_matching(dgl_g, t_feat, q_feat)

        rep = torch.cat((bert_rep, gnn_rep), -1)

        score = self.regression(rep)

        return score

    def text_table_matching(self, dgl_graph, table_embs, query_emb):
        """text-table matching module"""
        creps = self.gnn(dgl_graph, table_embs)

        tmapping = self.project_table(creps)
        qmapping = query_emb.repeat(creps.shape[0], 1)

        hidden = torch.cat((tmapping, qmapping, tmapping - qmapping, tmapping * qmapping), 1)

        hidden = self.dim_reduction(hidden)

        hidden = torch.max(hidden, 0)[0]

        return hidden

    def text_matching(self, table, query):
        """text-text matching module"""

        tokens = ["[CLS]"]
        tokens += self.tokenizer.tokenize(" ".join(query))[:64]
        tokens += ["[SEP]"]

        token_types = [0 for _ in range(len(tokens))]

        tokens += self.tokenizer.tokenize(table["caption"])[:20]
        tokens += ["[SEP]"]

        if 'subcaption' in table:
            tokens += self.tokenizer.tokenize(table["subcaption"])[:20]
            # tokens += ["[SEP]"]

        if 'pgTitle' in table:
            tokens += self.tokenizer.tokenize(table["pgTitle"])[:10]
            # tokens += ["[SEP]"]

        if 'secondTitle' in table:
            tokens += self.tokenizer.tokenize(table["secondTitle"])[:10]
            # tokens += ["[SEP]"]

        token_types += [1 for _ in range(len(tokens) - len(token_types))]

        # truncate and pad
        tokens = tokens[:128]
        token_types = token_types[:128]

        assert len(tokens) == len(token_types)

        token_indices = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_tensor = torch.tensor([token_indices]).to("cuda")
        token_type_tensor = torch.tensor([token_types]).to("cuda")

        outputs = self.bert(tokens_tensor, token_type_ids=token_type_tensor)

        return outputs[1][0]  # pooled output of the [CLS] token
