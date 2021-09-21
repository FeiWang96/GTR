import dgl
import numpy as np
import networkx as nx
import fasttext
import unicodedata
import re

SPACE_NORMALIZER = re.compile(r"\s+")


class TabularGraph:
    def __init__(self, w2v_path=None, merge_same_cells=False):
        self.w2v = fasttext.load_model(w2v_path)
        self.merge_same_cells = merge_same_cells

    def construct_graph(self, table):
        table_data = table['table_array']
        n = len(table_data)
        m = len(table_data[0]) if n > 0 else 0

        node_features = self._node_embs(table_data, n, m)
        if self.merge_same_cells:
            graph = self._build_complex_graph(table_data, n, m)
        else:
            graph = self._build_graph(n, m)

        graph = dgl.from_networkx(graph)

        # graph = dgl.add_self_loop(graph)

        if graph.num_nodes() != node_features.shape[0]:
            print(f"error: graph.num_nodes() != node_features.shape[0], "
                  f"{graph.num_nodes()} != {node_features.shape[0]}")

        return graph, node_features, table_data

    def _node_embs(self, tarr, n, m):
        tarr = [[self._normalize_text(c) for c in row] for row in tarr]

        features_textual = []
        for i, row in enumerate(tarr):
            for j, c in enumerate(row):
                features_textual.append(self._fasttext_sentence_emb(c))

        for i in range(n):
            row_feat = self._fasttext_sentence_emb(' ')
            for j in range(m):
                row_feat += features_textual[i * m + j]
            features_textual.append(row_feat / m)

        for j in range(m):
            col_feat = self._fasttext_sentence_emb(' ')
            for i in range(n):
                col_feat += features_textual[i * m + j]
            features_textual.append(col_feat / n)

        features = np.array(features_textual)
        return features

    def _fasttext_sentence_emb(self, text):
        if isinstance(text, list):
            text = " ".join(text)
        res = self.w2v[text]
        return res

    @staticmethod
    def _normalize_text(x):
        x = unicodedata.normalize('NFD', x).encode('ascii', 'ignore').decode("utf-8").strip()
        x = x.replace("'", " ")
        x = x.replace('"', " ")
        x = x.replace('|', " ")
        x = x.replace('#', " ")
        x = x.replace('/', " ")
        x = x.replace('\\', " ")
        x = x.replace('(', " ").replace(')', " ")
        x = x.replace('[', " ").replace(']', " ")
        x = SPACE_NORMALIZER.sub(" ", x)
        return x

    def _build_graph(self, n, m):
        graph = nx.DiGraph()
        graph.add_nodes_from(np.arange(n * m + n + m))
        edges = set()

        # edges between cell nodes
        for i in range(n * m):
            for j in self._get_neighbors(i, n, m):
                edges.add((i, j))
                edges.add((j, i))
            # add self loop
            edges.add((i, i))

        # edges from cell nodes to row nodes
        for row_id in range(n):
            row_node = n * m + row_id
            for j in range(m):
                edges.add((row_id * m + j, row_node))

        # edges from cell nodes to col nodes
        for col_id in range(m):
            col_node = n * m + n + col_id
            for i in range(n):
                edges.add((i * m + col_id, col_node))

        graph.add_edges_from(edges)
        return graph

    def _build_complex_graph(self, tarr, n, m):
        graph = nx.DiGraph()
        graph.add_nodes_from(np.arange(n * m + n + m))
        edges = set()

        # merge cells
        cell_node_id = [[i * m + j for j in range(m)] for i in range(n)]
        for j in range(m):
            last_text = self._normalize_text(tarr[0][j])
            for i in range(1, n):
                curr_text = self._normalize_text(tarr[i][j])
                if last_text == curr_text:
                    cell_node_id[i][j] = cell_node_id[i-1][j]
                last_text = curr_text
        cell_node_id = [idx for row in cell_node_id for idx in row]

        # edges between cell nodes
        for i in range(n * m):
            for j in self._get_neighbors(i, n, m):
                idxi = cell_node_id[i]
                idxj = cell_node_id[j]
                edges.add((idxi, idxj))
                edges.add((idxj, idxi))
            # add self loop
            edges.add((i, i))

        # edges from cell nodes to row nodes
        for row_id in range(n):
            row_node = n * m + row_id
            for j in range(m):
                cell_node = cell_node_id[row_id * m + j]
                edges.add((cell_node, row_node))

        # edges from cell nodes to col nodes
        for col_id in range(m):
            col_node = n * m + n + col_id
            for i in range(n):
                cell_node = cell_node_id[i * m + col_id]
                edges.add((cell_node, col_node))

        graph.add_edges_from(edges)
        return graph

    @staticmethod
    def _get_neighbors(ind, n, m):
        i = ind // m
        j = ind % m
        res = []

        # right neighbor
        if j < m - 1:
            res.append(ind + 1)

        # bottom neighbor
        if i < n - 1:
            res.append(ind + m)

        return res
