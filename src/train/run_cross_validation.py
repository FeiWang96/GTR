import random
import datetime
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from pytorch_transformers import WarmupLinearSchedule

from src.utils.trec_eval import write_trec_result, get_metrics
from src.utils.load_data import load_queries, load_tables, load_qt_relations
from src.utils.reproducibility import set_random_seed
from src.model.retrieval import MatchingModel
from src.graph_construction.tabular_graph import TabularGraph
from src.utils.process_table_and_query import *

queries = None
tables = None
qtrels = None


def evaluate(config, model, query_id_list):
    qids = []
    docids = []
    gold_rel = []
    pred_rel = []

    model.eval()
    with torch.no_grad():
        for qid in query_id_list:
            query = queries["sentence"][qid]
            query_feature = queries["feature"][qid].to("cuda")

            for (tid, rel) in qtrels[qid].items():
                table = tables[tid]
                dgl_graph = tables[tid]["dgl_graph"].to("cuda")
                node_features = tables[tid]["node_features"].to("cuda")

                score = model(table, query, dgl_graph, node_features, query_feature).item()

                qids.append(qid)
                docids.append(tid)
                gold_rel.append(rel)
                pred_rel.append(score * config["relevance_score_scale"])

    eval_df = pd.DataFrame(data={
        'id_left': qids,
        'id_right': docids,
        'true': gold_rel,
        'pred': pred_rel
    })

    write_trec_result(eval_df)
    metrics = get_metrics('ndcg_cut')
    metrics.update(get_metrics('map'))
    return metrics


def train(config, model, train_pairs, optimizer, scheduler, loss_func):
    random.shuffle(train_pairs)

    model.train()

    eloss = 0
    batch_loss = 0
    n_iter = 0
    for (qid, tid, rel) in train_pairs:
        n_iter += 1

        label = rel * 1.0 / config["relevance_score_scale"]

        query = queries["sentence"][qid]
        query_feature = queries["feature"][qid].to("cuda")

        table = tables[tid]
        dgl_graph = tables[tid]["dgl_graph"].to("cuda")
        node_features = tables[tid]["node_features"].to("cuda")

        prob = model(table, query, dgl_graph, node_features, query_feature)

        loss = loss_func(prob.reshape(-1), torch.FloatTensor([label]).to("cuda"))

        batch_loss += loss

        if n_iter % config["batch_size"] == 0 or n_iter == len(train_pairs):
            batch_loss /= config["batch_size"]
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()
            batch_loss = 0

        eloss += loss.item()

    return eloss / len(train_pairs)


def cross_validation(config):
    set_random_seed()

    # The Answer to the Ultimate Question of Life, the Universe, and Everything is 42
    # -- The Hitchhiker's Guide to the Galaxy
    ss = ShuffleSplit(n_splits=5, train_size=0.8, random_state=42)

    global queries, tables, qtrels
    queries = {}
    queries["sentence"] = load_queries(config["data_dir"])
    tables = load_tables(config["data_dir"])
    qtrels = load_qt_relations(config["data_dir"])

    model = MatchingModel(bert_dir=config["bert_dir"], do_lower_case=config["do_lower_case"],
                          bert_size=config["bert_size"], gnn_output_size=config["gnn_size"])
    print(config)
    print(model, flush=True)

    constructor = TabularGraph(config["fasttext"], config["merge_same_cells"])

    queries["feature"] = process_queries(queries["sentence"], constructor)
    process_tables(tables, constructor)

    loss_func = torch.nn.MSELoss()

    qindex = list(queries["sentence"].keys())
    sample_index = np.array(range(len(qindex))).reshape((-1, 1))

    best_cv_metrics = [None for _ in range(5)]
    for n_fold, (train_data, validation_data) in enumerate(ss.split(sample_index)):

        train_query_ids = [qindex[idx] for idx in train_data]
        validation_query_ids = [qindex[idx] for idx in validation_data]

        del model

        model = MatchingModel(bert_dir=config["bert_dir"], do_lower_case=config["do_lower_case"],
                              bert_size=config["bert_size"], gnn_output_size=config["gnn_size"])

        if config["use_pretrained_model"]:
            pretrain_model = torch.load(config["pretrained_model_path"])
            model.load_state_dict(pretrain_model, strict=False)

        model = model.to("cuda")

        optimizer = torch.optim.Adam([
            {'params': model.bert.parameters(), 'lr': config["bert_lr"]},
            {'params': (x for x in model.parameters() if x not in set(model.bert.parameters())), 'lr': config["gnn_lr"]}
        ])
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=config["warmup_steps"], t_total=config["total_steps"])

        train_pairs = [(qid, tid, rel) for qid in train_query_ids for tid, rel in qtrels[qid].items()]

        best_metrics = None
        for epoch in range(config['epoch']):
            train(config, model, train_pairs, optimizer, scheduler, loss_func)

            train_metrics = evaluate(config, model, train_query_ids)
            test_metrics = evaluate(config, model, validation_query_ids)

            if best_metrics is None or test_metrics[config['key_metric']] > best_metrics[config['key_metric']]:
                best_metrics = test_metrics
                best_cv_metrics[n_fold] = best_metrics
                print(datetime.datetime.now(), 'epoch', epoch, 'train', train_metrics, 'test', test_metrics, "*",
                      flush=True)
            else:
                print(datetime.datetime.now(), 'epoch', epoch, 'train', train_metrics, 'test', test_metrics, flush=True)

    avg_metrics = best_cv_metrics[0]
    for key in avg_metrics.keys():
        for metrics in best_cv_metrics[1:]:
            avg_metrics[key] += metrics[key]
        avg_metrics[key] /= 5
    print("5-fold cv scores", avg_metrics)
