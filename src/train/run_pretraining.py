import random
import datetime
import torch
from tqdm import tqdm

from src.utils.load_data import load_tables
from src.utils.reproducibility import set_random_seed
from src.model.pretrain import PretrainMatching
from src.graph_construction.tabular_graph import TabularGraph

tables = None


def pretrain(config):
    set_random_seed()

    global tables
    tables = load_tables(config["data_dir"])

    model = PretrainMatching(gnn_output_size=config["gnn_size"]).to('cuda')

    print(config)
    print(model)

    constructor = TabularGraph(config["use_fasttext"], config["fasttext"])

    loss_func = torch.nn.MSELoss()

    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': config["lr"]}])

    train_dataset = list(tables.keys())

    labels = torch.FloatTensor([1, 0, 0, 1]).to("cuda")

    for e in range(config["epoch"]):
        random.shuffle(train_dataset)

        model.train()

        eloss = 0
        batch_loss = 0

        pbar = tqdm(range(len(train_dataset) - 1), desc=f'Epoch {e+1}/{config["epoch"]}', unit='it')

        for i in pbar:
            tid_a = train_dataset[i]
            tid_b = train_dataset[i + 1]
            table_a = tables[tid_a]
            table_b = tables[tid_b]

            grahp_a, feature_a, table_data_a = constructor.construct_graph(table_a)
            grahp_b, feature_b, table_data_b = constructor.construct_graph(table_b)

            grahp_a = grahp_a.to("cuda")
            grahp_b = grahp_b.to("cuda")

            feature_a = torch.FloatTensor(feature_a).to("cuda")
            feature_b = torch.FloatTensor(feature_b).to("cuda")

            caption_a = constructor.w2v[table_a["caption"]]
            caption_b = constructor.w2v[table_b["caption"]]
            pgtitle_a = constructor.w2v[table_a["pgTitle"]]
            pgtitle_b = constructor.w2v[table_b["pgTitle"]]
            sectitle_a = constructor.w2v[table_a["secondTitle"]]
            sectitle_b = constructor.w2v[table_b["secondTitle"]]

            caption_a = torch.FloatTensor(caption_a).to("cuda")
            caption_b = torch.FloatTensor(caption_b).to("cuda")
            pgtitle_a = torch.FloatTensor(pgtitle_a).to("cuda")
            pgtitle_b = torch.FloatTensor(pgtitle_b).to("cuda")
            sectitle_a = torch.FloatTensor(sectitle_a).to("cuda")
            sectitle_b = torch.FloatTensor(sectitle_b).to("cuda")

            preds = model(grahp_a, grahp_b, feature_a, feature_b,
                          caption_a, caption_b, pgtitle_a, pgtitle_b, sectitle_a, sectitle_b)

            loss = loss_func(preds.reshape(-1, 1), labels.reshape(-1, 1))

            batch_loss += loss

            if i % config["batch_size"] == 0 or i + 1 >= len(train_dataset):
                batch_loss /= config["batch_size"]

                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                pbar.set_postfix(batch_loss=batch_loss.item())

                optimizer.zero_grad()
                batch_loss = 0

            eloss += loss.item()

        del pbar

        print(datetime.datetime.now(), e, eloss / len(train_dataset), flush=True)

        torch.save(model.state_dict(), "pretrained_models/" + config["save_prefix"] + "_" + str(e) + ".bin")
