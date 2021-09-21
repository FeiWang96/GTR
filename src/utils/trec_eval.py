import subprocess


def write_trec_result(eval_df, rank_path='trec_rank.txt', qrel_path='trec_qrel.txt'):
    with open(rank_path, 'w') as f_rank, open(qrel_path, 'w') as f_rel:
        qids = [each for each in sorted(list(set([each for each in eval_df['id_left'].unique()])))]
        for qid in qids:
            qid_docs = eval_df[eval_df['id_left'] == qid]
            qid_docs = qid_docs.sort_values(by=['pred'], ascending=False)
            for r, value in enumerate(qid_docs.values):
                f_rank.write(f"{value[0]}\tQ0\t{value[1]}\t{r+1}\t{value[3]}\t0\n")
                f_rel.write(f"{value[0]}\t0\t{value[1]}\t{value[2]}\n")


def get_metrics(metric='ndcg_cut', rank_path='trec_rank.txt', qrel_path='trec_qrel.txt'):
    if metric == 'ndcg_cut':
        metrics = ['ndcg_cut_5', 'ndcg_cut_10', 'ndcg_cut_15', 'ndcg_cut_20']
    elif metric == 'map':
        metrics = ['map']
    else:
        raise ValueError(f"Invalid metric {metric}.")

    results = subprocess.run(['./trec_eval/trec_eval', '-c', '-m', metric, '-q', qrel_path, rank_path],
                             stdout=subprocess.PIPE).stdout.decode('utf-8')

    ndcg_scores = dict()
    for line in results.strip().split("\n"):
        seps = line.split('\t')
        metric_name = seps[0].strip()
        qid = seps[1].strip()
        if metric_name not in metrics or qid != 'all':
            continue
        ndcg_scores[seps[0].strip()] = float(seps[2])
    return ndcg_scores


