import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'docs/tables'
FIGURES = ROOT / 'docs/figures'
NAMES = {'bbc': 'BBC', '20ng_4cats': '20NG (4 кат.)', 'agnews': 'AG News', 'techcrunch': 'TechCrunch', 'pubmed': 'PubMed'}
CORPORA = list(NAMES)


def read(name):
    with (ROOT / 'data/results' / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (TABLES / name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(name, caption, label, columns, rows, alignment=None):
    content = '\\begin{table}[H]\n\\centering\n\\small\n'
    content += '\\caption{' + caption + '}\n\\label{' + label + '}\n'
    content += '\\begin{tabular}{' + (alignment or 'l' + 'r' * (len(columns) - 1)) + '}\n\\toprule\n'
    content += ' & '.join(columns) + ' \\\\\n\\midrule\n'
    content += ''.join(' & '.join(str(v) for v in row) + ' \\\\\n' for row in rows)
    content += '\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    (TABLES / name).write_text(content, encoding='utf-8')


def gain(baseline, method):
    return 100 * (float(baseline) - float(method)) / float(baseline)


def ms(values, digits=3):
    return f'${mean(values):.{digits}f}\\pm{stdev(values):.{digits}f}$'


def main():
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    small = read('small_runs.csv')
    keys = [(r['corpus'], r['size'], r['seed'], r['name']) for r in small]
    assert len(keys) == len(set(keys)) == 171
    base = {(r['corpus'], r['size'], r['seed']): r for r in small if r['name'] == 'aartm'}
    pairs = []
    for r in small:
        if r['name'] == 'aartm':
            continue
        b = base[r['corpus'], r['size'], r['seed']]
        assert all(r[k] == b[k] for k in ['vocab_size', 'n_tokens', 'n_topics', 'coverage'])
        pairs.append(dict(corpus=r['corpus'], size=int(r['size']), seed=int(r['seed']), variant=r['name'],
                          completion_gain=gain(b['completion_perplexity'], r['completion_perplexity']),
                          train_gain=gain(b['perplexity'], r['perplexity']),
                          pmi_delta=float(r['coherence'])-float(b['coherence']),
                          npmi_delta=float(r['reference_npmi'])-float(b['reference_npmi'])))
    write_csv('paired_effects.csv', pairs)
    full = [p for p in pairs if p['variant'] == 'contextual_kl']
    assert len(full) == 45
    assert abs(mean(p['completion_gain'] for p in full) - 7.4047) < 0.0001
    assert sum(p['completion_gain'] > 0 for p in full) == 42
    assert sum(p['npmi_delta'] > 0 for p in full) == 17
    stats = []
    for corpus in CORPORA:
        subset = [r for r in small if r['corpus'] == corpus and r['name'] == 'aartm']
        rr = [p for p in full if p['corpus'] == corpus]
        stats.append(dict(corpus=corpus, n_topics=int(subset[0]['n_topics']),
                          vocab_min=min(int(r['vocab_size']) for r in subset), vocab_max=max(int(r['vocab_size']) for r in subset),
                          tokens_min=min(int(r['n_tokens']) for r in subset), tokens_max=max(int(r['n_tokens']) for r in subset),
                          coverage_mean=mean(float(r['coverage']) for r in subset),
                          completion_gain_mean=mean(p['completion_gain'] for p in rr), completion_gain_sd=stdev(p['completion_gain'] for p in rr),
                          completion_wins=sum(p['completion_gain'] > 0 for p in rr), npmi_delta_mean=mean(p['npmi_delta'] for p in rr)))
    write_csv('small_summary.csv', stats)
    table('data_statistics.tex', 'Число тем, диапазоны размера словаря и числа обучающих токенов на малых коллекциях. Покрытие тестовых токенов усреднено по девяти выборкам; его различия следует учитывать при сравнении PPL.', 'tab:data',
          ['Корпус', '$T$', '$|V|$', 'Токены', 'Покрытие'],
          [[NAMES[r['corpus']], r['n_topics'], f"{r['vocab_min']}--{r['vocab_max']}", f"{r['tokens_min']}--{r['tokens_max']}", f"{r['coverage_mean']:.3f}"] for r in stats])
    table('small_results.tex', 'Среднее снижение completion PPL и изменение внешней NPMI для AARTM+Q относительно AARTM, по девять пар на корпус.', 'tab:small',
          ['Корпус', 'Снижение PPL, \\%', '$\\Delta$NPMI'],
          [[NAMES[r['corpus']], f"{r['completion_gain_mean']:.2f}", f"{r['npmi_delta_mean']:+.4f}"] for r in stats])
    ablations = []
    variants = [('contextual_kl', 'Инициализация + KL'), ('contextual_init_only', 'Только инициализация'), ('contextual_no_init', 'Только KL'), ('contextual_shuffled', 'Переставленный учитель')]
    for variant, title in variants:
        pp = [p for p in pairs if p['variant'] == variant and p['corpus'] in ['bbc', 'techcrunch', 'pubmed']]
        assert len(pp) == 27
        ablations.append(dict(variant=variant, label=title, n_pairs=len(pp), gain_mean=mean(p['completion_gain'] for p in pp), gain_sd=stdev(p['completion_gain'] for p in pp), wins=sum(p['completion_gain'] > 0 for p in pp)))
    write_csv('ablation_summary.csv', ablations)
    table('ablation_results.tex', 'Снижение completion PPL~\\eqref{eq:gain} на BBC, TechCrunch и PubMed, 27 общих пар. Инициализация сохраняет большую часть выигрыша полного метода.', 'tab:ablation',
          ['Вариант', '$\\bar g$, \\%', 'SD, п.п.', 'Победы'], [[r['label'], f"{r['gain_mean']:.2f}", f"{r['gain_sd']:.2f}", f"{r['wins']}/27"] for r in ablations])
    arxiv = read('arxiv_runs.csv')
    arxiv_summary = []
    assert len(arxiv) == 40
    for topics in [50, 100, 200, 300, 500]:
        aa = sorted([r for r in arxiv if int(r['n_topics']) == topics and r['model'] == 'aartm'], key=lambda r: r['seed'])
        qq = sorted([r for r in arxiv if int(r['n_topics']) == topics and r['model'] == 'aartm_q'], key=lambda r: r['seed'])
        assert [r['seed'] for r in aa] == [r['seed'] for r in qq] == ['42', '43', '44', '45']
        pmi_a = [float(r['coherence']) for r in aa]
        pmi_q = [float(r['coherence']) for r in qq]
        ppl_a = [float(r['perplexity']) for r in aa]
        ppl_q = [float(r['perplexity']) for r in qq]
        arxiv_summary.append(dict(topics=topics, pmi_aartm_mean=mean(pmi_a), pmi_aartm_sd=stdev(pmi_a), pmi_q_mean=mean(pmi_q), pmi_q_sd=stdev(pmi_q),
                                  ppl_aartm_mean=mean(ppl_a), ppl_aartm_sd=stdev(ppl_a), ppl_q_mean=mean(ppl_q), ppl_q_sd=stdev(ppl_q),
                                  ppl_gain_mean=mean(gain(a,b) for a,b in zip(ppl_a,ppl_q)), pmi_ratio=mean(pmi_q)/mean(pmi_a), pmi_wins=sum(q > a for a,q in zip(pmi_a,pmi_q))))
    assert sum(r['pmi_wins'] for r in arxiv_summary) == 20
    write_csv('arxiv_summary.csv', arxiv_summary)
    baselines = read('arxiv_baseline_runs.csv')
    assert len(baselines) == 80
    models = ['AARTM+Q', 'FASTopic', 'CombinedTM+MiniLM']
    bs = []
    pmi_rows, time_rows = [], []
    for topics in [50,100,200,300,500]:
        pmis, times = [], []
        for model in models:
            rr = [r for r in baselines if r['model'] == model and int(r['topics']) == topics]
            assert len(rr) == 4 and len({r['seed'] for r in rr}) == 4
            pmi = [float(r['train_pmi_top10']) for r in rr]
            runtime = [float(r['reported_runtime_seconds']) for r in rr]
            pmis.append(ms(pmi))
            times.append(ms(runtime, 0))
            bs.append(dict(topics=topics, model=model, n_seeds=4, pmi_mean=mean(pmi), pmi_sd=stdev(pmi), runtime_mean=mean(runtime), runtime_sd=stdev(runtime)))
        pmi_rows.append([topics]+pmis)
        time_rows.append([topics]+times)
    write_csv('baseline_summary.csv', bs)
    comparison_models = ['AARTM', 'AARTM+Q', 'FASTopic', 'CombinedTM+MiniLM']
    comparison = [[topics] + [ms([float(r['train_pmi_top10']) for r in baselines if int(r['topics']) == topics and r['model'] == model]) for model in comparison_models] for topics in [50, 100, 200, 300, 500]]
    table('arxiv_results.tex', 'Обучающая PMI на arXiv: среднее $\\pm$ выборочное стандартное отклонение по четырём seed.', 'tab:arxiv', ['$T$'] + comparison_models, comparison)
    table('baseline_pmi.tex', 'Обучающая PMI~\\eqref{eq:coherence} на arXiv, среднее $\\pm$ выборочное SD по четырём seed. AARTM+Q имеет наибольшую PMI при заданных бюджетах обучения.', 'tab:baselines', ['$T$']+models, pmi_rows)
    table('baseline_runtime.tex', 'Время в секундах, среднее $\\pm$ выборочное SD. Для AARTM+Q учтены построение $Q$ и обучение, для остальных --- обучение; первичное кодирование исключено. Соотношение времён меняется с числом тем.', 'tab:runtime', ['$T$']+models, time_rows)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(11,3.8), layout='constrained')
    ax = axes[0]
    for k, size in enumerate([10,30,50]):
        vals = [p['completion_gain'] for p in full if p['size']==size]
        ax.scatter([k + (i-(len(vals)-1)/2)*.013 for i in range(len(vals))], vals, s=22, color='#9cb9ce', alpha=.8)
        ax.scatter(k, mean(vals), marker='D', s=55, color='#153f5e', zorder=3)
        ax.annotate(f'{mean(vals):.2f}%', (k,mean(vals)), xytext=(6,7), textcoords='offset points', color='#153f5e')
    ax.axhline(0,color='#888888',linewidth=.8)
    ax.set(xticks=[0,1,2],xticklabels=['10','30','50'],xlabel='Обучающих документов',ylabel='Снижение completion PPL, %')
    ax=axes[1]
    ax.barh([NAMES[r['corpus']] for r in stats],[r['npmi_delta_mean'] for r in stats],color=['#218381' if r['npmi_delta_mean']>0 else '#c77845' for r in stats])
    ax.axvline(0,color='#666666',linewidth=.8)
    ax.set(xlabel='Изменение внешней NPMI')
    fig.savefig(FIGURES/'small_effects.pdf')
    fig.savefig(FIGURES/'small_effects.png',dpi=180)
    plt.close(fig)
    fig, ax=plt.subplots(figsize=(7,3.8),layout='constrained')
    for name,prefix,color in [('AARTM','aartm','#6c7b86'),('AARTM+Q','q','#176e83')]:
        ax.errorbar([r['topics'] for r in arxiv_summary],[r[f'pmi_{prefix}_mean'] for r in arxiv_summary],yerr=[r[f'pmi_{prefix}_sd'] for r in arxiv_summary],marker='o',capsize=4,label=name,color=color)
    ax.set(xlabel='Число тем T',ylabel='Training PMI top-10',xticks=[50,100,200,300,500])
    ax.legend(frameon=False)
    fig.savefig(FIGURES/'arxiv_pmi.pdf')
    fig.savefig(FIGURES/'arxiv_pmi.png',dpi=180)
    plt.close(fig)
    for path in sorted((ROOT/'data/manifests').glob('*.json')):
        manifest=json.loads(path.read_text(encoding='utf-8'))
        corpus, size, seed = path.stem.removesuffix('_manifest').rsplit('_', 2)
        assert len(manifest['train']) == int(size)
        assert len(manifest['vocabulary']) == int(base[corpus, size, seed]['vocab_size'])
        sets=[set(manifest[k]) for k in ['train','reference','evaluation']]
        assert all(not sets[i].intersection(sets[j]) for i in range(3) for j in range(i))
        assert len(manifest['evaluation'])==40 and len(manifest['reference'])==200
    for artifact in json.loads((ROOT/'data/provenance.json').read_text(encoding='utf-8'))['files']:
        assert hashlib.sha256((ROOT/artifact['published']).read_bytes()).hexdigest()==artifact['sha256']
    metrics=dict(n_pairs=45, completion_gain=mean(p['completion_gain'] for p in full), completion_wins=42,
                 train_gain=mean(p['train_gain'] for p in full), pmi_delta=mean(p['pmi_delta'] for p in full),
                 npmi_delta=mean(p['npmi_delta'] for p in full), npmi_wins=17,
                 n_manifests=len(list((ROOT/'data/manifests').glob('*.json'))))
    (TABLES/'verification.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(metrics,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
