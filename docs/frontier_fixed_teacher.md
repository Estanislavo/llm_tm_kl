# Источники и границы научной проверки

Дата: 23 октября 2026. Область — подготовка четырёх учебных разделов о фиксированном контекстном учителе AARTM+Q по существующим экспериментам. Это целевая проверка ближайших работ и ссылок, а не систематический поиск всех методов на дату публикации. Актуальный SOTA не устанавливался. Все ссылки ниже открыты при подготовке материалов; для литературы сверены название, авторы, статус и год по первичным страницам. У Hoyle, CombinedTM и FASTopic дополнительно открыты полные статьи для проверки описания метода и протокола.

## Ближайшие работы

| Источник | Проверенный вклад и отношение к нашей работе |
|---|---|
| [Blei, Ng, Jordan. JMLR 3, 2003](https://jmlr.org/papers/v3/blei03a.html) | Latent Dirichlet Allocation; темы как распределения слов и документы как смеси тем. Фундаментальный источник, не экспериментальный baseline этой серии |
| [Vorontsov, Potapenko. Machine Learning 101, 2015](https://link.springer.com/article/10.1007/s10994-014-5476-6) | Additive Regularization of Topic Models; регуляризация критерия. Не приписываем этой статье локальное внимание AARTM |
| [Dieng, Ruiz, Blei. TACL 8, 2020](https://aclanthology.org/2020.tacl-1.29/) | Topic Modeling in Embedding Spaces; введение семантики через эмбеддинги слов и тем |
| [Hoyle, Goel, Resnik. EMNLP 2020](https://aclanthology.org/2020.emnlp-main.137/) | Improving Neural Topic Models using Knowledge Distillation; ближайший по идее учителя предшественник. BERT-based autoencoder поставляет распределения слов документа. У нас Q задаёт темы слов; нельзя заявлять новизну teacher guidance вообще |
| [Bianchi, Terragni, Hovy. ACL-IJCNLP 2021](https://aclanthology.org/2021.acl-short.96/) | Pre-training is a Hot Topic; контекстные представления документов в тематической модели. Наш CombinedTM+MiniLM с общим словарём не воспроизводит авторскую конфигурацию |
| [Wu, Nguyen, Zhang, Wang, Luu. NeurIPS 37, 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/998f8d8ff2697da2eae5f87143668754-Abstract-Conference.html) | FASTopic; фиксированные эмбеддинги документов, обучаемые слова и темы, optimal transport. Наши 30 эпох не устанавливают его лучшую сходимость |
| [Fang, He, Procter. LREC-COLING 2024](https://aclanthology.org/2024.lrec-main.382/) | CWTM; контекстные представления слов для neural topic modeling. Использован как источник обзора, сопоставимых локальных запусков нет |
| [Angelov, Inkpen. Findings EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.790/) | Contextual-Top2Vec; contextual token embeddings, темы и spans. В тексте только обзор, ограниченная историческая серия не включена в основные таблицы |
| [Reimers, Gurevych. EMNLP-IJCNLP 2019](https://aclanthology.org/D19-1410/) | Sentence-BERT; семантические представления, сравниваемые по cosine similarity |
| [Sentence Transformers: MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Реальный используемый checkpoint, 384 измерения и ограничение 256 word pieces; вектор целого документа не означает, что кодировщик увидел весь текст |

Сравнимые численные результаты других статей сюда не переносятся: корпус, словарь, разбиение и оценщик отличаются. Простые внутренние baseline — AARTM, только инициализация, только KL, перестановка учителя — дают наиболее прямую проверку нашей гипотезы.

## Метрики и данные

- [Wallach, Murray, Salakhutdinov, Mimno. Evaluation Methods for Topic Models, ICML 2009](https://homepages.inf.ed.ac.uk/imurray2/pub/09etm/) — мотивация held-out evaluation. Наши конкретные 40 обновлений theta и OOV-правило заданы кодом проекта, а не приписываются статье.
- [Newman, Lau, Grieser, Baldwin. NAACL-HLT 2010](https://aclanthology.org/N10-1012/) — Automatic Evaluation of Topic Coherence. Конкретный top-10, document-presence и penalty zero-pairs описаны как наша реализация.
- [Bouma. Normalized (Pointwise) Mutual Information in Collocation Extraction, 2009](https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf) — формула NPMI; в `.bib` указан доступный авторский текст без неподтверждённых выходных данных.
- Карточки [BBC](https://huggingface.co/datasets/SetFit/bbc-news), [20NG](https://huggingface.co/datasets/SetFit/20_newsgroups), [AG News](https://huggingface.co/datasets/fancyzhx/ag_news), [TechCrunch](https://huggingface.co/datasets/abhilash88/techcrunch-articles), [PubMed](https://huggingface.co/datasets/common-pile/pubmed) и страница [Kaggle arXiv](https://www.kaggle.com/datasets/Cornell-University/arxiv) открыты как ссылки на источники данных. Страница Kaggle не предоставила текст для извлечения; фактический состав локального arXiv подтверждается config и CSV, не содержимым этой страницы. Карточки датасетов не подтверждают точный исторический revision.

## Допустимые выводы

45 пар подтверждают среднее снижение completion PPL 7.4047% на известном словаре. Внешняя NPMI в среднем меняется на −0.002731. На 27 общих парах почти весь эффект даёт инициализация. На arXiv training PMI AARTM+Q выше AARTM во всех 20 парах; сильные альтернативы не доведены до общего проверенного критерия сходимости. Полного измерения времени от исходного текста нет. Эти ограничения сохранены в аннотации, введении, эксперименте и README.

Неподтверждённая анонимная рукопись AARTM не добавлялась в библиографию; формулы базовой модели опираются на предоставленный локальный код проекта. Новые результаты обучаемого внимания, SOTA-утверждения, прогоны на миллионе документов и выводы об ускорении на порядок не включены.
