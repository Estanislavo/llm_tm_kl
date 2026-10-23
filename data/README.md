# Данные и происхождение результатов

В репозитории находятся компактные результаты и метаданные. Полные корпуса и веса моделей сюда не загружаются.

| Файл | Содержание |
|---|---|
| [small_runs.csv](results/small_runs.csv) | 171 строка: AARTM, полный фиксированный учитель и три абляции. Числа в отобранных строках не изменены |
| [arxiv_runs.csv](results/arxiv_runs.csv) | 40 запусков AARTM/AARTM+Q: пять значений T, четыре seed |
| [arxiv_baseline_runs.csv](results/arxiv_baseline_runs.csv) | 80 строк AARTM, AARTM+Q, FASTopic, CombinedTM+MiniLM; для времени AARTM+Q T500 seed43/45 здесь учтены повторные измерения |
| [manifests](manifests/) | 45 манифестов: SHA-256 обучающих, reference и тестовых текстов; обучающий словарь |
| [configs](configs/) | Исходные конфигурации, окружение, локальный манифест корпусов |
| [provenance.json](provenance.json) | Пути источников относительно локального проекта, преобразования при экспорте, SHA-256 опубликованных файлов |
| [demo_documents.json](demo_documents.json) | Искусственная коллекция только для проверки выполнения модели |

Имена `semantic_confirmation_v1` и `semantic_transfer_v1` в исходных конфигурациях различают первую тройку корпусов и два дополнительных корпуса. Поле `stage: screen` в исходных JSON осталось от интерфейса runner; статус оценивания задаётся `evaluation: test`, конфигурации и seed сверены по сохранённым строкам. В конфигурациях присутствуют исторические `legacy_*` варианты, но в `small_runs.csv` и таблицы они не включены.

Некоторые исторические JSON содержат абсолютные пути исходного компьютера. Они являются метаданными происхождения, а команды пересчёта используют относительные пути этого репозитория.

## Источники корпусов

| Корпус | Источник | Использование |
|---|---|---|
| BBC | [SetFit/bbc-news](https://huggingface.co/datasets/SetFit/bbc-news) | 5 тем, 10/30/50 документов |
| 20NG | [SetFit/20_newsgroups](https://huggingface.co/datasets/SetFit/20_newsgroups) | 4 категории: comp.graphics, rec.sport.hockey, sci.space, talk.politics.misc |
| AG News | [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news) | 4 темы, 10/30/50 документов |
| TechCrunch | [abhilash88/techcrunch-articles](https://huggingface.co/datasets/abhilash88/techcrunch-articles) | 10 тем, 10/30/50 документов |
| PubMed | [common-pile/pubmed](https://huggingface.co/datasets/common-pile/pubmed) | Тексты статей PubMed Central, 10 тем |
| arXiv | [Cornell University / Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv) | Заголовки и аннотации, 99 049 документов после фильтрации |

Источник MiniLM — [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), revision малой серии `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Revision датасетов в исторических выгрузках не записаны. Размеры локальных выгрузок, предобработка, параметры и ограничения контроля приведены в разделе 3 PDF.
