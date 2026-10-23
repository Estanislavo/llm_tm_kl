# Фиксированный контекстный учитель для AARTM

Предобученная языковая модель задаёт распределения тем для слов. Эти распределения используются для инициализации и KL-регуляризации AARTM; кодировщик при обучении не меняется.

[Текст работы в PDF](docs/main.pdf) · [LaTeX](docs/main.tex) · [Литература](docs/references.bib)

В PDF собраны аннотация, короткие введение и постановка задачи, эксперименты. Формулировки взяты за основу из статьи и курсовой. Разделы для редактирования находятся в [docs/sections](docs/sections/).

В [data](data/) сохранены результаты запусков, в [code](code/) — модель и пересчёт таблиц. Полные сводки остаются в [docs/tables](docs/tables/), в краткий текст включены две таблицы и один рисунок.

Пересчитать результаты:

```bash
python -m pip install -r code/requirements-report.txt
python code/reproduce_report.py
```

Собрать PDF в PowerShell: `./docs/build.ps1`. Подробности запуска — в [code/README.md](code/README.md), источники корпусов — в [data/README.md](data/README.md).

Презентация экспериментов: [PDF](exp.pdf), [PowerPoint](exp.pptx).
