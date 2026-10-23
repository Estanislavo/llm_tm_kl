# Воспроизведение

Команды выполняются из корня репозитория. Для модели исходная версия Python — 3.12.5; версии исследовательского окружения сохранены в [small_environment.json](../data/configs/small_environment.json).

## Таблицы и графики без обучения моделей

```bash
python -m pip install -r code/requirements-report.txt
python code/reproduce_report.py
```

Команда читает CSV из `data/results`, пересчитывает все таблицы и два рисунка в `docs/tables` и `docs/figures`. Проценты считаются внутри пар, SD — выборочное (`n-1`). Проверяются 45 основных пар, 27 сочетаний каждой абляции, 20 пар arXiv, 80 строк сравнительной arXiv-серии, контрольные суммы источников и непересечение train/reference/evaluation во всех 45 манифестах. Результат проверки — [verification.json](../docs/tables/verification.json).

## Минимальный запуск AARTM+Q

```bash
python -m pip install -r code/requirements-model.txt
python -m nltk.downloader punkt_tab stopwords
python code/run_demo.py --output docs/build/demo.npz
```

Демонстрация использует девять искусственно составленных английских текстов из [demo_documents.json](../data/demo_documents.json). Она строит контекстный учитель и выполняет исходное обновление AARTM+Q, проверяет конечность и нормировку матрицы, печатает top-10 слов тем. При первом запуске загружается MiniLM; revision фиксирован. Результат демонстрации **не включён** в таблицы статьи и не является новым сравнительным экспериментом.

`aartm_q/core.py`, `semantic_aartm.py` и `preprocessing.py` экспортированы из локального исследовательского проекта без изменений. Происхождение и SHA-256 — в [provenance.json](../data/provenance.json). В реализации сохранены дополнительные исторические аргументы для совместимости; демонстрация задаёт только фиксированный контекстный учитель, инициализацию 0.95 и KL 0.1. Дополнительные обучаемые параметры внимания отсутствуют.

## Сборка PDF

Нужны XeLaTeX, BibTeX, стандартные пакеты из `docs/main.tex` и шрифты Times New Roman, Arial, Consolas.

PowerShell:

```powershell
./docs/build.ps1
```

Другие системы — из `docs/`:

```bash
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
```

Готовый файл: [docs/main.pdf](../docs/main.pdf).

## Граница воспроизводимости

CSV позволяют повторно вычислить опубликованные сводки. Манифесты содержат хеши текстов и обучающий словарь. Для точного полного переобучения старых серий не хватает зафиксированных ревизий исходных датасетов и контрольной суммы arXiv snapshot. Потоковая CUDA-инфраструктура и веса большой серии здесь не публикуются. Минимальная демонстрация не заменяет эти серии.
