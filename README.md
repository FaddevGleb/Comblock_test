# Submission package v33

Готовый комплект для тестового задания.

Обязательные файлы:

- `outputs_type1.txt` — 5 генераций для аналитического/финансового типа;
- `outputs_type2.txt` — 5 генераций для городского новостного типа;
- `RESULTS.md` — описание решения, v30/v33, synthetic-data pipeline и vLLM.

Основной вариант решения — v33: общая 4-bit базовая модель и два отдельных DoRA-адаптера, `banki_oil` и `moscowach` с масками стиля по расширенному пулу сообщений, few-shot prompting

`outputs_type1.txt` и `outputs_type2.txt` сформированы по `data/test_inputs.jsonl`. Validation-отчёты находятся в `reports/`.
