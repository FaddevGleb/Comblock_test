# GigaChat synthetic news pipeline v1

Скрипт `gigachat_synthetic_news_pipeline_v1.py` создаёт синтетические редакторские вариации уже собранных Telegram-постов.

## Что делает пайплайн

1. Берёт исходные записи из `data/telegram_news_dataset.jsonl`.
2. Исправляет старую CP1251/UTF-8 mojibake-кодировку локальных выгрузок перед отправкой в модель.
3. Для каждого канала передаёт его реальную style mask:
   - `style_masks/style_banki_oil.json`;
   - соответствующий профиль из `style_masks/style_masks_all.json`.
4. Добавляет два few-shot примера из `train_type1.jsonl` и `train_type2.jsonl`.
5. Включает `function_call: auto` и встроенные функции GigaChat, чтобы модель могла использовать поиск при проверке событий.
6. Использует `response_format: json_schema` со строгой схемой. Модель должна вернуть только `fragment`, `fact_basis`, `search_used` и `search_summary`.
7. Сохраняет исходный пост, результат, style mask, признаки текста и сведения о подтверждённых фактах.

## Настройка

Скопируйте `.env.gigachat.example` в `.env` и заполните `GIGACHAT_AUTH_KEY`. Ключ не должен попадать в исходный код, Git или итоговые JSON-файлы. Короткоживущий access token не нужно хранить: скрипт сам получает его через OAuth и обновляет примерно раз в 30 минут.

Установка зависимостей:

```powershell
python -m pip install requests python-dotenv
```

Примеры запуска:

```powershell
python gigachat_synthetic_news_pipeline_v1.py --limit 20
python gigachat_synthetic_news_pipeline_v1.py --channel banki_oil --limit 100
```

По умолчанию создаются:

- `synthetic_news_gigachat_v1.json` — объект с реальными масками и массивом записей;
- `synthetic_news_gigachat_v1.jsonl` — записи в форме, близкой к `telegram_news_dataset.jsonl`.

Важное ограничение: включение `function_call: auto` разрешает модели обратиться к встроенному поиску, но API не гарантирует поиск для каждого запроса. Поэтому результат содержит `search_used` и `fact_basis`; записи без подтверждения нужно отфильтровать перед обучением.
