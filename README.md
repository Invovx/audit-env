# AuditEnv — RL-среда для ИИ-аудитора (Genesis, трек FinTech)

Воспроизводимый симулятор: генератор вымышленных компаний со скрытыми финансовыми
ошибками, среда в стиле Gymnasium (`reset` / `step`), изолированный оценщик и
терминальная награда. Все компании, люди и деньги вымышлены.

## Проверка одной командой

```bash
docker build -t audit-env .
docker run --rm --network none audit-env
```

Без Docker (Python 3.11):

```bash
pip install -r requirements.txt
python check.py
```

`check.py` по очереди:

1. запускает все тесты (`pytest`);
2. генерирует 10 train и 10 dev эпизодов из `configs/train.json` и `configs/dev.json`;
3. прогоняет 4 базовые политики: случайный исследователь, простая сверка, атака «пометить всё», oracle;
4. пишет `artifacts/trajectories.jsonl` и `artifacts/report.json`;
5. проверяет инварианты (oracle = 1.0, атака наказана, награда в [-1, 1], ≤ 60 с на эпизод) и
   завершается с кодом 1, если что-то не так.

Сеть, GPU и платные API не нужны. Время полной проверки — около 5 секунд на 2 CPU.

## Структура

| Файл | Что внутри |
|---|---|
| `schemas.py` | контракты, лимиты, валидаторы файлов, разбор `submit`, деньги и FX через `Decimal` |
| `generator.py` | `generate_episode(seed, config)`: чистая история → проверка → внесение ошибок → файлы, `truth`, `provenance` |
| `env.py` | `AuditEnv`: `reset(seed, config)`, `step(action)`, 5 инструментов, лимит 100 шагов |
| `safe_calc.py` | калькулятор на `ast` + `Decimal`, без `eval` |
| `evaluator.py` | сопоставление один к одному, P/R/F1, награда; не импортирует генератор и среду |
| `baselines.py` | 4 базовые политики |
| `runner.py` | запуск агента в отдельном процессе, общение только JSON-строками |
| `agent_protocol.py` | сторона агента для `runner.py` + тестовые агенты (random, crash, hostile…) |
| `check.py` | единая команда проверки |
| `configs/` | версионированные конфигурации train/dev (`config_version`) |
| `tests/` | тесты контракта, изоляции, генератора, калькулятора, сопоставления и награды |
| `artifacts/` | сгенерированные эпизоды, траектории, отчёт |
| `DESIGN.md` | записка о дизайне |

## Использование среды

```python
from env import AuditEnv

env = AuditEnv(mode="train")          # "eval": награда и диагностика остаются у оценщика
obs, info = env.reset(seed=1000, config={"n_ledger_rows": 600, "n_documents": 60, "n_errors": 4})
episode_id = obs["manifest"]["episode_id"]
obs, reward, terminated, truncated, info = env.step(
    {"tool": "list_records", "table": "invoices", "cursor": None, "limit": 50})
obs, reward, terminated, truncated, info = env.step(
    {"tool": "submit", "findings": {"episode_id": episode_id, "findings": []}})
```

Действия: `list_records(table, cursor, limit≤50)`, `get_record(table, id)`, `get_document(doc_id)`,
`calculator(expression)`, `submit(findings)`. Таблицы: `ledger`, `invoices`, `payments`, `fx`.

Награда: `clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1)`, выдаётся только при завершении.

## Агент в отдельном процессе

```bash
python runner.py --seed 1000 --dummy-agent random
```

Эталон и оценщик остаются в процессе `runner.py`; агент видит только наблюдения.

## Файлы эпизода

`artifacts/episodes/<split>/<episode_id>/`

- `public/` — то, что агент может увидеть через инструменты: `manifest.json`, `policy.json`, `ledger.csv`,
  `invoices.jsonl`, `payments.jsonl`, `fx.csv`;
- `private/` — только для оценщика: `truth.json`, `provenance.json`;
- `config.json` — seed и конфигурация, из которых эпизод воспроизводится байт в байт.

## Траектории

`artifacts/trajectories.jsonl`, одна строка на шаг:
`episode_id, step, action, observation, reward, terminated, truncated` (+ `policy`, `split`).
Траектории oracle не публикуются: его действия содержат эталон.

## Лицензия

MIT, см. `LICENSE`. Зависимости зафиксированы в `requirements.txt`.
