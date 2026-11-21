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

## Подключение жюри: `judge_adapter.py`

Точки входа по опубликованному протоколу проверок (20.11.2026). Рабочая директория в контейнере — `/app`
(она же в `PYTHONPATH`), Python 3.11, сеть не нужна:

```bash
docker build -t audit-env .
docker run --rm --network none audit-env                       # полная проверка (check.py)
docker run --rm --network none audit-env python -c "import judge_adapter"   # точка входа жюри
```

```python
import judge_adapter as J
J.generate_episode(seed, team_config, output_dir)        # пишет output_dir/public/* и output_dir/private/*
env = J.make_env()
obs, info = env.reset(seed, {"mode": "closed", "episode_dir": output_dir, "team_config": {}})
obs, reward, terminated, truncated, info = env.step({"name": "list_records",
                                                     "args": {"table": "ledger", "cursor": None, "limit": 50}})
J.score_episode(output_dir, findings_json, inspection_steps)
```

| Что | Как устроено |
|---|---|
| Каталог эпизода | `public/` — manifest.json, policy.json, ledger.csv, invoices.jsonl, payments.jsonl, fx.csv; `private/` — truth.json, provenance.json. Плоский вариант (все файлы в одном каталоге) тоже читается |
| `reset` с `episode_dir` | загружает переданный эпизод как есть: ничего не генерирует, ID не меняет; повторный reset начинает его заново. Без `episode_dir` — генератор команды с `seed` и `team_config` |
| Режимы | `closed`: reward политике всегда 0, в info нет диагностики; `train`: терминальная награда и счётчики TP/FP/FN в info |
| Первое наблюдение | `{"task", "manifest", "policy" (весь policy.json), "remaining_steps": 100}` |
| Следующие | `{"result", "error", "remaining_steps"}`; ошибка — `{"code", "message"}` без путей, seed и введённых агентом строк |
| Результаты действий | list_records → `{"records", "next_cursor"}`; get_record / get_document → объект; calculator → десятичная строка (`"0.3"`); submit → `{"accepted": true}` или ошибка структуры |
| Таблицы | `ledger` (ID = row_id), `invoices` и `payments` (ID = doc_id), `fx` (ID = `FX-<YYYY-MM-DD>-<SRC>-<TGT>`, выводится из даты и пары валют) |
| Лимиты | 100 действий вместе с submit; submit не входит в inspection_steps; 100 исследований без submit → truncated, пустой ответ; 60 с от reset (с временем политики) → truncated |
| `score_episode` | валидный submit → `{tp, fp, fn, precision, recall, f1, return}`; невалидная структура → `return = -1`, остальные шесть полей `null` |

**Matcher** (`evaluator.py`): affected_ids и evidence_ids — множества точных строк; TP требует тип, точное
множество affected_ids, валюту, сумму в пределах τ и полный допустимый набор доказательств, а **все**
evidence_ids должны существовать среди row_id, doc_id и fx ID эпизода (иначе FP, пример M04).
Допуск τ = `policy.amount_tolerance_minor`, по умолчанию **0** (точное сравнение). Паросочетание —
максимальное (augmenting paths), повторный вывод — отдельный FP.

**Четыре типа ошибок** и канонические affected_ids:

| Тип | affected_ids | Сумма и валюта |
|---|---|---|
| `duplicate_invoice` | оба doc_id счетов-дублей | сумма счёта, валюта счёта |
| `partial_payment_allocation` | doc_id платежа + doc_id счёта | неприменённая сумма, валюта платежа |
| `fx_rate_error` | doc_id счёта или платежа с неверным курсом | \|проведено − верно\| в базовой валюте |
| `period_cutoff_error` | doc_id счёта | сумма счёта, валюта счёта |

Сценарии взаимодействия (`team_config.interactions`): `partial_fx_boundary`, `cutoff_fx`.

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
| `judge_adapter.py` | точки входа жюри: generate_episode, make_env, score_episode |
| `episode_io.py` | чтение каталога эпизода с диска (ID — непрозрачные строки) |
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
