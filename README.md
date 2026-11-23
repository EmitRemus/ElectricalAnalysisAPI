# ElectricalAnalysisAPI

Python-сервіс для імпорту наборів електрообліку v1. Специфікація HTTP міститься у docs/openapi.json, опис полів — у docs/data-contract.md.

## Структура Python-коду

| Папка | Призначення |
|---|---|
| `electrical_api/web/` | Маршрути Flask, параметри HTTP і довідник у браузері. |
| `electrical_api/datasets/` | Контракт даних, імпорт JSON і NDJSON, зберігання Parquet. |
| `electrical_api/readings/` | Дедуплікація, якість показів, регресія споживання та пріоритет перевірки лічильника. |
| `electrical_api/customers/` | Профілі, сегментація, карти Кохонена, правила переходу та статистичні перевірки. |
| `electrical_api/equipment/` | Аудит даних обладнання й оцінювання моделей відмов. |
| `electrical_api/service_cases/` | Кластеризація й моделі вартості обслуговування. |
| `tests/` | Автоматичні перевірки API та його розрахунків. |

## Запуск у VS Code на Windows

Відкрийте папку проєкту та виберіть інтерпретатор .venv через Python: Select Interpreter. Виконайте в PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m flask --app electrical_api.app:create_app run
```

У другому терміналі з тієї самої папки:

```powershell
curl.exe -i -H "Content-Type: application/json" --data-binary "@examples/dataset.json" http://127.0.0.1:5000/v1/datasets
curl.exe -i -H "Content-Type: application/x-ndjson" --data-binary "@examples/dataset.ndjson" http://127.0.0.1:5000/v1/datasets
```

Кожен успішний запит повертає HTTP 201, новий dataset_id і кількість записів за типами. Таблиці та metadata.json доступні в data/<dataset_id>/. Змінна середовища ELECTRICAL_DATA_DIR задає інше місце збереження. JSON обробляється цілим документом, NDJSON — по рядку з буферами до 8192 записів на таблицю. Вихідні повтори залишаються у readings.parquet.

## Документація у браузері

- [Довідник API](http://127.0.0.1:5000/docs) містить маршрути, параметри, приклади викликів і відповіді.
- [Вхідні дані](http://127.0.0.1:5000/docs/data) містить формати JSON та NDJSON і таблиці полів усіх типів записів.
- [Swagger UI](http://127.0.0.1:5000/docs/swagger) дозволяє виконувати HTTP-запити у браузері.
- [OpenAPI JSON](http://127.0.0.1:5000/docs/openapi.json) доступний для імпорту в клієнти й генератори коду.

У Swagger UI розгорніть `POST /v1/datasets`, виберіть JSON або NDJSON у списку типів вмісту, натисніть **Try it out** і **Execute**. Скопіюйте `dataset_id` з відповіді `201` і введіть його в `GET /v1/datasets/{dataset_id}/readings/quality`.

Файли Swagger UI постачаються разом із застосунком у `electrical_api/static/vendor/swagger-ui/`. [Випуск Swagger UI 5.36.0](https://github.com/swagger-api/swagger-ui/releases/tag/v5.36.0)

## Завантаження вимірювань після дедуплікації

Підставте `dataset_id`, отриманий під час імпорту.

```powershell
curl.exe --fail --output readings-deduplicated.parquet "http://127.0.0.1:5000/v1/datasets/<dataset_id>/readings/deduplicated"
```

Для демонстраційного набору файл містить одне вимірювання з енергією `1.45 kWh`.

## Перегляд показів і підсумки споживання

Підставте `dataset_id`, отриманий під час імпорту. Ці запити повертають покази за період, статистику за весь набір, статистику для одного лічильника за період і підсумки за днями.

```powershell
curl.exe --fail "http://127.0.0.1:5000/v1/datasets/<dataset_id>/readings?start=2026-01-14&end=2026-01-16&limit=20"
curl.exe --fail "http://127.0.0.1:5000/v1/datasets/<dataset_id>/readings/summary"
curl.exe --fail "http://127.0.0.1:5000/v1/datasets/<dataset_id>/readings/summary?meter_id=UA100035&start=2026-01-14&end=2026-01-16"
curl.exe --fail "http://127.0.0.1:5000/v1/datasets/<dataset_id>/readings/daily?start=2026-01-14&end=2026-01-16"
```

`start` включає нижню межу, `end` виключає верхню. Дати без часу позначають північ UTC. Енергія повертається в кВт·год, `average_kwh` є середнім на один показ. Для наступної сторінки передайте `next_offset` у параметрі `offset`.

### Фільтрування за значенням

Додайте `gt` (більше), `gte` (більше або дорівнює), `lt` (менше), `lte` (менше або дорівнює) чи `eq` (дорівнює) до будь-якого з трьох маршрутів. Для діапазону задайте обидві межі. Усі задані умови застосовуються одночасно.

```powershell
curl.exe --fail "http://127.0.0.1:5000/v1/datasets/<dataset_id>/readings?gt=1.4"
curl.exe --fail "http://127.0.0.1:5000/v1/datasets/<dataset_id>/readings/summary?gte=1&lte=2"
curl.exe --fail "http://127.0.0.1:5000/v1/datasets/<dataset_id>/readings/daily?filter_field=interval_minutes&eq=60"
```

За замовчуванням порівнюється `energy_kwh`. Вкажіть `filter_field=interval_minutes`, щоб відбирати покази за тривалістю інтервалу у цілих хвилинах. Підсумки використають лише відібрані покази. Фільтри можна поєднати з `meter_id`, `start` та `end`.

## Звіт про якість вимірювань

```powershell
curl.exe --fail "http://127.0.0.1:5000/v1/datasets/<dataset_id>/readings/quality?window_end=2026-01-18T07:00:00Z"
```

Звіт повертає кількість повторних передач, замовклі та завмерлі лічильники й підозри на помилковий масштаб погодинної енергії. Усі параметри необов’язкові. Пороги наведені в [контракті](docs/data-contract.md#якість-вимірювань).

## Регресія споживання домогосподарств

Для набору з погодинними вимірюваннями й реєстром `meters` виконайте запит.

```powershell
curl.exe --fail-with-body "http://127.0.0.1:5000/v1/datasets/<dataset_id>/readings/household-regression"
```

Модель повертає базове добове споживання, коефіцієнти електроопалення й двозонного тарифу та навчальний R². Потрібні щонайменше три придатні домогосподарства з незалежною варіацією ознак. [Правила регресії](docs/data-contract.md#регресія-споживання-домогосподарств)

## Сегментація споживачів

Для набору з таблицею `customer_profiles` виконайте запит.

```powershell
curl.exe --fail-with-body "http://127.0.0.1:5000/v1/datasets/<dataset_id>/customers/segmentation?k=4"
```

Відповідь містить PCA, порівняння k-середніх і методу Варда, профілі сегментів та мітки споживачів. Для `k=4` потрібні щонайменше п’ять профілів із чотирма різними векторами ознак. [Правила сегментації](docs/data-contract.md#сегментація-споживачів)

## Карта Кохонена

```powershell
curl.exe --fail-with-body "http://127.0.0.1:5000/v1/datasets/<dataset_id>/customers/som?k=4"
```

Відповідь містить дві карти Кохонена, їхні топографічні помилки й координати споживачів на сітці. Параметр `k` визначає мітки сегментів споживачів. [Правила побудови карти](docs/data-contract.md#карта-кохонена)

## Асоціативні правила переходу

```powershell
curl.exe --fail-with-body "http://127.0.0.1:5000/v1/datasets/<dataset_id>/customers/churn-rules"
```

Для набору з профілями, відомим `churned` і подіями API повертає асоціативні правила для підтримки `0.05` та `0.10`. Параметри `min_support` і `min_confidence` дають змогу задати власні пороги. [Контракт звіту](docs/data-contract.md#асоціативні-правила-переходу-до-іншого-постачальника)

## Перевірки T-1, T-2 і T-3

```powershell
curl.exe --fail-with-body "http://127.0.0.1:5000/v1/datasets/<dataset_id>/customers/pitfalls"
```

Звіт повертає результати трьох перевірок із кореляціями, середніми значеннями, порогами та результатом `detected`. [Контракт звіту](docs/data-contract.md#перевірки-t-1-t-2-і-t-3)

## Аудит даних обладнання

```powershell
curl.exe --fail-with-body "http://127.0.0.1:5000/v1/datasets/<dataset_id>/equipment/audit"
```

Звіт повертає підозрілі ознаки за ROC AUC, покриття когорти `history` і зсув середнього навантаження `load_mean` між `training` та `prediction`. [Контракт звіту](docs/data-contract.md#аудит-даних-обладнання)

## Оцінювання моделі відмов обладнання

```powershell
curl.exe --fail-with-body "http://127.0.0.1:5000/v1/datasets/<dataset_id>/equipment/model-evaluation?model=perceptron&alpha=0.01"
```

Один запит навчає одну вибрану модель та повертає PR‑AUC і ROC AUC для навчальної й валідаційної частин. За замовчуванням використовується перцептрон з `alpha=1.0`. `pr_auc_gap` показує різницю між навчальним і валідаційним PR‑AUC. [Моделі й підготовка ознак](docs/data-contract.md#оцінювання-моделі-відмов-обладнання)

## Пріоритет перевірки лічильника

```powershell
curl.exe --fail-with-body "http://127.0.0.1:5000/v1/datasets/<dataset_id>/readings/inspection-priority?meter_id=UA100035&window_end=2026-01-17T06:00:00Z"
curl.exe --fail-with-body "http://127.0.0.1:5000/v1/readings/inspection-priority/surface"
```

Перший запит оцінює пріоритет ручної перевірки методом Мамдані. Для демонстраційного набору він повертає 50 балів за 48 годин від останнього показу. Другий повертає сітку балів для різних затримок і часток повторюваних значень. [Правила нечіткої оцінки](docs/data-contract.md#пріоритет-перевірки-лічильника)

## Нейро-нечітка модель вартості з КП

```powershell
curl.exe --fail-with-body "http://127.0.0.1:5000/v1/datasets/<dataset_id>/service-cases/subtractive-clustering?radius=0.5"
curl.exe --fail-with-body "http://127.0.0.1:5000/v1/datasets/<dataset_id>/service-cases/cost-model?rules=3"
```

Маршрути використовують імпортовані `service_cases`. Перший повертає кількість правил субтрактивної кластеризації. Другий звіряє RMSE моделі Сугено з прогнозом середнім і оцінює випадки без заданої вартості. [Контракт КП](docs/data-contract.md#нейро-нечітка-модель-вартості-з-кп)

## Автоматична перевірка

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Тести надсилають запити через тестовий HTTP-клієнт Flask і читають створені Parquet-файли з тимчасової папки. Команди curl вище перевіряють запущений сервер через мережевий порт.

## Окрема демонстраційна програма

Запустіть Flask, а в другому терміналі виконайте команду з кореня проєкту.

```powershell
.\.venv\Scripts\python.exe -B examples/api_client_demo.py
```

Файл `examples/meter_readings.ndjson` містить покази двох лічильників за 26 січня — 1 лютого 2026 року з `ElectricalAnalysisSources/AnalysisData/meters.parquet`. Клієнт надсилає файл через `POST /v1/datasets`, викликає `GET /v1/datasets/{dataset_id}/readings/quality` і друкує таблицю у терміналі.
