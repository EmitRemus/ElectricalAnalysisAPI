"""Readable documentation for the HTTP API and its import contract."""
import json
from urllib.parse import urlencode

from flask import Blueprint, render_template, request

from ..datasets.contract import DEFINITIONS, ROOT, TABLES


documentation = Blueprint("documentation", __name__)
OPENAPI = json.loads((ROOT / "docs/openapi.json").read_text(encoding="utf-8"))
EXAMPLE = json.loads((ROOT / "examples/dataset.json").read_text(encoding="utf-8"))

GROUPS = {
    "datasets": "Імпорт наборів",
    "readings": "Покази лічильників",
    "customers": "Споживачі",
    "equipment": "Обладнання",
    "service-cases": "Вартість послуг",
}
RECORDS = {
    "reading": ("Показ лічильника", "Обсяг енергії за один інтервал і час надходження показу."),
    "meter": ("Паспорт лічильника", "Тип підключення, тариф і характеристики об’єкта."),
    "customer_profile": ("Профіль споживача", "Ознаки споживання, нарахувань і переходу до іншого постачальника."),
    "customer_event": ("Подія споживача", "Датована подія, пов’язана з профілем за customer_id."),
    "equipment": ("Паспорт обладнання", "Реєстровий запис для зіставлення зі спостереженнями за equipment_id."),
    "equipment_observation": ("Спостереження обладнання", "Виміряні ознаки навантаження, температури й часткових розрядів."),
    "service_case": ("Випадок обслуговування", "Вхідні ознаки для моделі вартості. Ціна відома для навчальних випадків."),
}
OPERATIONS = {
    "importDataset": ("Зберігає один набір і повертає dataset_id для подальших запитів.", ()),
    "listReadings": (
        "Повертає останні покази в кВт·год у хронологічному порядку. Задайте start і end для періоду, "
        "meter_id для одного лічильника та limit і offset для поділу на сторінки.",
        ("reading",),
    ),
    "summarizeReadings": (
        "Рахує суму, середнє на один показ, медіану, мінімум і максимум у кВт·год. "
        "Без фільтрів підсумовує весь набір.",
        ("reading",),
    ),
    "summarizeReadingsByDay": (
        "Повертає статистику окремо за кожен день із показами. День визначається за початком інтервалу в UTC.",
        ("reading",),
    ),
    "downloadDeduplicatedReadings": (
        "Для кожного лічильника та інтервалу залишає останню передачу. Результат завантажується як файл Parquet.",
        ("reading",),
    ),
    "reportReadingQuality": (
        "Рахує повтори, виявляє замовклі й завмерлі лічильники. Додайте meter для перевірки масштабу енергії.",
        ("reading",),
    ),
    "fitHouseholdRegression": (
        "Оцінює добове споживання з урахуванням електроопалення та двозонного тарифу. Потрібні погодинні покази та принаймні три придатні домогосподарства з незалежною варіацією ознак.",
        ("reading", "meter"),
    ),
    "segmentCustomers": (
        "Повертає профілі й мітки сегментів, PCA та порівняння k-середніх із методом Варда. Потрібно щонайменше k + 1 профілів і k різних векторів ознак.",
        ("customer_profile",),
    ),
    "trainCustomerMaps": (
        "Повертає дві карти 15 × 15, координати споживачів і топографічну помилку. Діють ті самі вимоги до вибірки, що й для сегментації.",
        ("customer_profile",),
    ),
    "findCustomerChurnRules": (
        "Виявляє поєднання подій, пов’язані з переходом. Розрахунок використовує споживачів із відомим churned.",
        ("customer_profile", "customer_event"),
    ),
    "reportCustomerPitfalls": (
        "Перевіряє зв’язки за районами, сезонність за типами лічильників і зв’язок ціни з переходом.",
        ("customer_profile",),
    ),
    "reportEquipmentAudit": (
        "Перевіряє витік цільової змінної, покриття історичної когорти та зсув середнього електричного навантаження між вибірками.",
        ("equipment", "equipment_observation"),
    ),
    "evaluateEquipmentFailureModel": (
        "Навчає одну вибрану модель й повертає її PR-AUC і ROC AUC. У навчальній та валідаційній частинах мають бути обидва класи.",
        ("equipment", "equipment_observation"),
    ),
    "assessMeterInspectionPriority": (
        "Оцінює пріоритет ручної перевірки за часом від останнього показу й часткою повторюваних значень. Бал є евристикою для сортування заявок на перевірку.",
        ("reading",),
    ),
    "getMeterInspectionPrioritySurface": (
        "Показує, як змінюється бал за різної затримки й частки повторюваних значень. Повертає сітку 21 × 21 без імпорту.",
        (),
    ),
    "countSubtractiveServiceCostRules": (
        "Рахує потенційні правила кластеризації для одного радіуса. Потрібні щонайменше три випадки з відомою вартістю та мінливими навчальними цінами.",
        ("service_case",),
    ),
    "evaluateSugenoServiceCostModel": (
        "Порівнює похибку моделі Сугено з прогнозом середнім і оцінює вартість випадків без ціни. У навчальній частині має бути щонайменше стільки випадків, скільки правил.",
        ("service_case",),
    ),
}
FIELD_NOTES = {
    "record_type": "Позначка, за якою імпортер визначає вид запису.",
    "schema_version": "Версія контракту набору.",
    "source_timezone": "Часова зона джерела за базою IANA.",
    "records": "Предметні записи. У масиві можна поєднувати різні типи.",
    "meter_id": "Ідентифікатор лічильника.",
    "customer_type": "Тип споживача, наприклад домогосподарство або малий бізнес.",
    "tariff": "Код тарифу. Регресія домогосподарств використовує single і dual_zone.",
    "electric_heating": "Наявність електроопалення.",
    "connection_year": "Рік підключення.",
    "years_connected": "Кількість років із моменту підключення.",
    "district": "Код району.",
    "interval_start": "Початок інтервалу вимірювання.",
    "interval_minutes": "Тривалість інтервалу у хвилинах.",
    "received_at": "Час надходження передачі.",
    "energy": "Енергія за інтервал у заявлених одиницях.",
    "energy_unit": "Одиниця енергії для цього показу.",
    "customer_id": "Ідентифікатор споживача.",
    "avg_daily_kwh": "Середнє добове споживання у кВт·год.",
    "night_share": "Частка нічного споживання.",
    "seasonality": "Показник сезонності споживання.",
    "weekend_ratio": "Показник співвідношення споживання у вихідні.",
    "stability": "Показник стабільності споживання.",
    "annual_charge_uah": "Річні нарахування у гривнях.",
    "avg_price_uah_kwh": "Середня ціна у гривнях за кВт·год.",
    "voltage_dev_mean": "Середній показник відхилення напруги.",
    "loss_index_mean": "Середній індекс втрат.",
    "meter_type": "Код типу лічильника.",
    "churned": "Ознака переходу до іншого постачальника. За невідомої мітки поле пропускають.",
    "event_at": "Час події.",
    "event_type": "Код події, наприклад complaint.",
    "equipment_id": "Ідентифікатор об’єкта обладнання.",
    "cohort": "Когорта реєстру. Аудит покриття використовує history.",
    "equipment_type": "Тип обладнання.",
    "voltage_class": "Клас напруги.",
    "region": "Регіон встановлення.",
    "install_year": "Рік встановлення.",
    "last_overhaul_year": "Рік останнього капітального ремонту.",
    "outdoor": "Ознака встановлення просто неба.",
    "role": "Роль спостереження під час аналізу.",
    "load_mean": "Середнє навантаження.",
    "load_std": "Стандартне відхилення навантаження.",
    "load_stability": "Показник стабільності навантаження.",
    "load_weekend_ratio": "Показник співвідношення навантаження у вихідні.",
    "temp_rise_mean": "Середній приріст температури.",
    "temp_rise_std": "Стандартне відхилення приросту температури.",
    "temp_rise_median": "Медіанний приріст температури.",
    "pd_mean": "Середній рівень часткових розрядів.",
    "pd_std": "Стандартне відхилення рівня часткових розрядів.",
    "pd_stability": "Показник стабільності часткових розрядів.",
    "pd_per_load": "Показник часткових розрядів на одиницю навантаження.",
    "temp_per_load": "Показник температури на одиницю навантаження.",
    "work_order_opened": "Ознака відкритого наряду. Використовується в аудиті витоку.",
    "failed": "Чи відмовило обладнання. Обов’язково для training, заборонено для prediction.",
    "case_id": "Ідентифікатор випадку обслуговування.",
    "reactive_energy_mvarh": "Реактивна енергія у Мвар·год.",
    "heat_energy_gcal": "Теплова енергія у Гкал.",
    "contracted_power_kw": "Договірна потужність у кВт.",
    "outage_hours": "Тривалість перерви в годинах.",
    "service_cost_kuah": "Відома вартість у тисячах гривень. Пропустіть для прогнозування.",
}
PARAMETER_NOTES = {
    "dataset_id": "Ідентифікатор із відповіді успішного імпорту.",
    "silence_hours": "Поріг затримки останнього показу в годинах.",
    "frozen_ratio": "Мінімальна частка різних значень показів.",
    "unit_ratio": "Поріг відношення медіани лічильника до медіани його групи.",
    "k": "Кількість сегментів.",
    "min_confidence": "Мінімальна умовна частка переходів для правила.",
    "model": "Класифікатор, який буде навчено.",
    "radius": "Радіус субтрактивної кластеризації.",
    "rules": "Кількість правил моделі Сугено.",
}
RESPONSE_NOTES = {
    "dataset_id": "Ідентифікатор збереженого набору.",
    "duplicate_readings": "Кількість вилучених повторів.",
    "silent_meters": "Лічильники, які перевищили поріг затримки.",
    "frozen_meters": "Лічильники із замалою часткою різних значень.",
    "unit_check": "Перевірка масштабу показів за типом споживача.",
    "intercept": "Базове добове споживання у кВт·год.",
    "coefficients": "Внески електроопалення і тарифу у кВт·год за добу.",
    "r_squared": "Коефіцієнт R² на навчальній вибірці.",
    "adjusted_rand_index": "Скоригований індекс Ренда для двох поділів.",
    "assignments": "Ідентифікатори споживачів і їхні мітки сегментів.",
    "maps": "Дві карти, значення похибки й координати клієнтів.",
    "reports": "Правила й метрики для кожного порога підтримки.",
    "m1": "Витік цільової ознаки.",
    "m2": "Покриття історичної когорти.",
    "meter_id": "Лічильник, якого стосується оцінка.",
    "last_interval_start": "Початок останнього інтервалу з показом.",
    "hours_since_last_reading": "Години від початку останнього інтервалу до контрольного часу.",
    "reading_count": "Кількість показів після дедуплікації.",
    "unique_values": "Кількість різних значень енергії в кВт·год.",
    "unique_ratio": "Частка різних значень серед очищених показів.",
    "repetition_ratio": "Одиниця мінус частка різних значень.",
    "rule_strengths": "Сили правил планової та першочергової перевірки.",
    "priority_scores": "Рядки відповідають частці повторюваних значень, стовпці — годинам затримки.",
    "rule_count": "Кількість правил.",
    "predictions": "Прогнозовані ціни випадків без відомої вартості.",
}


def json_value(value):
    return json.dumps(value, ensure_ascii=False)


def resolve_openapi(schema):
    ref = schema.get("$ref", "")
    if ref.startswith("#/"):
        target = OPENAPI
        for key in ref[2:].split("/"):
            target = target[key.replace("~1", "/").replace("~0", "~")]
        return {**target, **{key: value for key, value in schema.items() if key != "$ref"}}
    return schema


def schema_type(schema):
    if "$ref" in schema:
        return schema["$ref"].rsplit("/", 1)[-1]
    if "const" in schema:
        return "константа"
    if "enum" in schema and "type" not in schema:
        return "перелік"
    if "oneOf" in schema:
        return " | ".join(schema_type(variant) for variant in schema["oneOf"])
    kind = schema.get("type")
    if isinstance(kind, list):
        return " | ".join(kind)
    if kind == "array":
        return f"array<{schema_type(schema.get('items', {}))}>"
    if kind and schema.get("format"):
        return f"{kind} ({schema['format']})"
    return kind or "JSON"


def constraints(schema):
    labels = {
        "minimum": "≥", "maximum": "≤", "exclusiveMinimum": ">", "exclusiveMaximum": "<",
        "minLength": "Мін. символів", "minItems": "Мін. елементів", "maxItems": "Макс. елементів",
        "default": "Типово",
    }
    limits = [f"{label} {json_value(schema[key])}" for key, label in labels.items() if key in schema]
    if "const" in schema:
        limits.append(f"Значення {json_value(schema['const'])}")
    if "enum" in schema:
        limits.append(" | ".join(map(json_value, schema["enum"])))
    return limits


def fields(schema, notes, example=None):
    example = example or {}
    return [
        {
            "name": name,
            "kind": schema_type(rule),
            "required": "Умовно" if name == "failed" and "allOf" in schema else
                        "Так" if name in schema.get("required", []) else "Ні",
            "description": rule.get("description", notes.get(name, "")),
            "limits": constraints(rule),
            "example": json_value(example[name]) if name in example else "",
        }
        for name, rule in schema.get("properties", {}).items()
    ]


def describe_operation(path, method, operation):
    parameters = [resolve_openapi(param) for param in operation.get("parameters", [])]
    query = {
        param["name"]: param.get("example", param["schema"].get("default"))
        for param in parameters if param["in"] == "query"
    }
    query = urlencode({key: value for key, value in query.items() if value is not None})
    url = request.url_root.rstrip("/") + path.replace("{dataset_id}", "$id")
    if query:
        url += "?" + query
    command = f'curl.exe "{url}"'
    if method == "post":
        command = f'curl.exe -i -H "Content-Type: application/json" --data-binary "@examples/dataset.json" "{url}"'
    elif path.endswith("/deduplicated"):
        command = f'{command} --output readings-deduplicated.parquet'

    responses = []
    for status, definition in operation["responses"].items():
        response = resolve_openapi(definition)
        content = response.get("content", {})
        schema = resolve_openapi(content.get("application/json", {}).get("schema", {}))
        output = fields(schema, RESPONSE_NOTES) if status.startswith("2") else []
        responses.append({"status": status, "description": response["description"], "formats": list(content),
                          "fields": output, "success": status.startswith("2")})
    purpose, records = OPERATIONS.get(operation["operationId"], ("", ()))
    return {
        "id": operation["operationId"],
        "summary": operation["summary"],
        "method": method.upper(),
        "path": path,
        "purpose": purpose,
        "records": records,
        "command": command,
        "formats": list(operation.get("requestBody", {}).get("content", {})),
        "parameters": [
            {"name": param["name"], "location": "Шлях" if param["in"] == "path" else "Query",
             "required": "Так" if param.get("required") else "Ні", "kind": schema_type(param["schema"]),
             "description": param.get("description", PARAMETER_NOTES.get(param["name"], "")),
             "limits": constraints(param["schema"])}
            for param in parameters
        ],
        "responses": responses,
    }


@documentation.get("/docs", strict_slashes=False)
def reference():
    groups = {key: [] for key in GROUPS}
    for path, methods in OPENAPI["paths"].items():
        parts = path.split("/")
        key = parts[4] if "{dataset_id}" in parts and len(parts) > 4 else parts[2]
        for method, operation in methods.items():
            groups.setdefault(key, []).append(describe_operation(path, method, operation))
    return render_template("documentation/reference.html", groups=[
        {"key": key, "title": GROUPS.get(key, key), "operations": operations}
        for key, operations in groups.items() if operations
    ], openapi_version=OPENAPI["openapi"])


@documentation.get("/docs/swagger", strict_slashes=False)
def swagger():
    return render_template("documentation/swagger.html")


@documentation.get("/docs/data", strict_slashes=False)
def input_data():
    examples = {}
    for record in EXAMPLE["records"]:
        examples.setdefault(record["record_type"], record)
    dataset = {**EXAMPLE, "records": [examples["reading"]]}
    header = {"record_type": "dataset", **{key: value for key, value in EXAMPLE.items() if key != "records"}}
    return render_template(
        "documentation/data.html",
        json_example=json.dumps(dataset, ensure_ascii=False, indent=2),
        ndjson_example="\n".join(map(json_value, (header, examples["reading"]))),
        formats=list(OPENAPI["paths"]["/v1/datasets"]["post"]["requestBody"]["content"]),
        envelope=[
            ("JSON", fields(DEFINITIONS["dataset"], FIELD_NOTES, dataset)),
            ("NDJSON", fields(DEFINITIONS["header"], FIELD_NOTES, header)),
        ],
        records=[
            {"key": key, "title": title, "description": note, "table": TABLES[key],
             "fields": fields(DEFINITIONS[key], FIELD_NOTES, examples.get(key))}
            for key, (title, note) in RECORDS.items()
        ],
    )
