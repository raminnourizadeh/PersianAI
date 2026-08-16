"""Small, dependency-free XLSX analytics engine for the HR assistant."""

from collections import Counter
from pathlib import Path
import json
import re
import statistics
import xml.etree.ElementTree as ET
import zipfile


XML_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _normalize(value):
    text = "" if value is None else str(value)
    text = text.replace("ي", "ی").replace("ك", "ک").replace("ة", "ه")
    text = text.replace("\u200c", " ")
    return re.sub(r"\s+", " ", text).strip()


HEADER_ALIASES = {
    "کدشناسایی": "کد شناسایی",
    "نام خانو ادگی": "نام خانوادگی",
    "و احد محل خدمت": "واحد محل خدمت",
    "تاریخ اجر ای حکم": "تاریخ اجرای حکم",
    "طلب ازمرخصی ساعتی": "طلب مرخصی ساعتی",
    "طلب ازمرخصی روزانه": "طلب مرخصی روزانه",
    "کدهزینه": "کد هزینه",
    "عنو ان پست": "عنوان پست",
}

DATE_FIELDS = {
    "تاریخ تولد", "تاریخ اجرای حکم", "تاریخ اخذ مدرک", "تاریخ استخدام",
}

NUMERIC_FIELDS = {
    "طلب مرخصی ساعتی", "طلب مرخصی روزانه",
    "تعداد فرزند تحت تکفل", "گروه شغل",
}


def _clean_number(value):
    if not value:
        return ""
    text = value.replace(",", "")
    if re.fullmatch(r"[-+]?\d+\.0+", text):
        return text.split(".", 1)[0]
    return text


def _clean_date(value):
    text = _normalize(value).replace(" ", "")
    if not text or text == "//":
        return ""
    match = re.fullmatch(r"(\d{4})/(\d{1,2})/(\d{1,2})", text)
    if not match:
        return ""
    year, month, day = map(int, match.groups())
    max_day = 31 if month <= 6 else 30
    if not 1200 <= year <= 1499 or not 1 <= month <= 12 or not 1 <= day <= max_day:
        return ""
    return f"{year:04d}/{month:02d}/{day:02d}"


def _clean_text(value):
    text = _normalize(value)
    text = re.sub(r"\s*\(\s*", " (", text)
    text = re.sub(r"\s*\)\s*", ")", text)
    text = re.sub(r"\s+", " ", text).strip()
    replacements = {
        "نا مشخص": "نامشخص",
        "نامشخص ": "نامشخص",
        "قرارداد دایم": "قرارداد دائم",
        "قراداد داخلی": "قرارداد داخلی",
        "بیسواد": "بی سواد",
        "اولراهنمایی": "اول راهنمایی",
        "ابتدائی": "ابتدایی",
    }
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    return text


def clean_value(header, value):
    if header in DATE_FIELDS:
        return _clean_date(value)
    if header in NUMERIC_FIELDS:
        return _clean_number(value)
    return _clean_text(value)


def _column_index(reference):
    letters = re.match(r"[A-Z]+", reference or "A").group(0)
    result = 0
    for letter in letters:
        result = result * 26 + ord(letter) - 64
    return result - 1


def read_xlsx(path):
    """Read the first worksheet of a basic XLSX file using the standard library."""
    with zipfile.ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", XML_NS):
                shared.append("".join(node.text or "" for node in item.iter(
                    "{%s}t" % XML_NS["m"]
                )))
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        result = []
        for row in sheet.findall(".//m:sheetData/m:row", XML_NS):
            cells = {}
            for cell in row.findall("m:c", XML_NS):
                index = _column_index(cell.attrib.get("r"))
                value_node = cell.find("m:v", XML_NS)
                value = "" if value_node is None else value_node.text or ""
                if cell.attrib.get("t") == "s" and value:
                    value = shared[int(value)]
                elif cell.attrib.get("t") == "inlineStr":
                    value = "".join(node.text or "" for node in cell.iter(
                        "{%s}t" % XML_NS["m"]
                    ))
                cells[index] = _normalize(value)
            if cells:
                width = max(cells) + 1
                result.append([cells.get(index, "") for index in range(width)])
    return result


class HRDataset:
    SENSITIVE_FIELDS = {
        "شماره ملی", "شماره شناسنامه", "شماره بیمه", "نام پدر",
    }
    FILTER_FIELDS = {
        "عنوان جنسیت", "عنوان مدرک تحصیلی", "عنوان نوع استخدام",
        "عنوان وضعیت اشتغال", "عنوان حوزه محل خدمت",
        "عنوان واحد محل خدمت", "عنوان تاهل", "عنوان رشته تحصیلی",
        "عنوان رشته شغل", "محل تولد", "محل صدور",
    }
    NUMERIC_FIELDS = {"تعداد فرزند تحت تکفل", "گروه شغل"}

    def __init__(self, path):
        self.path = Path(path)
        rows = read_xlsx(self.path)
        if len(rows) < 2:
            raise ValueError("فایل منابع انسانی فاقد رکورد است.")
        self.headers = self._unique_headers(rows[0])
        self.cleaning_report = Counter()
        self.records = []
        for row in rows[1:]:
            if not any(row):
                continue
            record = {}
            for index, header in enumerate(self.headers):
                raw_value = row[index] if index < len(row) else ""
                cleaned_value = clean_value(header, raw_value)
                if cleaned_value != raw_value:
                    self.cleaning_report[header] += 1
                record[header] = cleaned_value
            self.records.append(record)
        self.values = self._build_values()
        self.quality_report = self._quality_report()

    @staticmethod
    def _unique_headers(headers):
        seen = Counter()
        result = []
        for index, raw in enumerate(headers, start=1):
            header = HEADER_ALIASES.get(_normalize(raw), _normalize(raw)) or f"ستون {index}"
            seen[header] += 1
            result.append(header if seen[header] == 1 else f"{header} ({seen[header]})")
        return result

    def _build_values(self):
        values = {}
        for header in self.headers:
            counter = Counter(record[header] for record in self.records if record[header])
            values[header] = counter
        return values

    def _quality_report(self):
        def duplicate_count(field):
            values = [record.get(field) for record in self.records if record.get(field)]
            return len(values) - len(set(values))

        national_ids = [record.get("شماره ملی", "") for record in self.records]
        return {
            "ردیف‌های تکراری بر اساس کد شناسایی": duplicate_count("کد شناسایی"),
            "ردیف‌های تکراری بر اساس شماره ملی": duplicate_count("شماره ملی"),
            "شماره ملی خالی یا با قالب نامعتبر": sum(
                not re.fullmatch(r"\d{10}", value) for value in national_ids
            ),
            "تاریخ تولد ناموجود": sum(
                not record.get("تاریخ تولد") for record in self.records
            ),
            "تاریخ استخدام ناموجود": sum(
                not record.get("تاریخ استخدام") for record in self.records
            ),
        }

    @staticmethod
    def _number(value):
        try:
            return float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return None

    def _mentioned_fields(self, question):
        question = _normalize(question)
        aliases = {
            "جنسیت": ("جنسیت", "زن", "مرد"),
            "عنوان مدرک تحصیلی": ("تحصیلات", "مدرک", "کارشناسی", "دیپلم"),
            "عنوان نوع استخدام": ("نوع استخدام", "استخدام", "قرارداد"),
            "عنوان وضعیت اشتغال": ("وضعیت اشتغال", "شاغل", "بازنشسته"),
            "عنوان حوزه محل خدمت": ("حوزه", "محل خدمت"),
            "عنوان واحد محل خدمت": ("واحد", "معاونت"),
            "عنوان پست سازمانی": ("پست", "سمت"),
            "عنوان رشته تحصیلی": ("رشته تحصیلی",),
            "عنوان تاهل": ("تاهل", "تأهل", "متاهل", "مجرد"),
            "تعداد فرزند تحت تکفل": ("فرزند", "تکفل"),
            "گروه شغل": ("گروه شغل", "گروه شغلی"),
            "تاریخ استخدام": ("سابقه", "تاریخ استخدام"),
            "تاریخ تولد": ("سن", "تاریخ تولد"),
        }
        fields = []
        for preferred, terms in aliases.items():
            matching = next((h for h in self.headers if _normalize(h) == preferred), None)
            if matching and any(term in question for term in terms):
                fields.append(matching)
        return fields

    def _filters(self, question):
        normalized = _normalize(question)
        padded_question = f" {normalized} "
        filters = {}
        # Longest matching category per field avoids matching both
        # «کارشناسی» and «کارشناسی ارشد» in the same question.
        for header, counter in self.values.items():
            if header not in self.FILTER_FIELDS:
                continue
            candidates = []
            for value in counter:
                normalized_value = _normalize(value)
                if len(normalized_value) < 2 or value.replace(".", "").isdigit():
                    continue
                exact_phrase = f" {normalized_value} " in padded_question
                inflected_gender = normalized_value in {"زن", "مرد"} and any(
                    form in normalized.split()
                    for form in (normalized_value, f"{normalized_value}ان")
                )
                if exact_phrase or inflected_gender:
                    candidates.append(value)
            if candidates:
                filters[header] = max(candidates, key=len)
        first_name = next((h for h in self.headers if h == "نام"), None)
        family_name = next((h for h in self.headers if "نام خانو" in h), None)
        if first_name and family_name:
            for record in self.records:
                full_name = _normalize(
                    f"{record.get(first_name, '')} {record.get(family_name, '')}"
                )
                reverse_name = _normalize(
                    f"{record.get(family_name, '')} {record.get(first_name, '')}"
                )
                if len(full_name) >= 5 and (full_name in normalized or reverse_name in normalized):
                    filters[first_name] = record[first_name]
                    filters[family_name] = record[family_name]
                    break
        return filters

    def _matching_records(self, filters):
        return [
            record for record in self.records
            if all(record.get(field) == value for field, value in filters.items())
        ]

    def _distribution(self, records, field, limit=25):
        counts = Counter(record.get(field) for record in records if record.get(field))
        return dict(counts.most_common(limit))

    def query_context(self, question):
        filters = self._filters(question)
        records = self._matching_records(filters)
        fields = self._mentioned_fields(question)
        if not fields:
            defaults = (
                "عنوان وضعیت اشتغال", "عنوان جنسیت", "جنسیت",
                "عنوان نوع استخدام", "عنوان مدرک تحصیلی",
            )
            fields = [h for h in self.headers if h in defaults][:4]

        distributions = {
            field: self._distribution(records, field)
            for field in fields
            if field not in self.SENSITIVE_FIELDS
        }
        numeric = {}
        numeric_candidates = set(fields).intersection(self.NUMERIC_FIELDS)
        numeric_candidates.update(h for h in self.headers if h in self.NUMERIC_FIELDS)
        for field in numeric_candidates:
            numbers = [self._number(record.get(field)) for record in records]
            numbers = [number for number in numbers if number is not None]
            if numbers:
                numeric[field] = {
                    "تعداد مقدار معتبر": len(numbers),
                    "مجموع": round(sum(numbers), 2),
                    "میانگین": round(statistics.fmean(numbers), 2),
                    "کمینه": min(numbers),
                    "بیشینه": max(numbers),
                }

        # Only expose full rows when the query narrows the data substantially.
        samples = []
        if filters and len(records) <= 20:
            safe_headers = [h for h in self.headers if h not in self.SENSITIVE_FIELDS]
            samples = [
                {h: record[h] for h in safe_headers if record.get(h)}
                for record in records[:20]
            ]

        payload = {
            "نام فایل": self.path.name,
            "تعداد کل کارکنان": len(self.records),
            "فیلترهای تشخیص داده شده": filters,
            "تعداد رکورد منطبق": len(records),
            "تفکیک‌ها": distributions,
            "آمار عددی": numeric,
            "رکوردهای منطبق بدون شناسه‌های حساس": samples,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def metadata(self):
        return {
            "ready": True,
            "file": self.path.name,
            "records": len(self.records),
            "columns": len(self.headers),
            "cleaned_values": sum(self.cleaning_report.values()),
            "cleaning_by_column": dict(self.cleaning_report),
            "quality": self.quality_report,
        }
