# Broker Reports to Snowball CSV

Утилита для конвертации HTML брокерских отчетов брокера Ньютон (более известного в широких кругах как Газпромбанк) в CSV-формат импорта [Snowball Income](https://snowball-income.com/).

## Что делает

- читает один или несколько `*.html` отчетов;
- извлекает сделки (`Buy`/`Sell`) и денежные операции;
- формирует единый CSV в формате Snowball (`Event,Date,Symbol,...`).

Поддерживаемые типы событий: `Buy`, `Sell`, `Dividend`, `Amortisation`, `Fee`, `Cash_In`, `Cash_Out`, `Cash_Gain`, `Cash_Expense`.

## Требования

- Python 3.9+;
- внешние зависимости не требуются (только стандартная библиотека).

## Быстрый старт

Конвертация всех отчетов в директории:

```bash
python3 convert_broker_reports.py /path/to/reports -o snowball_income_import.csv
```

Конвертация конкретных файлов:

```bash
python3 convert_broker_reports.py report1.html report2.html -o snowball_income_import.csv
```

## Примечания

- Скрипт убирает дубли сделок, которые могут повторяться в разных секциях отчета.
- Неторговые операции с ценными бумагами из секции `8.2` сейчас пропускаются с предупреждением.
