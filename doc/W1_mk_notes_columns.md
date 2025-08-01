## Client table

| Column | Description (English) | Beschreibung (Deutsch) | X/y |
|--------|-----------------------|-------------------------|-----|
| district | District where the client is located | Bezirk, in dem sich der Kunde befindet | X |
| client_id | Unique identifier for the client | Eindeutige Kennung des Kunden | X |
| client_catg | Category to which the client belongs | Kategorie, der der Kunde angehört | X |
| region | Area/region where the client operates | Region, in der der Kunde tätig ist | X |
| creation_date | Date the client record was created | Datum, an dem der Kunde angelegt wurde | X |
| target | Fraud label: 1 = fraud, 0 = not fraud | Betrugskennzeichen: 1 = Fraud, 0 = kein Fraud | y |

## Invoice table

| Column | Description (English) | Beschreibung (Deutsch) | X/y |
|--------|-----------------------|-------------------------|-----|
| client_id | Unique identifier for the client | Eindeutige Kennung des Kunden | X |
| invoice_date | Date the invoice was issued | Ausstellungsdatum der Rechnung | X |
| tarif_type | Tariff category applied to the invoice | Tarifart der Rechnung | X |
| counter_number | Meter (counter) serial number | Zähler­nummer | X |
| counter_statue | Current status of the meter (e.g. working, faulty, on hold) | Aktueller Status des Zählers (z. B. funktionsfähig, defekt, pausiert) | X |
| counter_code | Internal code identifying the meter | Interner Zähler­code | X |
| reading_remarque | Inspector notes taken during meter reading | Ablese­bemerkung des Technikers | X |
| counter_coefficient | Extra coefficient applied when standard consumption is exceeded | Zusatz­koeffizient bei Überschreitung des Standard­verbrauchs | X |
| consommation_level_1 | Consumption recorded in level 1 tier | Verbrauchs­stufe 1 | X |
| consommation_level_2 | Consumption recorded in level 2 tier | Verbrauchs­stufe 2 | X |
| consommation_level_3 | Consumption recorded in level 3 tier | Verbrauchs­stufe 3 | X |
| consommation_level_4 | Consumption recorded in level 4 tier | Verbrauchs­stufe 4 | X |
| old_index | Previous meter reading | Alter Zähler­stand | X |
| new_index | Current meter reading | Neuer Zähler­stand | X |
| months_number | Number of months covered by the invoice | Abgerechnete Monats­anzahl | X |
| counter_type | Type of meter (e.g. single-phase, three-phase) | Zähler­typ | X |