## Client table

| Column | Description (English) | Beschreibung (Deutsch) |
|--------|-----------------------|-------------------------|
| district | District where the client is located | Bezirk, in dem sich der Kunde befindet |
| client_id | Unique identifier for the client | Eindeutige Kennung des Kunden |
| client_catg | Category to which the client belongs | Kategorie, der der Kunde angehört |
| region | Area/region where the client operates | Region, in der der Kunde tätig ist |
| creation_date | Date the client record was created | Datum, an dem der Kunde angelegt wurde |
| target | Fraud label: 1 = fraud, 0 = not fraud | Betrugskennzeichen: 1 = Fraud, 0 = kein Fraud |

## Invoice table

| Column | Description (English) | Beschreibung (Deutsch) |
|--------|-----------------------|-------------------------|
| client_id | Unique identifier for the client | Eindeutige Kennung des Kunden |
| invoice_date | Date the invoice was issued | Ausstellungsdatum der Rechnung |
| tarif_type | Tariff category applied to the invoice | Tarifart der Rechnung |
| counter_number | Meter (counter) serial number | Zähler­nummer |
| counter_statue | Current status of the meter (e.g. working, faulty, on hold) | Aktueller Status des Zählers (z. B. funktionsfähig, defekt, pausiert) |
| counter_code | Internal code identifying the meter | Interner Zähler­code |
| reading_remarque | Inspector notes taken during meter reading | Ablese­bemerkung des Technikers |
| counter_coefficient | Extra coefficient applied when standard consumption is exceeded | Zusatz­koeffizient bei Überschreitung des Standard­verbrauchs |
| consommation_level_1 | Consumption recorded in level 1 tier | Verbrauchs­stufe 1 |
| consommation_level_2 | Consumption recorded in level 2 tier | Verbrauchs­stufe 2 |
| consommation_level_3 | Consumption recorded in level 3 tier | Verbrauchs­stufe 3 |
| consommation_level_4 | Consumption recorded in level 4 tier | Verbrauchs­stufe 4 |
| old_index | Previous meter reading | Alter Zähler­stand |
| new_index | Current meter reading | Neuer Zähler­stand |
| months_number | Number of months covered by the invoice | Abgerechnete Monats­anzahl |
| counter_type | Type of meter (e.g. single-phase, three-phase) | Zähler­typ |