-- MUST BE RUN AFTER labels.sql

-- creates a materialized view timeseries which looks like this:
/*
 ventid | measuredat |       item        | value
--------+------------+-------------------+-------
      0 |       -498 | CRP (bloed)       |     7
      0 |       -498 | Leuco's (bloed)   |   6.5
      0 |        342 | Ademfreq.         |    13
      0 |        342 | Hartfrequentie    |    83
      0 |        342 | ABP gemiddeld     |    68
      0 |        342 | Exp. tidal volume |   443
*/

/*
Extracts:
- ventid | integer | unique value to identify an individual ventilation episode
- measuredat | integer | number of minutes since the person was admitted to the ICU (N.B. can be negative if the ventilation is started before admission, although this only applies to 6 ventilation episodes)
- item | string | name of time series feature
- value | float | value of the item
*/

-- added APACHE IV parameters, thrombos because they are part of SOFA, and troponin because it was so predictive in eICU (ref TPC for LoS prediction paper)
-- no clean way of doing GCS scores
-- see table under preprocessing investigation for the reasons why these variables were chosen
drop materialized view if exists timeseries cascade;
create materialized view timeseries as
  select l.ventid, cast(n.measuredat as float)/(1000*60) as measuredat, n.item, n.value
    from numericitems as n
    inner join labels as l
      on n.admissionid = l.admissionid  -- only include the common respiratory chart values
    where n.item in ('PEEP (Set)', 'Ademfreq.', 'Exp. tidal volume', 'O2 concentratie', 'Piek druk',
    'PO2 (bloed)', 'pCO2 (bloed)', 'pH (bloed)', 'Saturatie (Monitor)', 'End tidal CO2 concentratie', 'Hartfrequentie',
    'ABP gemiddeld', 'Alb.Chem (bloed)', 'CRP (bloed)', 'Lactaat (bloed)', 'Leuco''s (bloed)', 'Kreatinine (bloed)',
    'Bilirubine (bloed)', 'UrineCAD', 'TroponineT (bloed)', 'Natrium (bloed)', 'Kalium (bloed)', 'Glucose (bloed)',
    'Temp Bloed', 'Temp Axillair', 'Thrombo''s (bloed)', 'Ht (bloed)')  -- the two temperature variables can be combined later
      and cast(n.measuredat as float)/(1000*60) between (l.ventstart - 24*60) and l.ventstop
      order by l.ventid, n.measuredat;


/* SANITY CHECKS
-- check if the variables that have been extracted are sufficiently filled in:

select t.item, count(distinct l.ventid)
  from timeseries as t
    inner join labels as l
      on l.ventid = t.ventid
  group by t.item
  order by count desc;

            item            | count
----------------------------+-------
 Hartfrequentie             | 14836
 Saturatie (Monitor)        | 14826
 ABP gemiddeld              | 14812
 UrineCAD                   | 14618
 Glucose (bloed)            | 14165
 Thrombo's (bloed)          | 14126
 Kreatinine (bloed)         | 14081
 pH (bloed)                 | 14059
 PO2 (bloed)                | 14059
 pCO2 (bloed)               | 14059
 PEEP (Set)                 | 13925
 Exp. tidal volume          | 13918
 Ademfreq.                  | 13914
 Piek druk                  | 13912
 O2 concentratie            | 13902
 Leuco's (bloed)            | 13657
 Ht (bloed)                 | 13469
 Natrium (bloed)            | 13135
 Alb.Chem (bloed)           | 13123
 Kalium (bloed)             | 13091
 End tidal CO2 concentratie | 11607
 Lactaat (bloed)            | 10395
 CRP (bloed)                |  8558
 Bilirubine (bloed)         |  8067
 Temp Bloed                 |  6241
 Temp Axillair              |  6229
 TroponineT (bloed)         |  5112
 */

/* PREPROCESSING INVESTIGATION

-- it doesn't seem like the variables labelled with APACHE are labelled well enough to be used.
-- we'll need to extract them separately

-- extract the most common numeric results and the corresponding counts of how many ventilation episodes have these entries
drop materialized view if exists commonnumeric cascade;
create materialized view commonnumeric as
  select n.item, count(distinct l.ventid)
    from numericitems as n
      inner join labels as l
        on l.admissionid = n.admissionid
      where cast(n.measuredat as float)/(1000*60) between (l.ventstart - 24*60) and l.ventstop  -- collect events throughout ventilation duration but also in the day leading up
    group by n.item
    -- only keep data that is present at some point for at least 25% of the patients
    having count(distinct l.ventid) > (select count(distinct ventid) from labels)*0.25
    order by count desc;

                 item                 | count
--------------------------------------+-------
 Hartfrequentie                       | 14836 **
 Saturatie (Monitor)                  | 14826 **
 ABP systolisch                       | 14812
 ABP diastolisch                      | 14812
 ABP gemiddeld                        | 14812 **
 UrineCAD                             | 14618 **
 Glucose (bloed)                      | 14165 **
 Thrombo's (bloed)                    | 14126 **
 Kreatinine (bloed)                   | 14081 **
 B.E. (bloed)                         | 14059
 pH (bloed)                           | 14059 **
 pCO2 (bloed)                         | 14059 **
 Act.HCO3 (bloed)                     | 14059
 PO2 (bloed)                          | 14059 **
 O2-Saturatie (bloed)                 | 14055
 Calcium totaal (bloed)               | 14010
 Fosfaat (bloed)                      | 13999
 Magnesium (bloed)                    | 13942
 PEEP (Set)                           | 13925 **
 Insp. tidal volume                   | 13921
 Exp. minuut volume                   | 13921
 Exp. tidal volume                    | 13918 **
 O2 concentratie (Set)                | 13916
 Ademfreq.                            | 13914 **
 Eind exp. druk                       | 13912
 Piek druk                            | 13912 **
 Mean luchtweg druk                   | 13912
 Insp. Minuut volume                  | 13909
 O2 concentratie                      | 13902 **
 P Bovenste (Set)                     | 13891
 CK (bloed)                           | 13689
 Hb (bloed)                           | 13677
 Leuco's (bloed)                      | 13657 **
 Adem Frequentie (Set)                | 13593
 Ht (bloed)                           | 13469 **
 PC boven PEEP (Set)                  | 13386
 PS boven PEEP (Set)                  | 13305
 ST segment analyse afleiding II      | 13238
 ST segment analyse afleiding         | 13161
 Natrium (bloed)                      | 13135 **
 Alb.Chem (bloed)                     | 13123 **
 Kalium (bloed)                       | 13091 **
 K (onv.ISE) (bloed)                  | 12793
 Insp. Rise time (Set, %) nieuw       | 12704
 WOBv                                 | 12688
 Cdyn                                 | 12688
 Insp. Rise time (Set, sec) nieuw     | 12667
 Na (onv.ISE) (bloed)                 | 12557
 Insp. Time (sec) nieuw               | 12416
 ST segment analyse afleiding I       | 12408
 CVD                                  | 12360
 Ti (sec) nieuw                       | 12317
 Eind insp. cyclus (Set)              | 12297
 Ademfrequentie Monitor               | 12173
 Barometer druk                       | 12163
 Hb(v.Bgs) (bloed)                    | 12162
 Ht(v.Bgs) (bloed)                    | 12160
 MVe Spontaan nieuw                   | 12152
 CO2 min prod                         | 12127
 CO2 tidal productie                  | 12125
 SBI (Zwakke ademhalingsindex)        | 12010
 End tidal CO2 concentratie           | 11607 **
 EtCO2 (%)                            | 11516
 Backup druk (Set)                    | 11472
 Ca-ion (7.4) (bloed)                 | 11114
 Anion-Gap (bloed)                    | 10964
 SIMV ademcyclus (Set)                | 10424
 Lactaat (bloed)                      | 10395 **
 MB-Massa (bloed)                     | 10116
 OxyHb-Fr. (bloed)                    |  9510
 APTT  (bloed)                        |  9240
 Prothrombinetijd  (bloed)            |  9235
 Ureum (bloed)                        |  8970 **
 ALAT (bloed)                         |  8856
 ASAT (bloed)                         |  8834
 CRP (bloed)                          |  8558
 eGFR(MDRD) (bloed)                   |  8521
 Y-GT (bloed)                         |  8370
 LD (bloed)                           |  8365
 Alk.Fosf. (bloed)                    |  8332
 O2 l/min                             |  8285
 Meettemp. (bloed)                    |  8170
 Chloor (bloed)                       |  8084
 Bilirubine (bloed)                   |  8067 **
 Thoraxdrain1 Stand                   |  7740
 Thoraxdrain1 Productie               |  7713
 Maaghevel                            |  7639
 O2-Content (bloed)                   |  7474
 CO-Hb (bloed)                        |  7461
 Cl (onv.ISE) (bloed)                 |  7402
 Met-Hb  (bloed)                      |  7400
 PAP gemiddeld                        |  6623
 PAP systolisch                       |  6620
 PAP diastolisch                      |  6609
 Amylase (bloed)                      |  6571
 Volume (Set)                         |  6337
 Temp Bloed                           |  6241 **
 Temp Axillair                        |  6229 **
 Cardiac Output                       |  6106
 Pauze druk                           |  6049
 PCWP wedge                           |  6031
 Ontlasting                           |  5976
 Teugvolume (Set)                     |  5602
 Cuff druk van Tube                   |  5583
 MaagRetNietWeg                       |  5472
 Pause Time (sec) nieuw               |  5454
 TroponineT (bloed)                   |  5112 **
 Niet invasieve bloeddruk systolisch  |  5062
 Niet invasieve bloeddruk gemiddeld   |  5041
 Niet invasieve bloeddruk diastolisch |  5023
 Prothrombinetijd (bloed)             |  4939
 APTT (bloed)                         |  4801
 MaagRetentieWeg                      |  4638
 Ery's (bloed)                        |  4529
 Bezinking (bloed)                    |  3924
 IC NAS Item 7 score                  |  3796
 IC NAS Item 1 score                  |  3796
 IC NAS Item 4 score                  |  3796
 IC NAS Item 8 score                  |  3796
 IC NAS Item 6 score                  |  3796
 IC NAS score                         |  3795
 IC NAS Item 2 score                  |  3795
 IC NAS Item 3 score                  |  3771
 IC NAS Item 10 score                 |  3767

Translation:

 Heart rate | 14836
 Saturation (Monitor) | 14826
 ABP systolic | 14812
 ABP diastolic | 14812
 ABP average | 14812
 UrineCAD | 14618
 Glucose (blood) | 14165
 Thrombos (blood) | 14126
 Creatinine (blood) | 14081
 B.E. (blood) | 14059
 pH (blood) | 14059
 pCO2 (blood) | 14059
 Act.HCO3 (blood) | 14059
 PO2 (blood) | 14059
 O2 Saturation (blood) | 14055
 Calcium total (blood) | 14010
 Phosphate (blood) | 13999
 Magnesium (blood) | 13942
 PEEP (Set) | 13925
 Insp. tidal volume | 13921
 Exp. minute volume | 13921
 Exp. tidal volume | 13918
 O2 concentration (Set) | 13916
 Breath freq. | 13914
 End exp. pressure | 13912
 Peak pressure | 13912
 Mean airway pressure | 13912
 Insp. Minute volume | 13909
 O2 concentration | 13902
 P Upper (Set) | 13891
 CK (blood) | 13689
 Hb (blood) | 13677
 Leuco's (blood) | 13657
 Breath Frequency (Set) | 13593
 Ht (blood) | 13469
 PC over PEEP (Set) | 13386
 PS over PEEP (Set) | 13305
 ST segment analysis derivation II | 13238
 ST segment analysis derivation | 13161
 Sodium (blood) | 13135
 Alb.Chem (blood) | 13123
 Potassium (blood) | 13091
 K (inc.ISE) (blood) | 12793
 Insp. Rise time (Set,%) new | 12704
 WOBv | 12688
 Cdyn | 12688
 Insp. Rise time (Set, sec) new | 12667
 Na (on IE) (blood) | 12557
 Insp. Time (sec) new | 12416
 ST segment analysis derivation I | 12408
 CVD | 12360
 Ti (sec) new | 12317
 End insp. cycle (Set) | 12297
 Breath Rate Monitor | 12173
 Barometer pressure | 12163
 Hb (v.Bgs) (blood) | 12162
 Ht (v. BGs) (blood) | 12160
 MVe Spontaneous new | 12152
 CO2 min prod | 12127
 CO2 tidal production | 12125
 SBI (Weak Respiratory Index) | 12010
 End tidal CO2 concentration | 11607
 EtCO2 (%) | 11516
 Backup press (Set) | 11472
 Ca ion (7.4) (blood) | 11114
 Anion-Gap (blood) | 10964
 SIMV Breathing Cycle (Set) | 10424
 Lactate (blood) | 10395
 MB-Mass (blood) | 10116
 OxyHb-Fr. (blood) | 9510
 APTT (blood) | 9240
 Prothrombin time (blood) | 9235
 Urea (blood) | 8970
 ALAT (blood) | 8856
 ASAT (blood) | 8834
 CRP (blood) | 8558
 eGFR (MDRD) (blood) | 8521
 Y-GT (blood) | 8370
 LD (blood) | 8365
 Alk.Phosph. (blood) | 8332
 O2 l / min | 8285
 Measuring temp. (blood) | 8170
 Chlorine (blood) | 8084
 Bilirubin (blood) | 8067
 Chest Drain1 Stand | 7740
 Thorax Drain1 Production | 7713
 Gastric Siphon | 7639
 O2 Content (blood) | 7474
 CO-Hb (blood) | 7461
 Cl (on IISE) (blood) | 7402
 Met-Hb (blood) | 7400
 PAP average | 6623
 PAP systolic | 6620
 PAP diastolic | 6609
 Amylase (blood) | 6571
 Volume (Set) | 6337
 Temp Blood | 6241
 Temp Axillary | 6229
 Cardiac Output | 6106
 Break busy | 6049
 PCWP wedge | 6031
 Stool | 5976
 Tidal Volume (Set) | 5602
 Cuff pressure from Tube | 5583
 StomachRetNotWard | 5472
 Pause Time (sec) new | 5454
 TroponinT (blood) | 5112
 Non invasive blood pressure systolic | 5062
 Non invasive blood pressure average | 5041
 Non invasive blood pressure diastolic | 5023
 Prothrombin time (blood) | 4939
 APTT (blood) | 4801
 Stomach Retention Away | 4638
 Ery's (blood) | 4529
 Sedimentation (blood) | 3924
 IC NAS Item 7 score | 3796
 IC NAS Item 1 score | 3796
 IC NAS Item 4 score | 3796
 IC NAS Item 8 score | 3796
 IC NAS Item 6 score | 3796
 IC NAS score | 3795
 IC NAS Item 2 score | 3795
 IC NAS Item 3 score | 3771
 IC NAS Item 10 score | 3767

-- extract the most common listitem results and the corresponding counts of how many ventilation episodes have these entries
drop materialized view if exists commonlist cascade;
create materialized view commonlist as
  select li.item, count(distinct l.ventid)
    from listitems as li
      inner join labels as l
        on l.admissionid = li.admissionid
      where cast(li.measuredat as float)/(1000*60) between (l.ventstart - 24*60) and l.ventstop  -- collect events throughout ventilation duration but also in the day leading up
    group by li.item
    -- only keep data that is present at some point for at least 25% of the patients
    having count(distinct l.ventid) > (select count(distinct ventid) from labels)*0.25
    order by count desc;

select value, count(distinct admissionid) from numericitems where item = 'IC NAS score' group by value order by count desc;

                    item                    | count
--------------------------------------------+-------
 Hartritme                                  | 14758
 Ventilatie Mode (Set)                      | 13936
 PatiëntGeslacht                            | 12834
 Reden voor ontslag                         | 12285 doesn't seem to be useful - is related to outcome on discharge
 Patiënt Specialisme                        | 11946 (this is the same as specialty extracted in flat)
 Ontslagbestemming                          | 11851
 Bedsoort                                   | 11273
 PatiëntWijzeVanOpname                      | 10482
 Pupil Links Grootte                        | 10474
 Pupil Rechts Grootte                       | 10466
 Pupil Links Reactie                        | 10324
 Pupil Rechts Reactie                       | 10295
 Houding patiënt                            | 10158
 Ramsay score                               |  9779 could be a replacement for the gcs?
 Gewicht bron                               |  9696
 D_Hoofdgroep                               |  9601
 Lengte bron                                |  9498
 Sonde maat                                 |  9224
 Locatie plaatsing                          |  8905
 Contactpersoon 1 Relatie                   |  8879
 Aantal Bronchiaaltoilet                    |  8598
 Hoeveelheid Sputum                         |  8541
 Hoestprikkel                               |  8452
 Tube diepte                                |  8430
 Kleur Sputum                               |  8288
 Toedieningsweg                             |  8281
 Mond/Keel toilet                           |  7884
 Balloneren                                 |  7856
 Aspect Sputum                              |  7499
 Actief openen van de ogen                  |  7458
 Druppelen NaCl                             |  7416
 Beste verbale reactie                      |  7371 shame there isn't motor and eyes
 Thoraxdrain1 Plaats                        |  7359
 Beste motore reactie van de armen          |  7358
 Thoraxdrain1 Zuigkracht                    |  7323
 Thoraxdrain1 Transport                     |  7025
 Thoraxdrain1 Luchtlekkage                  |  6858
 Opname Sepsis                              |  6594 doesn't seem to be as useful as it looked
 Neus Toilet                                |  6567
 Sonde functie                              |  6280
 Contactpersoon 2 Relatie                   |  6125
 Sonde positie                              |  5963
 Tube referentiepunt                        |  5763
 Ectopie                                    |  5749
 Sonde type                                 |  5562
 Tube route                                 |  5397
 Tube maat                                  |  5338
 Sonde route                                |  5202
 Beleid                                     |  5173
 NICE Herkomst                              |  4660
 NICE Opname type                           |  4650
 Reactie op Uitzuigen                       |  4456
 D_Subgroep_Thoraxchirurgie                 |  4307
 IC NAS Mobiliseren en positioneren         |  3796
 IC NAS Ondersteuning en zorg naasten       |  3796
 IC NAS Procedures  hygiëne                 |  3796
 IC NAS Administratieve en management taken |  3796
 IC NAS Monitoren                           |  3796
 D_Thoraxchirurgie_CABG en Klepchirurgie    |  3757

Translation:

 Heart rhythm | 14758
 Ventilation Mode (Set) | 13936
 Patient Gender | 12834
 Reason for dismissal | 12285
 Patient Specialty | 11946
 Destination of dismissal | 11851
 Bed type | 11273
 Patient Mode Of Admission | 10482
 Pupil Left Size | 10474
 Pupil Right Size | 10466
 Pupil Links Response | 10324
 Pupil Right Response | 10295
 Posture patient | 10158
 Ramsay score | 9779
 Source weight | 9696
 D_Main group | 9601
 Source length | 9498
 Probe size | 9224
 Placement location | 8905
 Contact person 1 Relationship | 8879
 Number of Bronchial toilet | 8598
 Amount of Sputum | 8541
 Cough Stimulus | 8452
 Tube depth | 8430
 Color Sputum | 8288
 Route of administration | 8281
 Mouth / Throat toilet | 7884
 Ballooning | 7856
 Aspect Sputum | 7499
 Actively opening the eyes | 7458
 Drip NaCl | 7416
 Best Verbal Response | 7371
 Thorax Drain1 Location | 7359
 Best engine response from the poor | 7358
 Thorax drain1 Suction | 7323
 Thorax Drain1 Transport | 7025
 Chest drain1 Air leakage | 6858
 Recording Sepsis | 6594
 Nose Toilet | 6567
 Probe function | 6280
 Contact person 2 Relationship | 6125
 Probe position | 5963
 Tube reference point | 5763
 Ectopia | 5749
 Probe type | 5562
 Tube route | 5397
 Tube size | 5338
 Probe route | 5202
 Policy | 5173
 NICE Origin | 4660
 NICE Recording type | 4650
 Response to Suction | 4456
 D_Subgroup_Thorax surgery | 4307
 IC NAS Mobilizing and positioning | 3796
 IC NAS Support and care for loved ones | 3796
 IC NAS Hygiene procedures | 3796
 IC NAS Administrative and management tasks | 3796
 IC NAS Monitors | 3796
 D_Thoraxchirurgie_CABG and Valve surgery | 3757
 */
