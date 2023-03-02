-- MUST BE RUN AFTER labels.sql

-- creates a materialized view flat which looks like this:
/*
 patientid | admissionid | ventid | admissioncount | ventcount | urgency |                 origin                 | location |    specialty    | gender | agegroup | weightgroup | weightsource | heightgroup | heightsource
-----------+-------------+--------+----------------+-----------+---------+----------------------------------------+----------+-----------------+--------+----------+-------------+--------------+-------------+--------------
         0 |           0 |      0 |              1 |         1 | 0       |                                        | IC       | Cardiochirurgie | Vrouw  | 80+      | 60-69       | Anamnestisch | 160-169     | Anamnestisch
         1 |           1 |      1 |              1 |         1 | 0       |                                        | IC       | Cardiochirurgie | Man    | 60-69    | 70-79       | Anamnestisch | 170-179     | Anamnestisch
         3 |           3 |      3 |              1 |         1 | 0       |                                        | IC       | Cardiochirurgie | Man    | 50-59    | 90-99       |              | 180-189     | Gemeten
         4 |           4 |      4 |              1 |         1 | 0       | Verpleegafdeling zelfde ziekenhuis     | IC&MC    | Cardiochirurgie | Man    | 70-79    | 70-79       | Anamnestisch | 170-179     | Anamnestisch
         5 |           5 |      5 |              1 |         1 | 1       | Eerste Hulp afdeling zelfde ziekenhuis | IC       | Longziekte      | Man    | 50-59    | 60-69       | Geschat      | 160-169     | Gemeten
         6 |           6 |      6 |              1 |         1 | 1       | Verpleegafdeling ander ziekenhuis      | IC       | Neurochirurgie  | Vrouw  | 80+      | 70-79       | Geschat      | 160-169     | Geschat
*/

/*
Extracts:
- patientid | integer | unique value to identify individual patients throughout their admission(s)
- admissionid | integer | unique value to identify the admission to the ICU. Patients may have multiple admissionid's during the same or another hospitalization, when they are discharge from the ICU or MC unit. Admissionid is used as an identifier in all other tables
- ventid | integer | unique value to identify an individual ventilation episode
- admissioncount | integer | for each patient, each additional ICU/MCU admission will increase this counter by one
- ventcount | integer | for each patient, each additional ventilation episode will increase this counter by one
- urgency | bit | determines whether the admission was urgent (1) (i.e. unplanned) or not (0) (planned)
- origin | string | department the patient originated from (e.g. Emergency department, regular ward)
- location | string | the department the patient has been admitted to, either IC, MC or both (IC&MC or MC&IC)
- specialty | string | medical specialty the patient has been admitted for
- gender | string | gender, 'Man' for male, 'Vrouw' for female
- agegroup | string | age at admission (years), categorised
- weightgroup | string | weight at admission (kg), categorised
- weightsource | string | method used to determine the weight at ICU/MCU admission, either measured ('gemeten'), estimated ('geschat') or asked ('anamnestisch')
- heightgroup | string | height of the patient (cm), categorised
- heightsource | string | method used to determine the height at ICU/MCU admission, either measured ('gemeten'), estimated ('geschat') or asked ('anamnestisch')
*/

-- delete the materialized view flat if it already exists
drop materialized view if exists flat cascade;
create materialized view flat as
  -- select all the data we need from the admissions table
  select a.patientid, a.admissionid, l.ventid, a.admissioncount,
    row_number() over(partition by a.admissionid order by l.ventstart) as ventcount, a.urgency, a.origin, a.location,
    a.specialty, a.gender, a.agegroup, a.weightgroup, a.weightsource, a.lengthgroup as heightgroup,
    a.lengthsource as heightsource
    from admissions as a
    inner join labels as l on l.admissionid = a.admissionid
    order by a.patientid, a.admissionid, l.ventid;
