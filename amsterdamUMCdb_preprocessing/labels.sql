-- MUST BE RUN FIRST

-- creates a materialized view labels which looks like this, it contains 14,836 distinct ventilation episodes from 13,502 admissionids from 12,597 patientids:
/*
 patientid | admissionid | ventid | admittedat | dischargedat | lengthofstay | destination | ventstart | ventstop | ventduration | early_termination | actualventduration | admissionyeargroup | dateofdeath | trachstart |                            diagnosis                            |   diagnosissubgroup   |        diagnosisgroup         | diagnosistype  | surgical | sepsisatadmission |                sepsisantibiotics                 |         otherantibiotics
-----------+-------------+--------+------------+--------------+--------------+-------------+-----------+----------+--------------+-------------------+--------------------+--------------------+-------------+------------+-----------------------------------------------------------------+-----------------------+-------------------------------+----------------+----------+-------------------+--------------------------------------------------+----------------------------------
         0 |           0 |      0 |          0 |         2480 |           42 | 16          |       342 |     1152 |          810 | N/A               |                810 | 2003-2009          |             |            | CABG                                                            | CABG en Klepchirurgie | Thoraxchirurgie               | Legacy ICU     |        1 |                   |                                                  | Cefazoline (Kefzol)
         1 |           1 |      1 |          0 |         1602 |           26 | 15          |        35 |      440 |          405 | N/A               |                405 | 2010-2016          |             |            | Bentall                                                         | Aorta chirurgie       | Thoraxchirurgie               | Legacy ICU     |        1 |                 0 |                                                  | Cefazoline (Kefzol); Vancomycine
         3 |           3 |      3 |          0 |         1415 |           23 | 14          |       114 |      534 |          420 | N/A               |                420 | 2003-2009          |             |            |                                                                 |                       |                               |                |          |                   |                                                  | Cefazoline (Kefzol)
         4 |           4 |      4 |          0 |         3015 |           50 | 19          |       115 |      591 |          476 | N/A               |                476 | 2010-2016          |             |            | CABG alone, coronary artery bypass grafting                     | CABG en Klepchirurgie | Post-operative cardiovascular | NICE APACHE IV |        1 |                 0 |                                                  | Cefazoline (Kefzol)
         5 |           5 |      5 |          0 |         4107 |           69 | 31          |        50 |     2662 |         2612 | N/A               |               2612 | 2010-2016          |             |            | Restrictive lung disease (i.e. sarcoidosis, pulmonary fibrosis) | Pulmonaal             | Non-operative respiratory     | NICE APACHE IV |        0 |                 0 | Ceftriaxon (Rocephin); Ciprofloxacine (Ciproxin) |
         6 |           6 |      6 |          0 |         2891 |           48 | 45          |        78 |     1956 |         1878 | N/A               |               1878 | 2010-2016          |             |            | Shunts and revisions                                            | Overige               | Neurochirurgie                | NICE APACHE IV |        1 |                 0 |                                                  |
*/

/*
Extracts:
- patientid | integer | unique value to identify individual patients throughout their admission(s)
- admissionid | integer | unique value to identify the admission to the ICU. Patients may have multiple admissionid's during the same or another hospitalization, when they are discharge from the ICU or MC unit. Admissionid is used as an identifier in all other tables
- ventid | integer | unique value to identify an individual ventilation episode
- admittedat | integer | number of minutes since the first admission. For the first admission this is set to zero
- dischargedat | integer | time of discharge in number of minutes since the first admission
- lengthofstay | integer | length of stay of the admission in hours
- destination | string | department the patient has been discharged to or 'Overleden' if the patient died during the ICU/MCU admission
- ventstart | integer | start time (time of introduction) of the process, in minutes since the first ICU admission
- ventstop | integer | end point of interest, in minutes since the first ICU admission
- ventduration | integer | the duration of the time period of interest i.e. the difference between the start and end point of interest (N.B. period of interest can be stopped early by a period of non-invasive ventilation or a tracheostomy - at this point we stop tracking the patient)
- early termination | string | log if the ventilation period was stopped early because of a tracheostomy or period of NIV lasting longer than one hour
- actualventduration | integer | the actual duration of the process, i.e. the difference between start and stop times, in minutes
- admissionyeargroup | string | year of admission, categorised
- dateofdeath | integer | the date on which the patient died calculated as the minutes since the first admission, if applicable (non-null)
*/

-- creates a materialized view ventilationsettings which looks like this:
/*
 ventid | value | measuredat
--------+-------+------------
      0 | PC    |        342
      0 | PC    |        372
      0 | PC    |        432
      0 | PC    |        492
      0 | PC    |        552
      0 | PC    |        612

Extracts:
- ventid | integer | unique value to identify an individual ventilation episode
- value | string | the setting on the ventilator
- measuredat | integer | number of minutes since the person was admitted to the ICU (N.B. can be negative if the ventilation is started before admission, although this only applies to 6 ventilation episodes)
*/

drop view if exists labelinfo cascade;
drop view if exists tracheostomy_start_times cascade;
drop view if exists niv_times cascade;
drop view if exists vent_end_points cascade;
drop materialized view if exists externalquery cascade;
drop materialized view if exists ventilationsettings cascade;
drop materialized view if exists labels cascade;


-- select all the data we need from the admissions table
-- there are 23,106 admissions in the admissions table, and 20,109 distinct patients
-- this query extracts all of the ventilation episodes in the ICU (18,471 ventilation episodes from 15,950 admissions from 14,807 patients).
create view labelinfo as
  -- I convert millisecond times to minutes by dividing by 1000*60
  select a.patientid, a.admissionid, row_number() over(order by a.patientid, a.admissionid, pi.start) - 1 as ventid,
    cast(a.admittedat as float)/(1000*60) as admittedat, cast(a.dischargedat as float)/(1000*60) as dischargedat,
    a.lengthofstay, a.destination, cast(pi.start as float)/(1000*60) as ventstart,
    cast(pi.stop as float)/(1000*60) as ventstop, pi.duration as ventduration, a.admissionyeargroup,
    cast(a.dateofdeath as float)/(1000*60) as dateofdeath
    from admissions as a
    inner join processitems as pi on a.admissionid = pi.admissionid
    and pi.item = 'Beademen';  -- extract the ventilation episodes


-- capture all the ventids which have tracheostomy measurements associated with them, and the first time which this
-- appears. We will end the ventilation episode when the patient has the tracheostomy tube inserted, and mark this as a
-- specific 'end point' to our period of interest.
create view tracheostomy_start_times as
  select linf.ventid, linf.admissionid, cast(min(li.measuredat) as float)/(1000*60) as trachstart
    from listitems as li
    inner join labelinfo as linf on linf.admissionid = li.admissionid
    where cast(li.measuredat as float)/(1000*60) between linf.ventstart and linf.ventstop
      and li.item in ('Tracheacanule type', 'Tracheacanule maat', 'Tracheotomie methode')
    group by linf.ventid, linf.admissionid
    order by linf.ventid;


-- as a heuristic for which patients I should exclude on account of non-invasive ventilation, I take the first and the
-- last NIV recordings, with the idea to exclude ventids which have NIV recordings that span more than an hour during
-- their stay (note that this will also capture patients who have multiple NIV stints which may be less than an hour
-- but where the total span exceeds one hour in length).
create view niv_times as
  select linf.ventid, min(cast(li.measuredat as float)/(1000*60)) as firstniv,
    max(cast(li.measuredat as float)/(1000*60)) as lastniv,
    cast((max(li.measuredat) - min(li.measuredat)) as float)/(1000*60) as niv_timediff
    from listitems as li
    inner join labelinfo as linf on linf.admissionid = li.admissionid
    where cast(li.measuredat as float)/(1000*60) between linf.ventstart and linf.ventstop
      and li.item in ('Ventilatie Mode (Set)', 'Type beademing Evita 1', 'Type Beademing Evita 4',
      'Type Beademing Evita 4(2)', 'Ventilatie Mode (Set) (2)')
      and li.value like '%NIV%'  -- this applies to 153 ventids but 6042 NIV recordings in total
    group by linf.ventid
    -- only take note in niv_times if the niv_timediff is more than 60 minutes - this takes the number of ventids from
    -- 153 to 42. N.B. most of the patients that we are ignoring with this statement just have one NIV entry i.e. niv_timediff is 0.
    having cast((max(li.measuredat) - min(li.measuredat)) as float)/(1000*60) > 60
    order by linf.ventid;


-- define the ventilation end points for each ventid (some of which need early termination)
/*
The counts of each type of early termination in labelinfo are shown by:
select early_termination, count(*) from vent_end_points group by early_termination;
 early_termination | count
-------------------+-------
 > 21 days         |   407
 N/A               | 17125
 NIV               |    41
 Tracheostomy      |   898
 */
create view vent_end_points as
  select linf.ventid, tst.trachstart, niv.firstniv, ventstop,
    -- note that case statements pick the first option that is true, so the order of these statements matter
    case
      -- when the patient is ventilated NIV after 21 days and has no tracheostomy, cut the episode at 21 days:
      when tst.trachstart is null
        and niv.firstniv is not null
        and niv.firstniv - linf.ventstart > 21*24*60
        and linf.ventstop - linf.ventstart > 21*24*60
        then linf.ventstart + 21*24*60
      -- when the patient is ventilated NIV before 21 days and has no tracheostomy, use the NIV time to prematurely
      -- end the episode:
      when tst.trachstart is null
        and niv.firstniv is not null
        and niv.firstniv < linf.ventstop
        then niv.firstniv
      -- when the patient has a tracheostomy after 21 days, but doesn't have NIV, cut the episode at 21 days:
      when niv.firstniv is null
        and tst.trachstart is not null
        and tst.trachstart - linf.ventstart > 21*24*60
        and linf.ventstop - linf.ventstart > 21*24*60
        then linf.ventstart + 21*24*60
      -- when the patient has a tracheostomy before 21 days but doesn't have NIV, use the tracheostomy time to
      -- prematurely end the episode:
      when niv.firstniv is null
        and tst.trachstart is not null
        and tst.trachstart < linf.ventstop
        then tst.trachstart
      -- when the patient has both a tracheostomy and NIV ventilation, but both happen after 21 days, cut the episode
      -- at 21 days:
      when tst.trachstart is not null
        and niv.firstniv is not null
        and niv.firstniv - linf.ventstart > 21*24*60
        and tst.trachstart - linf.ventstart > 21*24*60
        and linf.ventstop - linf.ventstart > 21*24*60
        then linf.ventstart + 21*24*60
      -- when the patient has both NIV and a tracheostomy, but the NIV happens first and before 21 days, use the NIV
      -- time to prematurely end the episode:
      when niv.firstniv is not null
        and tst.trachstart is not null
        and niv.firstniv < tst.trachstart
        and niv.firstniv < linf.ventstop
        then niv.firstniv
      -- when the patient has both NIV and a tracheostomy, but the tracheostomy happens first and it is before 21 days,
      -- use the tracheostomy time to prematurely end the episode:
      when niv.firstniv is not null
        and tst.trachstart is not null
        and tst.trachstart < niv.firstniv
        and tst.trachstart < linf.ventstop
        then tst.trachstart
      -- when the episode is longer than 21 days, cut the episode at 21 days:
      when linf.ventstop - linf.ventstart > 21*24*60
        then linf.ventstart + 21*24*60
      -- otherwise just use the original end time (will be the case for most patients)
      else linf.ventstop
      end as corrected_ventstop,
    case
      when tst.trachstart is null
        and niv.firstniv is not null
        and niv.firstniv - linf.ventstart > 21*24*60
        and linf.ventstop - linf.ventstart > 21*24*60
        then '> 21 days'
      when tst.trachstart is null
        and niv.firstniv is not null
        and niv.firstniv < linf.ventstop
        then 'NIV'
      when niv.firstniv is null
        and tst.trachstart is not null
        and tst.trachstart - linf.ventstart > 21*24*60
        and linf.ventstop - linf.ventstart > 21*24*60
        then '> 21 days'
      when niv.firstniv is null
        and tst.trachstart is not null
        and tst.trachstart < linf.ventstop
        then 'Tracheostomy'
      when tst.trachstart is not null
        and niv.firstniv is not null
        and niv.firstniv - linf.ventstart > 21*24*60
        and tst.trachstart - linf.ventstart > 21*24*60
        and linf.ventstop - linf.ventstart > 21*24*60
        then '> 21 days'
      when niv.firstniv is not null
        and tst.trachstart is not null
        and niv.firstniv < tst.trachstart
        and niv.firstniv < linf.ventstop
        then 'NIV'
      when niv.firstniv is not null
        and tst.trachstart is not null
        and tst.trachstart < niv.firstniv
        and tst.trachstart < linf.ventstop
        then 'Tracheostomy'
      when linf.ventstop - linf.ventstart > 21*24*60
        then '> 21 days'
      else 'N/A'
      end as early_termination
    from labelinfo as linf
      left join tracheostomy_start_times as tst on tst.ventid = linf.ventid
      left join niv_times as niv on niv.ventid = linf.ventid
    order by linf.ventid;


-- extracts ventilation settings during the ventilation window for the cohort of interest
-- this obtains 15,344 distinct ventilation episodes from 13,536 admissionids from 12,615 patientids
-- in labelinfo, there are 18,471 distinct ventids from 15,950 admissionids from 14,807 patientids
create materialized view ventilationsettings as
  select linf.ventid, li.value, cast(li.measuredat as float)/(1000*60) as measuredat
    from listitems as li
    inner join labelinfo as linf on linf.admissionid = li.admissionid
    inner join vent_end_points as vep on vep.ventid = linf.ventid
    -- joining by admissionid rather than ventid because we want to know whether there was a tracheostomy during the admission (even if outside the ventilation episode)
    left join (select admissionid, min(trachstart) as trachstart from tracheostomy_start_times group by admissionid) as tst
      on tst.admissionid = linf.admissionid
    -- extract ventilation settings between the ventilation start and end time (or early end time if the patient has a
    -- tracheostomy or gets NIV for more than an hour)
    where cast(li.measuredat as float)/(1000*60) between linf.ventstart and vep.corrected_ventstop
      and li.item in ('Ventilatie Mode (Set)', 'Type beademing Evita 1', 'Type Beademing Evita 4',
      'Type Beademing Evita 4(2)', 'Ventilatie Mode (Set) (2)')
      -- we ignore 'Stand By' modes, as they only apply to a small number of ventids labelinfo, and they are usually one-offs (see below in preprocessing investigation)
      and li.value != 'Stand By'
      -- we also discount the NIV modes here, but they are taken into account for early termination of the ventilation episode (see above)
      and li.value not like '%NIV%'
      -- so far the query obtains 18,104 distinct ventids from 15,778 admissionids from 14,701 patientids
      and vep.corrected_ventstop - linf.ventstart > 4*60  -- the requirement for at least 4 hours of ventilation reduces this to 15,344 distinct ventilation episodes from 13,536 admissionids from 12,615 patientids
      -- make sure the patient has not already got a tracheostomy at the start of the episode
      and (tst.trachstart > linf.ventstart or tst.trachstart is null)  -- this brings the number of ventids from 15,344 to 14,836
    order by linf.ventid, li.measuredat;


-- this next query is taken from the amsterdam repository https://github.com/AmsterdamUMC/AmsterdamUMCdb/blob/9c66f2bb801266b00cf3b4aaa9d0afc580cd7253/concepts/diagnosis/reason_for_admission.ipynb
-- this is just to get some extra info about the patients which we might be able to use posthoc
-- note that we include these things as labels because we can't be sure when we get this information (some is restricted to the first day only)
create materialized view externalquery as
WITH diagnosis_groups AS (
    SELECT admissionid,
        item,
        CASE
            WHEN itemid IN (
                18669, --NICE APACHEII diagnosen
                18671 --NICE APACHEIV diagnosen
            )
            THEN split_part(value, ' - ', 1)
            -- 'e.g. 'Non-operative cardiovascular - Anaphylaxis' -> Non-operative cardiovascular
            ELSE value
        END as diagnosis_group,
        valueid as diagnosis_group_id,
        ROW_NUMBER() OVER(PARTITION BY admissionid
        ORDER BY
            CASE --prefer NICE > APACHE IV > II > D
                WHEN itemid = 18671 THEN 6 --NICE APACHEIV diagnosen
                WHEN itemid = 18669 THEN 5 --NICE APACHEII diagnosen
                WHEN itemid BETWEEN 16998 AND 17017 THEN 4 --APACHE IV diagnosis
                WHEN itemid BETWEEN 18589 AND 18602 THEN 3 --APACHE II diagnosis
                WHEN itemid BETWEEN 13116 AND 13145 THEN 2 --D diagnosis ICU
                WHEN itemid BETWEEN 16642 AND 16673 THEN 1 --DMC diagnosis Medium Care
            END DESC,
        measuredat DESC) AS rownum
    FROM listitems
    WHERE itemid IN (
        --MAIN GROUP - LEVEL 0
        13110, --D_Hoofdgroep
        16651, --DMC_Hoofdgroep, Medium Care

        18588, --Apache II Hoofdgroep
        16997, --APACHE IV Groepen

        18669, --NICE APACHEII diagnosen
        18671 --NICE APACHEIV diagnosen
    )
),diagnosis_subgroups AS (
    SELECT admissionid,
        item,
        value as diagnosis_subgroup,
        valueid as diagnosis_subgroup_id,
        ROW_NUMBER() OVER(PARTITION BY admissionid
        ORDER BY measuredat DESC) AS rownum
    FROM listitems
    WHERE itemid IN (
        --SUB GROUP - LEVEL 1
        13111, --D_Subgroep_Thoraxchirurgie
        16669, --DMC_Subgroep_Thoraxchirurgie
        13112, --D_Subgroep_Algemene chirurgie
        16665, --DMC_Subgroep_Algemene chirurgie
        13113, --D_Subgroep_Neurochirurgie
        16667, --DMC_Subgroep_Neurochirurgie
        13114, --D_Subgroep_Neurologie
        16668, --DMC_Subgroep_Neurologie
        13115, --D_Subgroep_Interne geneeskunde
        16666 --DMC_Subgroep_Interne geneeskunde
    )
), diagnoses AS (
    SELECT admissionid,
        item,
        CASE
            WHEN itemid IN (
                18669, --NICE APACHEII diagnosen
                18671 --NICE APACHEIV diagnosen
            )
            THEN split_part(value, ' - ', 2)
            -- 'e.g. 'Non-operative cardiovascular - Anaphylaxis' -> Anaphylaxis
            ELSE value
        END as diagnosis,
        CASE
            WHEN itemid IN (
                --SURGICAL
                13116, --D_Thoraxchirurgie_CABG en Klepchirurgie
                16671, --DMC_Thoraxchirurgie_CABG en Klepchirurgie
                13117, --D_Thoraxchirurgie_Cardio anders
                16672, --DMC_Thoraxchirurgie_Cardio anders
                13118, --D_Thoraxchirurgie_Aorta chirurgie
                16670, --DMC_Thoraxchirurgie_Aorta chirurgie
                13119, --D_Thoraxchirurgie_Pulmonale chirurgie
                16673, --DMC_Thoraxchirurgie_Pulmonale chirurgie

                --Not surgical: 13141, --D_Algemene chirurgie_Algemeen
                --Not surgical: 16642, --DMC_Algemene chirurgie_Algemeen
                13121, --D_Algemene chirurgie_Buikchirurgie
                16643, --DMC_Algemene chirurgie_Buikchirurgie
                13123, --D_Algemene chirurgie_Endocrinologische chirurgie
                16644, --DMC_Algemene chirurgie_Endocrinologische chirurgie
                13145, --D_Algemene chirurgie_KNO/Overige
                16645, --DMC_Algemene chirurgie_KNO/Overige
                13125, --D_Algemene chirurgie_Orthopedische chirurgie
                16646, --DMC_Algemene chirurgie_Orthopedische chirurgie
                13122, --D_Algemene chirurgie_Transplantatie chirurgie
                16647, --DMC_Algemene chirurgie_Transplantatie chirurgie
                13124, --D_Algemene chirurgie_Trauma
                16648, --DMC_Algemene chirurgie_Trauma
                13126, --D_Algemene chirurgie_Urogenitaal
                16649, --DMC_Algemene chirurgie_Urogenitaal
                13120, --D_Algemene chirurgie_Vaatchirurgie
                16650, --DMC_Algemene chirurgie_Vaatchirurgie

                13128, --D_Neurochirurgie _Vasculair chirurgisch
                16661, --DMC_Neurochirurgie _Vasculair chirurgisch
                13129, --D_Neurochirurgie _Tumor chirurgie
                16660, --DMC_Neurochirurgie _Tumor chirurgie
                13130, --D_Neurochirurgie_Overige
                16662, --DMC_Neurochirurgie_Overige

                18596, --Apache II Operatief  Gastr-intenstinaal
                18597, --Apache II Operatief Cardiovasculair
                18598, --Apache II Operatief Hematologisch
                18599, --Apache II Operatief Metabolisme
                18600, --Apache II Operatief Neurologisch
                18601, --Apache II Operatief Renaal
                18602, --Apache II Operatief Respiratoir

                17008, --APACHEIV Post-operative cardiovascular
                17009, --APACHEIV Post-operative gastro-intestinal
                17010, --APACHEIV Post-operative genitourinary
                17011, --APACHEIV Post-operative hematology
                17012, --APACHEIV Post-operative metabolic
                17013, --APACHEIV Post-operative musculoskeletal /skin
                17014, --APACHEIV Post-operative neurologic
                17015, --APACHEIV Post-operative respiratory
                17016, --APACHEIV Post-operative transplant
                17017 --APACHEIV Post-operative trauma

            ) THEN 1
            WHEN itemid = 18669 AND valueid BETWEEN 1 AND 26 THEN 1 --NICE APACHEII diagnosen
            WHEN itemid = 18671 AND valueid BETWEEN 222 AND 452 THEN 1 --NICE APACHEIV diagnosen
            ELSE 0
        END AS surgical,
        valueid as diagnosis_id,
        CASE
                WHEN itemid = 18671 THEN 'NICE APACHE IV'
                WHEN itemid = 18669 THEN 'NICE APACHE II'
                WHEN itemid BETWEEN 16998 AND 17017 THEN 'APACHE IV'
                WHEN itemid BETWEEN 18589 AND 18602 THEN 'APACHE II'
                WHEN itemid BETWEEN 13116 AND 13145 THEN 'Legacy ICU'
                WHEN itemid BETWEEN 16642 AND 16673 THEN 'Legacy MCU'
        END AS diagnosis_type,
        ROW_NUMBER() OVER(PARTITION BY admissionid
        ORDER BY
            CASE --prefer NICE > APACHE IV > II > D
                WHEN itemid = 18671 THEN 6 --NICE APACHEIV diagnosen
                WHEN itemid = 18669 THEN 5 --NICE APACHEII diagnosen
                WHEN itemid BETWEEN 16998 AND 17017 THEN 4 --APACHE IV diagnosis
                WHEN itemid BETWEEN 18589 AND 18602 THEN 3 --APACHE II diagnosis
                WHEN itemid BETWEEN 13116 AND 13145 THEN 2 --D diagnosis ICU
                WHEN itemid BETWEEN 16642 AND 16673 THEN 1 --DMC diagnosis Medium Care
            END DESC,
            measuredat DESC) AS rownum
    FROM listitems
    WHERE itemid IN (
        -- Diagnosis - LEVEL 2
        --SURGICAL
        13116, --D_Thoraxchirurgie_CABG en Klepchirurgie
        16671, --DMC_Thoraxchirurgie_CABG en Klepchirurgie
        13117, --D_Thoraxchirurgie_Cardio anders
        16672, --DMC_Thoraxchirurgie_Cardio anders
        13118, --D_Thoraxchirurgie_Aorta chirurgie
        16670, --DMC_Thoraxchirurgie_Aorta chirurgie
        13119, --D_Thoraxchirurgie_Pulmonale chirurgie
        16673, --DMC_Thoraxchirurgie_Pulmonale chirurgie

        13141, --D_Algemene chirurgie_Algemeen
        16642, --DMC_Algemene chirurgie_Algemeen
        13121, --D_Algemene chirurgie_Buikchirurgie
        16643, --DMC_Algemene chirurgie_Buikchirurgie
        13123, --D_Algemene chirurgie_Endocrinologische chirurgie
        16644, --DMC_Algemene chirurgie_Endocrinologische chirurgie
        13145, --D_Algemene chirurgie_KNO/Overige
        16645, --DMC_Algemene chirurgie_KNO/Overige
        13125, --D_Algemene chirurgie_Orthopedische chirurgie
        16646, --DMC_Algemene chirurgie_Orthopedische chirurgie
        13122, --D_Algemene chirurgie_Transplantatie chirurgie
        16647, --DMC_Algemene chirurgie_Transplantatie chirurgie
        13124, --D_Algemene chirurgie_Trauma
        16648, --DMC_Algemene chirurgie_Trauma
        13126, --D_Algemene chirurgie_Urogenitaal
        16649, --DMC_Algemene chirurgie_Urogenitaal
        13120, --D_Algemene chirurgie_Vaatchirurgie
        16650, --DMC_Algemene chirurgie_Vaatchirurgie

        13128, --D_Neurochirurgie _Vasculair chirurgisch
        16661, --DMC_Neurochirurgie _Vasculair chirurgisch
        13129, --D_Neurochirurgie _Tumor chirurgie
        16660, --DMC_Neurochirurgie _Tumor chirurgie
        13130, --D_Neurochirurgie_Overige
        16662, --DMC_Neurochirurgie_Overige

        18596, --Apache II Operatief  Gastr-intenstinaal
        18597, --Apache II Operatief Cardiovasculair
        18598, --Apache II Operatief Hematologisch
        18599, --Apache II Operatief Metabolisme
        18600, --Apache II Operatief Neurologisch
        18601, --Apache II Operatief Renaal
        18602, --Apache II Operatief Respiratoir

        17008, --APACHEIV Post-operative cardiovascular
        17009, --APACHEIV Post-operative gastro-intestinal
        17010, --APACHEIV Post-operative genitourinary
        17011, --APACHEIV Post-operative hematology
        17012, --APACHEIV Post-operative metabolic
        17013, --APACHEIV Post-operative musculoskeletal /skin
        17014, --APACHEIV Post-operative neurologic
        17015, --APACHEIV Post-operative respiratory
        17016, --APACHEIV Post-operative transplant
        17017, --APACHEIV Post-operative trauma

        --MEDICAL
        13133, --D_Interne Geneeskunde_Cardiovasculair
        16653, --DMC_Interne Geneeskunde_Cardiovasculair
        13134, --D_Interne Geneeskunde_Pulmonaal
        16658, --DMC_Interne Geneeskunde_Pulmonaal
        13135, --D_Interne Geneeskunde_Abdominaal
        16652, --DMC_Interne Geneeskunde_Abdominaal
        13136, --D_Interne Geneeskunde_Infectieziekten
        16655, --DMC_Interne Geneeskunde_Infectieziekten
        13137, --D_Interne Geneeskunde_Metabool
        16656, --DMC_Interne Geneeskunde_Metabool
        13138, --D_Interne Geneeskunde_Renaal
        16659, --DMC_Interne Geneeskunde_Renaal
        13139, --D_Interne Geneeskunde_Hematologisch
        16654, --DMC_Interne Geneeskunde_Hematologisch
        13140, --D_Interne Geneeskunde_Overige
        16657, --DMC_Interne Geneeskunde_Overige

        13131, --D_Neurologie_Vasculair neurologisch
        16664, --DMC_Neurologie_Vasculair neurologisch
        13132, --D_Neurologie_Overige
        16663, --DMC_Neurologie_Overige
        13127, --D_KNO/Overige

        18589, --Apache II Non-Operatief Cardiovasculair
        18590, --Apache II Non-Operatief Gastro-intestinaal
        18591, --Apache II Non-Operatief Hematologisch
        18592, --Apache II Non-Operatief Metabolisme
        18593, --Apache II Non-Operatief Neurologisch
        18594, --Apache II Non-Operatief Renaal
        18595, --Apache II Non-Operatief Respiratoir

        16998, --APACHE IV Non-operative cardiovascular
        16999, --APACHE IV Non-operative Gastro-intestinal
        17000, --APACHE IV Non-operative genitourinary
        17001, --APACHEIV  Non-operative haematological
        17002, --APACHEIV  Non-operative metabolic
        17003, --APACHEIV Non-operative musculo-skeletal
        17004, --APACHEIV Non-operative neurologic
        17005, --APACHEIV Non-operative respiratory
        17006, --APACHEIV Non-operative transplant
        17007, --APACHEIV Non-operative trauma

        --NICE: surgical/medical combined in same parameter
        18669, --NICE APACHEII diagnosen
        18671 --NICE APACHEIV diagnosen
    )
), sepsis AS (
    SELECT
        admissionid,
        CASE valueid
            WHEN 1 THEN 1 --'Ja'
            WHEN 2 THEN 0 --'Nee'
        END as sepsis_at_admission,
        ROW_NUMBER() OVER(
            PARTITION BY
                admissionid
            ORDER BY
                measuredat DESC) AS rownum
    FROM listitems
    WHERE
        itemid = 15808
), sepsis_antibiotics AS ( --non prophylactic antibiotics
    SELECT
        admissionid,
        CASE
            WHEN COUNT(*) > 0 THEN 1
            ELSE 0
        END AS sepsis_antibiotics_bool,
        STRING_AGG(DISTINCT item, '; ') AS sepsis_antibiotics_given
    FROM drugitems
    WHERE
        itemid IN (
            6834, --Amikacine (Amukin)
            6847, --Amoxicilline (Clamoxyl/Flemoxin)
            6871, --Benzylpenicilline (Penicilline)
            6917, --Ceftazidim (Fortum)
            --6919, --Cefotaxim (Claforan) -> prophylaxis
            6948, --Ciprofloxacine (Ciproxin)
            6953, --Rifampicine (Rifadin)
            6958, --Clindamycine (Dalacin)
            7044, --Tobramycine (Obracin)
            --7064, --Vancomycine -> prophylaxis for valve surgery
            7123, --Imipenem (Tienam)
            7185, --Doxycycline (Vibramycine)
            --7187, --Metronidazol (Flagyl) -> often used for GI surgical prophylaxis
            --7208, --Erythromycine (Erythrocine) -> often used for gastroparesis
            7227, --Flucloxacilline (Stafoxil/Floxapen)
            7231, --Fluconazol (Diflucan)
            7232, --Ganciclovir (Cymevene)
            7233, --Flucytosine (Ancotil)
            7235, --Gentamicine (Garamycin)
            7243, --Foscarnet trinatrium (Foscavir)
            7450, --Amfotericine B (Fungizone)
            --7504, --X nader te bepalen --non-stock medication
            8127, --Meropenem (Meronem)
            8229, --Myambutol (ethambutol)
            8374, --Kinine dihydrocloride
            --8375, --Immunoglobuline (Nanogam) -> not anbiotic
            --8394, --Co-Trimoxazol (Bactrimel) -> often prophylactic (unless high dose)
            8547, --Voriconazol(VFEND)
            --9029, --Amoxicilline/Clavulaanzuur (Augmentin) -> often used for ENT surgical prophylaxis
            9030, --Aztreonam (Azactam)
            9047, --Chlooramfenicol
            --9075, --Fusidinezuur (Fucidin) -> prophylaxis
            9128, --Piperacilline (Pipcil)
            9133, --Ceftriaxon (Rocephin)
            --9151, --Cefuroxim (Zinacef) -> often used for GI/transplant surgical prophylaxis
            --9152, --Cefazoline (Kefzol) -> prophylaxis for cardiac surgery
            9458, --Caspofungine
            9542, --Itraconazol (Trisporal)
            --9602, --Tetanusimmunoglobuline -> prophylaxis/not antibiotic
            12398, --Levofloxacine (Tavanic)
            12772, --Amfotericine B lipidencomplex  (Abelcet)
            15739, --Ecalta (Anidulafungine)
            16367, --Research Anidulafungin/placebo
            16368, --Research Caspofungin/placebo
            18675, --Amfotericine B in liposomen (Ambisome )
            19137, --Linezolid (Zyvoxid)
            19764, --Tigecycline (Tygacil)
            19773, --Daptomycine (Cubicin)
            20175 --Colistine
        )
        AND start < 24*60*60*1000 --within 24 hours (to correct for antibiotics administered before ICU)
    GROUP BY admissionid
), other_antibiotics AS ( --'prophylactic' antibiotics that may be used for sepsis
    SELECT
        admissionid,
        CASE
            WHEN COUNT(*) > 0 THEN 1
            ELSE 0
        END AS other_antibiotics_bool,
        STRING_AGG(DISTINCT item, '; ') AS other_antibiotics_given
    FROM drugitems
    WHERE
        itemid IN (
            7064, --Vancomycine -> prophylaxis for valve surgery
            7187, --Metronidazol (Flagyl) -> often used for GI surgical prophylaxis
            8394, --Co-Trimoxazol (Bactrimel) -> often prophylactic (unless high dose)
            9029, --Amoxicilline/Clavulaanzuur (Augmentin) -> often used for ENT surgical prophylaxis
            9151, --Cefuroxim (Zinacef) -> often used for GI surgical prophylaxis
            9152 --Cefazoline (Kefzol) -> prophylaxis
        )
        AND start < 24*60*60*1000 --within 24 hours (to correct for antibiotics administered before ICU)
    GROUP BY admissionid
), cultures AS (
    SELECT
        admissionid,
        CASE
            WHEN COUNT(*) > 0 THEN 1
            ELSE 0
        END AS sepsis_cultures_bool,
        STRING_AGG(DISTINCT item, '; ') AS sepsis_cultures_drawn
    FROM procedureorderitems
    WHERE
        itemid IN (
        --8097, --Sputumkweek afnemen -> often used routinely
        --8418, --Urinekweek afnemen
        --8588, --MRSA kweken afnemen
        9189, --Bloedkweken afnemen
        9190, --Cathetertipkweek afnemen
        --9191, --Drainvochtkweek afnemen
        --9192, --Faeceskweek afnemen -> Clostridium
        --9193, --X-Kweek nader te bepalen
        --9194, --Liquorkweek afnemen
        --9195, --Neuskweek afnemen
        --9197, --Perineumkweek afnemen -> often used routinely
        --9198, --Rectumkweek afnemen -> often used routinely
        9200, --Wondkweek afnemen
        9202, --Ascitesvochtkweek afnemen
        --9203, --Keelkweek afnemen -> often used routinely
        --9204, --SDD-kweken afnemen -> often used routinely
        9205 --Legionella sneltest (urine)
        --1302, --SDD Inventarisatiekweken afnemen -> often used routinely
        --19663, --Research Neuskweek COUrSe
        --19664, --Research Sputumkweek COUrSe
        )
        AND registeredat < 6*60*60*1000 --within 6 hours
    GROUP BY admissionid
)
SELECT
    admissions.*
    , diagnosis_type
    , diagnosis, diagnosis_id
    , diagnosis_subgroup
    , diagnosis_subgroup_id
    , diagnosis_group
    , diagnosis_group_id
    , surgical
    , sepsis_at_admission
    , sepsis_antibiotics_bool
    , sepsis_antibiotics_given
    , other_antibiotics_bool
    , other_antibiotics_given
    , sepsis_cultures_bool
    , sepsis_cultures_drawn
FROM admissions
LEFT JOIN diagnoses on admissions.admissionid = diagnoses.admissionid
LEFT JOIN diagnosis_subgroups on admissions.admissionid = diagnosis_subgroups.admissionid
LEFT JOIN diagnosis_groups on admissions.admissionid = diagnosis_groups.admissionid
LEFT JOIN sepsis on admissions.admissionid = sepsis.admissionid
LEFT JOIN sepsis_antibiotics on admissions.admissionid = sepsis_antibiotics.admissionid
LEFT JOIN other_antibiotics on admissions.admissionid = other_antibiotics.admissionid
LEFT JOIN cultures on admissions.admissionid = cultures.admissionid
WHERE --only last updated record
    (diagnoses.rownum = 1 OR diagnoses.rownum IS NULL) AND
    (diagnosis_subgroups.rownum = 1 OR diagnosis_subgroups.rownum IS NULL) AND
    (diagnosis_groups.rownum = 1 OR diagnosis_groups.rownum IS NULL) AND
    (sepsis.rownum = 1 OR sepsis.rownum IS NULL);


-- final labels table for cohort with ventilation settings, including early termination
-- 14,836 distinct ventilation episodes from 13,502 admissionids from 12,597 patientids
/*
The counts of each type of early termination in labels are shown by:
select early_termination, count(*) from labels group by early_termination;
 early_termination | count
-------------------+-------
 > 21 days         |   399
 Tracheostomy      |   648
 N/A               | 13783
 NIV               |     6
 */
create materialized view labels as
  select linf.patientid, linf.admissionid, linf.ventid, linf.admittedat, linf.dischargedat, linf.lengthofstay,
    linf.destination, linf.ventstart, vep.corrected_ventstop as ventstop, vep.corrected_ventstop - linf.ventstart
    as ventduration, vep.early_termination, linf.ventduration as actualventduration, linf.admissionyeargroup,
    linf.dateofdeath, tst.trachstart, eq.diagnosis, eq.diagnosis_subgroup as diagnosissubgroup, eq.diagnosis_group as
    diagnosisgroup, eq.diagnosis_type as diagnosistype, eq.surgical, eq.sepsis_at_admission as sepsisatadmission,
    eq.sepsis_antibiotics_given as sepsisantibiotics, eq.other_antibiotics_given as otherantibiotics
    from labelinfo as linf
    inner join vent_end_points as vep on vep.ventid = linf.ventid
    -- joining by admissionid rather than ventid because we want to know whether there was a tracheostomy during the admission (even if outside the ventilation episode)
    left join (select admissionid, min(trachstart) as trachstart from tracheostomy_start_times group by admissionid) as tst
      on tst.admissionid = linf.admissionid
    left join externalquery as eq on eq.admissionid = linf.ventid
    where linf.ventid in (select distinct ventid from ventilationsettings)
    order by linf.patientid, linf.admissionid, linf.ventid;


/* PREPROCESSING INVESTIGATION

-- N.B some of the numbers may be a little off because the cohort selection criteria were tweaked and the cohort got
-- smaller (therefore if any numbers are wrong they will be an overestimate).

VENTILATOR SETTINGS

-- First I ran a tester query to see what kinds of ventilator settings we could see for patients, using guidance from
https://github.com/AmsterdamUMC/AmsterdamUMCdb/blob/master/concepts/lifesupport/mechanical_ventilation.ipynb on what
to look for:

select item, count(distinct admissionid) from listitems
  group by item
  having lower(item) like '%beademing%'
   or lower(item) like '%ventilatie%'
   or lower(item) like '%toediening%'
   or lower(item) like '%bipap%';

             item              | count
-------------------------------+-------
 BPS - Acceptatie beademing    |  4341
 Beademingstoestel             |    76
 Beademingstoestel(2)          |     1
 DIM medicatie Toediening      |   281
 Mode (Bipap Vision)           |    23
 SEPSIS_PROTECTIEVE_VENTILATIE |   437
 Toedieningsweg                | 20646
 Type Beademing Evita 4        |   368
 Type Beademing Evita 4(2)     |     1
 Type beademing Evita 1        |   614
 Ventilatie Mode (Set)         | 15639
 Ventilatie Mode (Set) (2)     |    10

-- Checking values of Toedieningsweg, Acceptatie beademing, SEPSIS_PROTECTIEVE_VENTILATIE, Type Beademing Evita 4,
Type beademing Evita 1 and Ventilatie Mode (Set) - the others are too low in numbers to worry about:

select value, count(distinct admissionid)
  from listitems
  where item = 'Toedieningsweg'
  group by value;

-- Beademingstoestel:

    value     | count
--------------+-------
 Bipap Vision |    16
 Evita 1      |    38
 Evita 4      |    32

-- Toedieningsweg:

         value          | count
------------------------+-------
 Ambu                   |    18
 B.Lucht                |  1883
 CPAP                   |   396
 DL-tube                |     1
 Diep Nasaal            |   933
 Guedel                 |    30
 Kapje                  |  4913
 Kinnebak               |   622
 Kunstneus              |  2967
 Nasaal                 |   572
 Nebulizer              |    41
 Non-Rebreathing masker |  3342
 O2-bril                | 18459
 Spreekcanule           |   521
 Spreekklepje           |   725
 Trach.stoma            |    98
 Waterset               |    49

-- BPS - Acceptatie beademing:

                         value                          | count
--------------------------------------------------------+-------
 Ademt actief tegen beademing in                        |   206
 Af en toe hoesten maar geen reactie tegen beademing in |  1155
 Toestaan van bewegingen                                |  4269
 Vecht tegen beademing, beademing niet mogelijk         |    35

-- SEPSIS_PROTECTIEVE_VENTILATIE:

 value  | count
--------+-------
 Ja     |   290
 N.V.T. |   128
 Nee    |    22

-- Type Beademing Evita 4:

     value      | count
----------------+-------
 ASB            |    42
 BIPAP          |    84
 BIPAP-SIMV/ASB |     1
 BIPAP/ASB      |   134
 CPAP           |    51
 CPAP/ASB       |   330
 CPPV           |    32
 CPPV/ASSIST    |    20
 IPPV           |    41
 IPPV/ASSIST    |   287
 MMV            |     4
 MMV/ASB        |    25
 SIMV/ASB       |     2

-- Type Beademing Evita 4(2)

   value   | count
-----------+-------
 BIPAP/ASB |     1

-- Type beademing Evita 1:

       value        | count
--------------------+-------
 ASB                |   112
 BIPAP              |    28
 CPAP               |   122
 CPAP_ASB           |   536
 CPPV               |    69
 CPPV_Assist        |   561
 IPPV               |    54
 IPPV_Assist        |    33
 MMV                |     4
 MMV_ASB            |     2
 Pressure Controled |    84
 SIMV               |     3
 SIMV_ASB           |    10

-- Ventilatie Mode (Set):

     value      | count
----------------+-------
 Bi Vente       |    57
 NAVA           |   250
 PC             | 12948
 PC (No trig)   | 10357
 PC in NIV      |   170
 PRVC           |     5
 PRVC (No trig) |     7
 PRVC (trig)    |     1
 PS/CPAP        | 12993
 PS/CPAP (trig) |  9633
 PS/CPAP in NIV |  1379
 SIMV(PC)+PS    |   580
 SIMV(VC)+PS    |   138
 Stand By       |   123
 VC             |  6821
 VC (No trig)   |    11
 VC (trig)      |     3
 VS             |     2

-- Ventilatie Mode (Set) (2):

    value    | count
-------------+-------
 PC          |     2
 PS/CPAP     |     2
 SIMV(PC)+PS |     2
 SIMV(VC)+PS |     2
 Stand By    |     1
 VC          |     1

INVESTIGATION INTO STAND BY AND NIV MODES:

-- First we broadly have a look at how these modes tend to appear

select linf.ventid, li.value, cast(li.measuredat as float)/(1000*60) as measuredat
  from listitems as li
  inner join labelinfo as linf on linf.admissionid = li.admissionid
  where cast(li.measuredat as float)/(1000*60) between linf.ventstart and linf.ventstop
    and li.item in ('Ventilatie Mode (Set)', 'Type beademing Evita 1', 'Type Beademing Evita 4',
      'Type Beademing Evita 4(2)', 'Ventilatie Mode (Set) (2)')
    and (li.value = 'Stand By'
      or li.value like '%NIV%')
    order by linf.ventid, li.measuredat limit 100;

 ventid |     value      | measuredat
--------+----------------+------------
     17 | Stand By       |       8449
    155 | PS/CPAP in NIV |       2717
    204 | Stand By       |      30182
    207 | Stand By       |     104162
    216 | Stand By       |      20221
    223 | Stand By       |      24381
    223 | Stand By       |      24409
    389 | PS/CPAP in NIV |       1877
    605 | Stand By       |      11137
    619 | PS/CPAP in NIV |      14065
    621 | PS/CPAP in NIV |      28264
    624 | PS/CPAP in NIV |      34057
    624 | PS/CPAP in NIV |      34058
    624 | PS/CPAP in NIV |      34059
    624 | PS/CPAP in NIV |      34060
    624 | PS/CPAP in NIV |      34061
    624 | PS/CPAP in NIV |      34062
    624 | PS/CPAP in NIV |      34063
    624 | PS/CPAP in NIV |      34064
    624 | PS/CPAP in NIV |      34065
    624 | PS/CPAP in NIV |      34066
    624 | PS/CPAP in NIV |      34067
    624 | PS/CPAP in NIV |      34068
    624 | PS/CPAP in NIV |      34069
    624 | PS/CPAP in NIV |      34070
    624 | PS/CPAP in NIV |      34071
    624 | PS/CPAP in NIV |      34072
    624 | PS/CPAP in NIV |      34073
    624 | PS/CPAP in NIV |      34074
    624 | PS/CPAP in NIV |      34075
    624 | PS/CPAP in NIV |      34076
    624 | PS/CPAP in NIV |      34077
    624 | PS/CPAP in NIV |      34078
    624 | PS/CPAP in NIV |      34079
    624 | PS/CPAP in NIV |      34080
    624 | PS/CPAP in NIV |      34081
    624 | PS/CPAP in NIV |      34082
    624 | PS/CPAP in NIV |      34083
    624 | PS/CPAP in NIV |      34084
    625 | PS/CPAP in NIV |      35464
    627 | PS/CPAP in NIV |      41280
    627 | PS/CPAP in NIV |      41281
    627 | PS/CPAP in NIV |      41282
    627 | PS/CPAP in NIV |      41283
    627 | PS/CPAP in NIV |      41284
    768 | Stand By       |      28476
    822 | Stand By       |      20433
    853 | PS/CPAP in NIV |      23451
    853 | PS/CPAP in NIV |      23452
    853 | PS/CPAP in NIV |      23453
    853 | PS/CPAP in NIV |      23454
    853 | PS/CPAP in NIV |      23455
    853 | PS/CPAP in NIV |      23456
    960 | PS/CPAP in NIV |     499162
    960 | PS/CPAP in NIV |     499163
    960 | PS/CPAP in NIV |     499164
    986 | PS/CPAP in NIV |       1176
   1179 | PS/CPAP in NIV |       2513
   1179 | PS/CPAP in NIV |       2517
   1179 | PS/CPAP in NIV |       2577
   1179 | PS/CPAP in NIV |       2637
   1179 | PS/CPAP in NIV |       2697
   1179 | PS/CPAP in NIV |       2757
   1179 | PS/CPAP in NIV |       2817
   1179 | PS/CPAP in NIV |       2877
   1179 | PS/CPAP in NIV |       2937
   1179 | PS/CPAP in NIV |       2997
   1179 | PS/CPAP in NIV |       3057
   1179 | PS/CPAP in NIV |       3117
   1179 | PS/CPAP in NIV |       3177
   1179 | PS/CPAP in NIV |       3237
   1179 | PS/CPAP in NIV |       3297
   1179 | PS/CPAP in NIV |       3357
   1179 | PS/CPAP in NIV |       3417
   1179 | PS/CPAP in NIV |       3477
   1179 | PS/CPAP in NIV |       3537
   1179 | PS/CPAP in NIV |       3597
   1179 | PS/CPAP in NIV |       3657
   1179 | PS/CPAP in NIV |       3717
   1179 | PS/CPAP in NIV |       3777
   1179 | PS/CPAP in NIV |       3837
   1179 | PS/CPAP in NIV |       3975
   1179 | PS/CPAP in NIV |       4017
   1179 | PS/CPAP in NIV |       4077
   1179 | PS/CPAP in NIV |       4137
   1179 | PS/CPAP in NIV |       4197
   1179 | PS/CPAP in NIV |       4257
   1179 | PS/CPAP in NIV |       4317
   1179 | PS/CPAP in NIV |       4377
   1179 | PS/CPAP in NIV |       4437
   1179 | PS/CPAP in NIV |       4497
   1347 | PS/CPAP in NIV |       2665
   1382 | Stand By       |      31735
   1457 | Stand By       |      31363
   1615 | PS/CPAP in NIV |        530
   1734 | Stand By       |     129333
   1856 | PS/CPAP in NIV |        601
   1856 | PS/CPAP in NIV |        602
   1856 | PS/CPAP in NIV |        603
   1856 | PS/CPAP in NIV |        604

-- Firstly we look at 'Stand By' mode:
-- How many ventids does this apply to?

select count(distinct linf.ventid)
  from listitems as li
  inner join labelinfo as linf on linf.admissionid = li.admissionid
  where cast(li.measuredat as float)/(1000*60) between linf.ventstart and linf.ventstop
    and li.item in ('Ventilatie Mode (Set)', 'Type beademing Evita 1', 'Type Beademing Evita 4',
      'Type Beademing Evita 4(2)', 'Ventilatie Mode (Set) (2)')
    and li.value = 'Stand By';

 count
-------
   106

-- How many actual 'Stand By' entries do we have?

select count(*)
  from listitems as li
  inner join labelinfo as linf on linf.admissionid = li.admissionid
  where cast(li.measuredat as float)/(1000*60) between linf.ventstart and linf.ventstop
    and li.item in ('Ventilatie Mode (Set)', 'Type beademing Evita 1', 'Type Beademing Evita 4',
      'Type Beademing Evita 4(2)', 'Ventilatie Mode (Set) (2)')
    and li.value = 'Stand By';

 count
-------
   161

-- Since this isn't really enough instances for an ML model to pick up on, I'm happy to simply exclude these instances
-- as they don't represent a significant amount of time for which the ventilators are in this setting and it only
-- applies to a handful of patients in any case. The entries look to be consistent with simply moving the patient.

-- Now on to 'NIV' modes:
-- How many ventids does this apply to?

 count
-------
   153

-- How many actual 'NIV' entries do we have?

 count
-------
  6042

-- This is trickier because this is a longer period of time. I think it would be sensible to ignore periods of NIV
-- ventilation that are shorter than an hour, but if longer than we take it as an end point to the ventilation period
-- (a bit like how we treat tracheostomy as an end point).

TRACHEOSTOMY PATIENTS

-- Check there is nothing in processitems first:

select item, count(distinct admissionid) from processitems
  group by item
  having lower(item) like '%tracheacanule%'
  or lower(item) like '%tracheotomie%';

                              item                              | count
----------------------------------------------------------------+-------
 Xx-Niet meer gebruiken Miditracheotomie(= Proces Tracheostoma) |     1

--Next look in listitems

select item, count(distinct admissionid)
from listitems
where lower(item) like '%tracheacanule%'
or lower(item) like '%tracheotomie%'
group by item;

                   item                   | count
------------------------------------------+-------
 Tracheacanule maat                       |   985
 Tracheacanule type                       |   859
 Tracheotomie Late complicaties           |    12
 Tracheotomie Perioperatieve complicaties |    32
 Tracheotomie contraindicatie             |    22
 Tracheotomie indicatie                   |   269
 Tracheotomie methode                     |   202

 select value, count(distinct admissionid)
 from listitems
 where item = 'Tracheotomie methode'
 group by value;

-- Tracheotomie methode:

                 value                  | count
----------------------------------------+-------
 Griggs-forceps                         |    16
 Minitrach                              |    11
 PercuTwist dilatatie-schroef           |    87
 Portex Ultraperc Single-step dilatatie |    90

-- Tracheacanule type:

       value       | count
-------------------+-------
 Anders            |    16
 Portex            |   244
 Portex Minitrach  |    29
 Rusch Tracheoflex |   229
 Shiley            |   480

-- Tracheacanule maat:

   value   | count
-----------+-------
 4         |   150
 5         |    13
 6         |   265
 7         |   125
 7.5 Extra |    10
 8         |   562
 8.5 Extra |     3
 9         |    71

-- I'm going to treat any entry into the field of 'Tracheacanule maat', 'Tracheacanule type', or 'Tracheotomie methode'
-- as an indication that the patient has a tracheostomy. This will define an early end point of the ventilation for us.

*/