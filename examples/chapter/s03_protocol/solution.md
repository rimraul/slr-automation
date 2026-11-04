# S3 pavyzdinis atsakymas -- protokolo apibrėžimas

Tai vienas galimas geras atsakymas, ne vienintelis teisingas.

## (a) PCC ar PICO?

**PCC.** Tema yra apie tai, *kaip ir kur* dirbtinis intelektas naudojamas
literatūros apžvalgose (praktikų, metodų ir kontekstų žemėlapis), o ne apie
konkrečios intervencijos poveikį palyginti su kontrole. Tai tiriamosios
(scoping) apžvalgos klausimas, todėl PICO netinka -- nėra nei „intervencijos"
prieš „palyginimą", nei iš anksto apibrėžto „baigties" rodiklio.

## (b) PCC lentelė

| Elementas | Apibrėžimas |
|---|---|
| Populiacija (P) | Recenzuojamos publikacijos ir įrankių aprašymai, kuriuose vykdoma ar vertinama sisteminės, tiriamosios ar greitosios literatūros apžvalgos eiga. |
| Koncepcija (C) | Bent vieno apžvalgos etapo (paieška, deduplikacija, atranka, pilno teksto gavimas, duomenų išgava, sintezė) automatizavimas DKM, mašininio mokymosi ar taisyklių pagrindu. |
| Kontekstas (C) | Bet kuri mokslo sritis; angliškai skelbti darbai nuo 2015 m.; akademinės arba taikomosios apžvalgos. |

**Tyrimo klausimai.** RK1: kurie apžvalgos etapai automatizuojami dažniausiai
ir kokiais metodais? RK2: kaip darbuose vertinamas automatizuoto etapo
tikslumas ir kokia žmogaus priežiūra paliekama?

## (c) Kriterijai

**Įtraukimo:**
1. Darbe aprašomas arba vertinamas automatizuotas ar pusiau automatizuotas
   metodas bent vienam aiškiai įvardytam apžvalgos etapui.
2. Etapo automatizavimas yra darbo objektas (ne atsitiktinis paminėjimas).
3. Pateikta pakankamai metodo detalių, kad skaitytojas suprastų, kas buvo
   automatizuota ir kaip tai įvertinta.

**Atmetimo:**
1. Darbas apie pirminių tyrimų automatizavimą (pvz., tyrimų registrai,
   duomenų rinkimas), ne apie literatūros apžvalgos procesą.
2. Nuomonės straipsnis, redakcinis tekstas ar naujiena be metodo ir be
   vertinimo.
3. Pilnas tekstas ne anglų kalba.

## (d) Savęs patikra: `rec-09`

Santrauka: „BERTopic is applied to twelve thousand nursing-research abstracts
to describe the thematic structure of the field ... a substantive application
rather than a methods contribution".

**Sprendimas: borderline.** Įtraukimo kriterijus 1 dalinai tenkinamas (temų
modeliavimas gali būti sintezės etapo automatizavimas), bet kriterijus 2
netenkinamas: darbas naudoja BERTopic *kaip įrankį savo dalykinei analizei*,
o ne tiria patį automatizavimą. Toks atvejis siunčiamas žmogui peržiūrėti --
tai ir yra „borderline" prasmė S7 atrankoje.
