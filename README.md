# Medical Spell Corrector

A spelling corrector for medical incorrect words/text

## Data
Data has been scrapped from different websites using scrapy and is available in the data directory of this repository

## Usage
Clone the repository and run the following commands from the terminal to use spell corrector 

#### Spell Corrector for Text
```
 python .\medicalSpellCorrector_text.py --text "INCORRECT TEXT HERE"
```
#### Spell Corrector for Word
```
 python .\medicalSpellCorrector_word.py --word "INCORRECT WORD HERE"
```
#### Training on New Terms
To train new words just add those new words in one of the json files in data folder correctly and run the following command

```
 python .\medicalSpellCorrector_train.py  
```

## Results
#### Spell Corrector Text
```bash
INPUT TEXT     :: i ned gaencolgst
CORRECTED TEXT :: i need gynecologist
PROCESS TIME   :: 0.046874046325683594
```
```bash
INPUT TEXT     :: i want an obstrican apontmnt
CORRECTED TEXT :: i want an obstetrician appointment
PROCESS TIME   :: 0.055844783782958984
```
#### Spell Corrector Word
```bash
INPUT WORD     :: psicatrist
CORRECTED WORD :: psychiatrist
PROCESS TIME   :: 0.052858829498291016
```
```bash
INPUT WORD     :: abdmnal
CORRECTED WORD :: abdominal
PROCESS TIME   :: 0.04288673400878906
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

