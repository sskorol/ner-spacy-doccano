## NER using Doccano / Spacy EN  

This repository provides a basic example of a named entity extraction (NER) task for English model.

### Preparation

Make sure you have Python >= 3.6 and Docker installed.

Setup [Doccano](https://github.com/doccano/doccano):
```shell script
git clone https://github.com/doccano/doccano.git
cd doccano
docker-compose -f docker-compose.prod.yml up -d
```

Setup a sample project based on [Spacy](https://spacy.io/):
```shell script
git clone https://github.com/sskorol/ner-spacy-doccano.git
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

### Annotating Text

- Open Doccano UI: http://localhost
- Sign in with default credentials: admin / password
- Create a new project
- Watch provided tutorial on a home screen
- Import `./data_to_be_annotated.txt`
- Create a new label, e.g. `CATEGORY`
- Start annotating imported data by marking the following words as `CATEGORY`: product, process, service
- When all the sentences are labeled, export JSON with text labels
- Save it into project root and give it `categories.json` name

Note that exported JSON will be in unusual format. You have to convert the content into array: wrap it with `[]` and add commas after each line.

### Model Training

The following command will run a script which adjusts an existing English model with a new `CATEGORY` label and performs a training based on annotated data.  

```shell script
python3 ./ner_train.py
```

You should see something similar as an output:
```text
Loaded model 'en_core_web_sm'
Losses {'ner': 846.7723659528049}
Losses {'ner': 623.0931596025007}
Losses {'ner': 689.6105882608678}
------------------------
Entities in 'I'm thinking about several categories. Let me start with the service one.'
CATEGORY service
------------------------
Entities in 'Let’s choose a product'
CATEGORY product
------------------------
Entities in 'It is quite a well-known service'
CATEGORY service
------------------------
```

Verification data should give you a confidence if you model is accurate.

Apart from that, there should be a new NER `./model` folder created.

### Model Testing

Run the following command to test the generated model on a custom data:

```shell script
python3 net_test.py
```

You should see a similar output which confirms model's confidence level:

```text
Loading from ./model
------------------------
Entities in 'Sounds interesting. Let's say it will be a product!'
CATEGORY product
------------------------
Entities in 'Very nice idea. I'll pick a service probably.'
CATEGORY service
------------------------
Entities in 'Process'
CATEGORY Process
------------------------
``` 