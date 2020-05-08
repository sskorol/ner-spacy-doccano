# coding: utf8
import json
import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding

LABELS = ["CATEGORY"]
TRAINING_DATA = []
VERIFICATION_DATA = []


@plac.annotations(
    model=("Model name.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model='en_core_web_sm', new_model_name='ner', output_dir='./model', n_iter=30):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)

    # Create training and verification datasets from annotated file content
    __preprocess_data('categories.json')

    # Load existing spaCy model
    nlp = spacy.load(model)
    print("Loaded model '%s'" % model)

    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    # Add new labels to entity recognizer
    for label in LABELS:
        ner.add_label(label)

    # We assume an existing model modification
    optimizer = nlp.resume_training()

    move_names = list(ner.move_names)
    # Get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        sizes = compounding(1.0, 4.0, 1.001)
        # Batch up the examples using spaCy's minibatch
        for itn in range(n_iter):
            batches = minibatch(TRAINING_DATA, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
            print("Losses", losses)

    # Test the trained model via verification dataset
    test_model(nlp, [item[0] for item in VERIFICATION_DATA])

    # Save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()

        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # Test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # Check the classes have loaded back consistently
        assert nlp2.get_pipe("ner").move_names == move_names
        test_model(nlp2, [item[0] for item in VERIFICATION_DATA])


def test_model(nlp, data):
    for item in data:
        doc = nlp(item)
        print('------------------------')
        print("Entities in '%s'" % item)
        for entity in doc.ents:
            print(entity.label_, entity.text)


def __preprocess_data(file_name):
    transformed_data = []
    with open(file_name) as json_file:
        annotations = json.load(json_file)
        for annotation in annotations:
            labels = annotation['labels']
            if len(labels) > 0:
                entities = [tuple((label[0], label[1], label[2])) for label in labels]
                transformed_data.append(tuple((annotation['text'], {'entities': entities})))

    # Randomize and split into training and verification datasets
    random.shuffle(transformed_data)
    transformed_data_len = len(transformed_data)
    training_data_len = int(round(0.7 * transformed_data_len))

    for i in range(training_data_len):
        TRAINING_DATA.append(transformed_data[i])

    for i in range(training_data_len, transformed_data_len):
        VERIFICATION_DATA.append(transformed_data[i])


if __name__ == "__main__":
    plac.call(main)
