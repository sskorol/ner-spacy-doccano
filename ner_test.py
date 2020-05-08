# coding: utf8
import plac
import spacy
from ner_train import test_model

DATA = [
    "Sounds interesting. Let's say it will be a product!",
    "Very nice idea. I'll pick a service probably.",
    "Process",
    "should be a service",
    "it seems like a product",
    "let's stick with a process",
    "I'd say it's a service, but not sure right now",
    "for me it's very close to process I guess",
    "chances are it's a product or service",
    "most likely I'd stick with a process",
    "obviously, it must be a process!",
    "I must confess: it's a service",
    "this product would be a perfect choice",
    "I can't say exactly, but my sixth sense tell me a process",
    "nice try, but there's nothing except product I can think of right now",
    "I don't know",
    "leave me alone",
    "it's a bullshit",
    "there's nothing I can think of right now",
    "nah",
    "don't know"
]


@plac.annotations(
    model=("Model name.", "option", "m", str)
)
def main(model='./model'):
    print("Loading from", model)
    nlp = spacy.load(model)
    test_model(nlp, DATA)


if __name__ == "__main__":
    plac.call(main)
