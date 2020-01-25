import json
from langutils import *
import numpy as np
from clusterutils import *
from cluster import *

def read_dataset(filename):
    with open(filename) as dataset_file:
        dataset = json.load(dataset_file)
    return dataset

def prepare_dataset(dataset):
    processed_dataset = dict()

    for question, data in dataset.items():
        answer_vectors = list(map(prepare_vector, data['Answers']))
        processed_dataset[question] = {
            'answer_vectors': answer_vectors,
            'marks': data['Marks']
        }

    return processed_dataset

def build_models(dataset):
    models = dict()

    for question, data in dataset.items():
        marks = data['marks']
        model = DissimilarVectorsKMeans(n_clusters=marks)
        model.fit(data['answer_vectors'])

        models[question] = {
            'model': model,
            'marks': marks
        }
        print("model for '", question, "' prepared.")

    return models

def prepare_QA(QA_dataset):
    QA = dict()

    for question, answer in QA_dataset.items():
        QA[question] = {
            'answer': answer,
            'answer_vector': prepare_vector(answer)
        }

    return QA

def evaluate_QA(QA, models):
    evaluation = dict()

    for question, data in QA.items():
        model = models[question]['model']

        label = model.predict([data['answer_vector']])[0]
        center = model.cluster_centers[label]
        sorted_centers = sort_by_marks(model.cluster_centers)
        marks = get_marks(sorted_centers, center)
        feedback = get_feedback(sorted_centers[-1], data['answer_vector'])

        evaluation[question] = {
            'answer': data['answer'],
            'marks_awarded': marks,
            'max_marks': len(sorted_centers),
            'feedback': feedback
        }

    return evaluation

if __name__ == "__main__":
    print("reading dataset...")
    dataset = read_dataset("dataset.json")

    print("computing basic features, POS tag and cleaning, calculating freq dist, considering synonyms...")
    processed_dataset = prepare_dataset(dataset)

    print("building models...")
    models = build_models(processed_dataset)

    print("reading QA...")
    QA_dataset = read_dataset("QA.json")

    print("formatting QA...")
    QA = prepare_QA(QA_dataset)

    print("evaluating QA...")
    evaluation = evaluate_QA(QA, models)

    print("Results:")
    marks_secured = 0
    marks_allotted = 0
    for question, data in evaluation.items():
        marks_secured += data['marks_awarded']
        marks_allotted += data['max_marks']

        print("Question:", question)
        print("Answer:", data['answer'])
        print("Marks Awarded:", data['marks_awarded'])
        print("Max Marks:", data['max_marks'])
        print("Feedback:", data['feedback'])
        print()

    print("Total Marks Secured:", marks_secured, "/", marks_allotted)