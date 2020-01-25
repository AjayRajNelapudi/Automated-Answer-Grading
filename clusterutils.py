

def calculate_weight(vector):
    feature_weights = {
        'word_count': 0.2,
        'sentence_count': 0.2,
        'correct_ratio': 0.1,
        'clean_words': 0.5
    }
    vector_weight = 0
    for feature, weight in feature_weights.items():
        vector_weight += weight * vector[feature]

    return vector_weight

def sort_by_marks(cluster_centers):
    cluster_centers = list(cluster_centers)
    cluster_centers.sort(key=calculate_weight)
    return cluster_centers

def get_marks(cluster_centers, matching_centroid):
     marks = cluster_centers.index(matching_centroid) + 1
     return marks

def get_feedback(best_answer, student_answer):
    feedback = []

    if best_answer['clean_words'] > student_answer['clean_words']:
        feedback.append("Increase content.")
    if best_answer['sentence_count'] > student_answer['sentence_count']:
        feedback.append("Improve presentation.")
    if best_answer['correct_ratio'] > student_answer['correct_ratio']:
        feedback.append("Reduce spelling mistakes.")
    if feedback == []:
        feedback.append("Proper answer.")

    return "\n".join(feedback)
