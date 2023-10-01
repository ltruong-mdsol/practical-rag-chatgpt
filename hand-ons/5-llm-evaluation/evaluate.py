import pandas as pd
from vector_rag import query_engine
from datasets import load_dataset
from ragas import evaluate

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas.metrics.critique import harmfulness


dataset = load_dataset('csv', data_files='predcition.csv')


result = evaluate(
    dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
        harmfulness,
    ],
)

print(result)

# dataset = pd.read_csv('eval_dataset.csv')


# def predict(question):
#     response = query_engine.query(question).response
    
#     print(response)
    
#     return response


# dataset['prediction'] = dataset['question'].map(predict)

# dataset.to_csv('predcition.csv', index=False)


