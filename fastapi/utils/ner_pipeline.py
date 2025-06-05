from transformers import pipeline

# NER 파이프라인
model_name = 'KPF/KPF-BERT-NER'
ner_pipeline = pipeline(
    task='ner',
    model=model_name,
    tokenizer=model_name,
    aggregation_strategy='simple',
    framework='pt'
)