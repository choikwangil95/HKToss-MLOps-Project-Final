from fastapi.utils.label import id2label
from utils.ner_pipeline import load_ner_pipeline
from utils.stock_dict import load_company_names

ner_pipeline = load_ner_pipeline()
company_names = load_company_names()

def split_text(text, max_length=500):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def extract_stock_labels_by_label_id(text: str):
    chunks = split_text(text)
    matched_names = set()
    
    for chunk in chunks:
        entities = ner_pipeline(chunk)

        merged_entities = []
        current_word = ''
        current_score = []
        current_label = ''
        
        for ent in entities:
            word = ent['word']
            score = ent['score']
            label_id = ent['entity_group']
            label_num = int(label_id.split('_')[1])
            label_name = id2label.get(label_num, "")
            key = label_name[2:] if label_name.startswith(('B-', 'I-')) else label_name
            
            if key.startswith('OGG_ECONOMY'):
                if word.startswith('##'):
                    current_word += word[2:]
                    current_score.append(score)
                else:
                    if current_word:
                        merged_entities.append((current_word, current_label, sum(current_score)/len(current_score)))
                    current_word = word
                    current_score = [score]
                    current_label = label_name

        if current_word:
            merged_entities.append((current_word, current_label, sum(current_score)/len(current_score)))

        for word, label, score in merged_entities:
            if word in company_names:
                matched_names.add(word)

    return list(matched_names)

