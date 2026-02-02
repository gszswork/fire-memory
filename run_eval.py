import json
from sklearn.metrics import classification_report, balanced_accuracy_score

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def eval_averitec():
    averitec_fire_res = "fire/results/fire_averitec_gpt-4o-mini.jsonl"
    averitec_truth = "fire/datasets/averitec/data.jsonl"

    pred = load_jsonl(averitec_fire_res)
    truth = load_jsonl(averitec_truth)

    p, t = [],[]
    for i in range(len(truth)):
        assert pred[i]['claim'] == truth[i]['claim']
        if pred[i]['result']['answer'].lower() == 'false':
            p.append(0)
        elif pred[i]['result']['answer'].lower() == 'true':
            p.append(1)
        else:
            assert "prediction is "+ pred[i]['result']['answer']
        
        if truth[i]['label'].lower() == 'supported':
            t.append(1)
        elif truth[i]['label'].lower() == 'refuted':
            t.append(0)
        else:
            t.append(0)
            assert 'groundtruth is ' + truth[i]['label']
    
    print(len(pred), len(truth), len(p), len(t))

    
    print(classification_report(t, p, digits=4))

    print("Balanced Accuracy: {:.4f}".format(balanced_accuracy_score(t, p)))


def eval_summeval():
    print('FIRE evaluation on Summeval dataset: ')
    summeval_fire_res = "results/fire_summeval_gpt-4o-mini.jsonl"
    summeval_truth = "datasets/summeval/data.jsonl"

    pred = load_jsonl(summeval_fire_res)
    truth = load_jsonl(summeval_truth)

    p, t = [],[]
    for i in range(len(truth)):
        assert pred[i]['claim'] == truth[i]['claim']
        if pred[i]['result']['answer'].lower() == 'false':
            p.append(0)
        elif pred[i]['result']['answer'].lower() == 'true':
            p.append(1)
        else:
            assert "prediction is "+ pred[i]['result']['answer']

        t.append(truth[i]['label'])        
        # if truth[i]['label'].lower() == 'supported':
        #     t.append(1)
        # elif truth[i]['label'].lower() == 'refuted':
        #     t.append(0)
        # else:
        #     t.append(0)
        #     assert 'groundtruth is ' + truth[i]['label']
    
    print(len(pred), len(truth), len(p), len(t))

    
    print(classification_report(t, p, digits=4))

    print("Balanced Accuracy: {:.4f}".format(balanced_accuracy_score(t, p)))
    

def eval_aggrefact_cnn():
    print('FIRE evaluation on Summeval dataset: ')
    aggrefact_cnn_fire_res = "results/fire_aggrefact_cnn_gpt-4o-mini.jsonl"
    aggrefact_cnn_truth = "datasets/aggrefact_cnn/data.jsonl"

    pred = load_jsonl(aggrefact_cnn_fire_res)
    truth = load_jsonl(aggrefact_cnn_truth)

    p, t = [],[]
    for i in range(len(truth)):
        assert pred[i]['claim'] == truth[i]['claim']
        if pred[i]['result']['answer'].lower() == 'false':
            p.append(0)
        elif pred[i]['result']['answer'].lower() == 'true':
            p.append(1)
        else:
            assert "prediction is "+ pred[i]['result']['answer']

        t.append(truth[i]['label'])        
        # if truth[i]['label'].lower() == 'supported':
        #     t.append(1)
        # elif truth[i]['label'].lower() == 'refuted':
        #     t.append(0)
        # else:
        #     t.append(0)
        #     assert 'groundtruth is ' + truth[i]['label']
    
    print(len(pred), len(truth), len(p), len(t))

    
    print(classification_report(t, p, digits=4))

    print("Balanced Accuracy: {:.4f}".format(balanced_accuracy_score(t, p)))
  
def eval_pubhealth():
    print('FIRE evaluation on Summeval dataset: ')
    pubhealth_fire_res = "results/fire_pubhealth_gpt-4o-mini.jsonl"
    pubhealth_truth = "datasets/pubhealth/data.jsonl"

    pred = load_jsonl(pubhealth_fire_res)
    truth = load_jsonl(pubhealth_truth)

    p, t = [],[]
    for i in range(len(truth)):
        assert pred[i]['claim'] == truth[i]['claim']
        if pred[i]['result']['answer'].lower() == 'false':
            p.append(0)
        elif pred[i]['result']['answer'].lower() == 'true':
            p.append(1)
        else:
            assert "prediction is "+ pred[i]['result']['answer']

        t.append(truth[i]['label'])        
        # if truth[i]['label'].lower() == 'supported':
        #     t.append(1)
        # elif truth[i]['label'].lower() == 'refuted':
        #     t.append(0)
        # else:
        #     t.append(0)
        #     assert 'groundtruth is ' + truth[i]['label']
    
    print(len(pred), len(truth), len(p), len(t))

    
    print(classification_report(t, p, digits=4))

    print("Balanced Accuracy: {:.4f}".format(balanced_accuracy_score(t, p)))
 

if __name__ == '__main__':
    # eval_averitec()
    # eval_summeval()
    # eval_aggrefact_cnn()
    # eval_pubhealth()
    pass