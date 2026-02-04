```
Input Claim
   │
   ▼
[FIRE Decision Module]
   ├── confident → Output Label (True / False)
   └── uncertain → Generate Search Query
                      │
                      ▼
          Web Search (via SerperAPI)
                      │
                      ▼
            Update Evidence Set
                      │
                      └── Loop until confident or max steps
```

GPT-4o knowledge cut-off date: Oct. 2023


```bash
git clone https://github.com/mbzuai-nlp/fire.git
cd fire
pip install -r requirements.txt

# Run FIRE with GPT-4o-mini
python run_fire.py --model gpt-4o-mini --dataset factcheck_bench
```

```bash
# construct the rag_db
python run_fire_rag.py --benchmark hover --build_ratio 0.8 --build_only      

# Test based on the constructed rag_db
python run_fire_rag.py --benchmark hover --train_ratio 0.2 --eval_only 
python run_fire_rag.py --benchmark hover --train_ratio 0.4 --eval_only 
python run_fire_rag.py --benchmark hover --train_ratio 0.6 --eval_only 
python run_fire_rag.py --benchmark hover --train_ratio 0.8 --eval_only 
```

### Some new thinking: 

When we test, we still allow web retrieval for the test samples. This is a weird setting for RAG benchmarking. 

We should just only use the knowlegde built on training set, and do not allow retrieval for test samples. Otherwise the test samples always get the latest web info and 
the web knowledge overwelms the rag_db knowledge. 


### Update notes

Feb. 1: Implemented rag_db memory for retrieval

Fed. 3: Serper only return a snippet for website, ADD web crawling to get whole web page's content.