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


```bash
git clone https://github.com/mbzuai-nlp/fire.git
cd fire
pip install -r requirements.txt

# Run FIRE with GPT-4o-mini
python run_fire.py --model gpt-4o-mini --dataset factcheck_bench
```

```bash
python run_fire_rag.py --benchmark hover --build_ratio 0.8 --build_only      
# Step 2: Evaluate at different train ratios - reuses same database                                                                                                                                                             
python run_fire_rag.py --benchmark hover --train_ratio 0.2 --eval_only                                                                                                                                                          
python run_fire_rag.py --benchmark hover --train_ratio 0.4 --eval_only                                                                                                                                                          
python run_fire_rag.py --benchmark hover --train_ratio 0.6 --eval_only                                                                                                                                                          
python run_fire_rag.py --benchmark hover --train_ratio 0.8 --eval_only 
```