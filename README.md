# Run server: 
python -m uvicorn main:app --reload

# Process ChartQA dataset:
./convert_json.sh

# Run Bayesian Optimization:
./vegaair_bo.sh
