# Employee Absenteeism Streamlit App

Deployed Link :- https://absenteeism-analysis-xbvkm5459u3t8tnjgjsnxh.streamlit.app/

## Run

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Then open `http://localhost:8501`.

## Important
- Keep `best_absenteeism_model.pkl` in the same folder as `app.py`.
- This model was created with scikit-learn 1.6.1, so the requirements file pins that version.
- The visual theme uses the supplied Buttercup Sky palette:
  - #FFF2B2
  - #A8C6E7
  - #FFE08A
  - #FFF7D6
  - #7FA8D6
