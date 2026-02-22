@echo off
setlocal

python code\evaluate_text_acoustic_narrative_pre_symptoms.py || exit /b 1
python code\train_text_acoustic_narrative_pre_symptoms.py || exit /b 1
python code\generate_pre_symptoms_plots.py || exit /b 1

echo Repro run completed.
endlocal
