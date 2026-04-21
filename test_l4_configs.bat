@echo off
echo Running L4 Inference Diagnostic Matrix
echo ---------------------------------------

set CHECKPOINT=checkpoints\best_model.pt

if not exist "%CHECKPOINT%" (
    echo Error: Checkpoint not found at %CHECKPOINT%
    exit /b 1
)

echo.
echo === 1. No Compile, No AMP (Safest) ===
.venv\Scripts\python -m src.training.train --profile l4 --eval_only --resume_checkpoint %CHECKPOINT% --compile_model_override False --amp_dtype_override disabled --dump_eval_predictions test_l4_no_compile_no_amp.jsonl

echo.
echo === 2. AMP Only ===
.venv\Scripts\python -m src.training.train --profile l4 --eval_only --resume_checkpoint %CHECKPOINT% --compile_model_override False --amp_dtype_override float16 --dump_eval_predictions test_l4_amp_only.jsonl

echo.
echo === 3. Compile Only ===
.venv\Scripts\python -m src.training.train --profile l4 --eval_only --resume_checkpoint %CHECKPOINT% --compile_model_override True --amp_dtype_override disabled --dump_eval_predictions test_l4_compile_only.jsonl

echo.
echo === 4. Compile and AMP (Full L4 Default) ===
.venv\Scripts\python -m src.training.train --profile l4 --eval_only --resume_checkpoint %CHECKPOINT% --compile_model_override True --amp_dtype_override float16 --dump_eval_predictions test_l4_compile_and_amp.jsonl

echo.
echo Diagnostic Matrix Complete. Compare the median RMSD and chemistry metrics across the 4 runs.
