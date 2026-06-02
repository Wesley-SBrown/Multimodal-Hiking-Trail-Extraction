# ==============================================================================
# MULTIMODAL HIKING TRAIL EXTRACTION AUTOMATION PIPELINE
# ==============================================================================
Clear-Host

# 1. Define Environmental Paths
$PYTHON_EXE = "C:/Users/flyin/Miniconda3/envs/hike/python.exe"
$PROJECT_ROOT = "C:/Users/flyin/Multimodal-Hiking-Trail-Extraction"

# Move to the project root and force clean variable indexing
cd $PROJECT_ROOT
$env:PYTHONPATH = $PROJECT_ROOT
$env:KMP_DUPLICATE_LIB_OK = "TRUE" # Suppresses OpenMP thread crashes on Windows

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "🚀 STARTING MULTIMODAL TRAIL PIPELINE" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# 2. Step 1: Preprocessing (Jumping inside its local directory due to relative path logic)
Write-Host "`n[STEP 1/4] Running Data Preprocessing..." -ForegroundColor Yellow
cd "$PROJECT_ROOT/src/data"
& $PYTHON_EXE preprocessing.py
if ($LASTEXITCODE -ne 0) { 
    Write-Host "❌ Preprocessing failed! Exiting pipeline." -ForegroundColor Red; exit 
}

# Return to root for standard package execution mapping
cd $PROJECT_ROOT

# 3. Step 2: Model Training
Write-Host "`n[STEP 2/4] Running Model Training (CUDA Optimized)..." -ForegroundColor Yellow
& $PYTHON_EXE -m src.training.train
if ($LASTEXITCODE -ne 0) { 
    Write-Host "❌ Training failed! Exiting pipeline." -ForegroundColor Red; exit 
}

# 4. Step 3: Inference & Topological Reconstruction
Write-Host "`n[STEP 3/4] Running Inference and Vectorization..." -ForegroundColor Yellow
& $PYTHON_EXE -m src.inference.reconstruct
if ($LASTEXITCODE -ne 0) { 
    Write-Host "❌ Inference failed! Exiting pipeline." -ForegroundColor Red; exit 
}

# 5. Step 4: Map Visualization
Write-Host "`n[STEP 4/4] Launching Interactive Output Visualizer..." -ForegroundColor Yellow
& $PYTHON_EXE -m src.visualization.view_output
if ($LASTEXITCODE -ne 0) { 
    Write-Host "⚠️ Visualizer closed or failed to launch. Data is safely saved." -ForegroundColor Yellow 
}

Write-Host "`n=========================================" -ForegroundColor Green
Write-Host "🎉 COMPLETE PIPELINE EXECUTED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "Final Matrix Saved to: data/output_extracted_trails.geojson" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green