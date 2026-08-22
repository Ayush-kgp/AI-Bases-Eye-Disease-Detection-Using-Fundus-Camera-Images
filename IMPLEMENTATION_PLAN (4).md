# RetinaScan AI — Phased Build Plan (for Antigravity)
### AI-Based Eye Disease Detection Using Fundus Camera Images

**How to use this document:** This is written for one continuous Antigravity session across all phases. Paste the "Session Context" block once at the very start, then paste each phase's prompt block in order. Phases are sequential — do not start Phase N+1 until Phase N's "Verification Gate" passes.

**Why phases still re-state file paths even in one session:** in a long session, an agent can drift — by Phase 6 it may "remember" a wrong version of what Phase 2 actually concluded (e.g. assume the ensemble won when the real numbers showed a solo model won), especially with this many JSON result files in play. To guard against that, every phase that depends on an earlier phase's *decision* (not just its files) tells the agent to re-read the actual file on disk and state what it finds, rather than relying on its memory of the conversation. Don't skip these re-confirmation steps even though it's the same session — they're cheap, and they're exactly where silent drift would otherwise slip through.

**Before Phase 0:** confirm your `.pth` checkpoints, `eval_baseline_results.json`, `ensemble_results.json`, and `*_test_probs.pt` files are all in one accessible directory, and know that path.

---

## Session Context (paste this once, first, before Phase 0)

```
We're building a backend for a fundus-image eye disease classifier across
several sequential phases in this one session. Here is the context you'll
need for all of it — refer back to this rather than asking me to repeat
paths later, but if any phase asks you to re-verify a fact against a file
(not just recall it), actually re-read that file rather than relying on
what I tell you here or what we discussed earlier in the session.

Existing files on disk, all under this base directory: [PASTE YOUR BASE
DIRECTORY PATH, e.g. /path/to/kaggle_outputs/]

- EfficientNet checkpoint: [FULL PATH].pth
- MobileNet checkpoint: [FULL PATH].pth
- Model architecture / class definitions: [FULL PATH to train.py or wherever
  the model classes are defined]
- eval_baseline_results.json: [FULL PATH]
- ensemble_results.json (from my existing ensemble.py): [FULL PATH]
- Test dataset / dataloader code (for real verification images, not dummy
  data): [FULL PATH to eval_baseline.py or dataset definition]
- class_to_idx mapping: [FULL PATH or note where it's saved/defined]
- My existing working Grad-CAM code: [PASTE THE CODE OR A FULL PATH TO IT]
- A few known-label test images for manual sanity checks later (at least
  3 Glaucoma, 3 Diabetic Retinopathy): [FULL PATH to a folder, or note that
  these can be pulled from the test dataloader]

We will build this new project in a separate directory:
retinascan-backend/ (I'll say when to create it)

10 classes total, this is a fundus (retinal) image classifier. Two
backbones were trained independently: EfficientNet and MobileNet.

One thing I want to be explicit about for the whole session: we do NOT
yet know whether the ensemble (combining both models) actually beat the
better solo model on real evaluation numbers — this gets checked for
real in Phase 2, using the actual JSON files above, not assumed. Please
don't default to assuming "the ensemble" is the final answer anywhere in
this session until Phase 2 confirms it with real numbers.

Acknowledge you have this context, then wait for Phase 0.
```

---

## Phase 0 — Repository Scaffold

**Goal:** Create the folder structure so every later phase has a known place to put files.

**Prompt block:**
```
Create the following empty directory structure for a Python backend project called
retinascan-backend. Do not write any code yet, just create the folders and empty
__init__.py files where needed for Python packages:

retinascan-backend/
├── app/
│   ├── __init__.py
│   ├── routes/
│   │   └── __init__.py
│   ├── services/
│   │   └── __init__.py
│   ├── schemas/
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── tests/
├── ml/
│   ├── export/
│   └── models/
├── docs/
└── streamlit_app/

Also create an empty requirements.txt and .gitignore (Python + model files:
ignore __pycache__, .onnx, .pth, venv/, .env — model artifacts should not be
committed directly to git; note this in a comment in .gitignore).
```

**Verification gate:** Folder structure exists, matches the tree above, `.gitignore` excludes model binaries.

---

## Phase 1 — ONNX Export + Numerical Verification

**Goal:** Export both `.pth` models to ONNX and *prove* the export didn't change their behavior. This phase is not done until the numbers match — an export that "runs without error" is not verified.

**Inputs the agent needs:** the checkpoint paths, architecture definitions, and test dataloader already given in the Session Context block.

**Prompt block:**
```
Using the EfficientNet and MobileNet checkpoint paths and model architecture
definitions from the session context above, write two export scripts,
ml/export/export_effnet.py and ml/export/export_mobilenet.py,
that each:
1. Load the .pth checkpoint into the correct model architecture
2. Call model.eval() before doing anything else — this is critical, do not skip it,
   since leaving BatchNorm/Dropout in training mode will silently produce wrong
   outputs after export
3. Export to ONNX using torch.onnx.export() with a dummy input tensor matching
   the real preprocessing output shape (confirm this shape from the training
   code, don't guess it)
4. Save to ml/models/efficientnet.onnx and ml/models/mobilenet.onnx respectively

Then write a single verification script, ml/export/verify_onnx_export.py, that:
1. Loads a batch of real test images using the existing test dataset/dataloader
   from the session context — do not use random dummy data, it must be real
   images
2. Runs them through the ORIGINAL .pth model (in eval mode) and separately
   through the EXPORTED .onnx model (via onnxruntime)
3. Compares the two sets of outputs numerically (use something like
   numpy.allclose with a small tolerance, e.g. atol=1e-4) and reports whether
   they match
4. Additionally recomputes macro-F1 on the ONNX model's predictions for this
   batch and prints it next to the macro-F1 already recorded in
   eval_baseline_results.json for that model, so I can visually confirm they
   are close
5. Prints a clear PASS or FAIL verdict for each model, not just raw numbers

Run this for BOTH models. Do not tell me the export is done until this
verification script shows PASS for both — if it shows a mismatch, debug
the export script (check eval mode, check input shape, check opset version)
rather than lowering the tolerance to force a pass.
```

**Verification gate:** `verify_onnx_export.py` prints PASS for both models, with ONNX macro-F1 visibly close to the existing `eval_baseline_results.json` numbers. Do not proceed to Phase 2 on a FAIL.

---

## Phase 2 — Ensemble Config Generation

**Goal:** Produce `ensemble_config.json` as the single source of truth for combination logic, class mappings, and which classes have thin test support. This file must reflect your *actual* evaluation results, not an assumed "the ensemble won" default.

**Inputs the agent needs:** `eval_baseline_results.json` and `ensemble_results.json`, both already given in the Session Context block. **This phase is the most important place in the whole session to insist on a fresh, literal file read** — don't let the agent answer from anything it may have inferred or assumed earlier in the session.

**Prompt block:**
```
Open and actually read eval_baseline_results.json and ensemble_results.json
fresh right now, from their paths in the session context — don't answer
from anything you may have inferred about them earlier in this session.

First, tell me: which of these four options had the
highest test macro-F1 — EfficientNet solo, MobileNet solo, simple-average
ensemble, or weighted-average ensemble? Show me the actual numbers you
found for all four before doing anything else.

Once I confirm which one won, write ml/export/generate_ensemble_config.py
that reads those same two JSON files plus the class_to_idx mapping from
the path given in the session context, and produces
ml/models/ensemble_config.json with this structure:

{
  "class_to_idx": { ... },
  "class_to_idx_reverse": { ... },
  "combination_method": "weighted_average" | "simple_average" | "efficientnet_solo" | "mobilenet_solo",
  "weights": { "efficientnet": <float>, "mobilenet": <float> },
  "solo_test_macro_f1": { "efficientnet": <float>, "mobilenet": <float> },
  "ensemble_test_macro_f1": <float or null if a solo model won>,
  "low_support_classes": [ list of class names with fewer than 25 test samples,
    read from eval_baseline_results.json's per-class support numbers ]
}

IMPORTANT: set "combination_method" to whichever option ACTUALLY won based
on the real numbers we just looked at — do not default to "weighted_average"
if it wasn't the best performer. If a solo model won, set weights to null
and set combination_method to "efficientnet_solo" or "mobilenet_solo"
accordingly. This file must honestly reflect what the evaluation actually
found.

Run the script and show me the generated ensemble_config.json content so
I can confirm it's correct before moving on.
```

**Verification gate:** You have personally looked at the printed macro-F1 comparison and confirmed `combination_method` in the generated file matches the actual winner. Do not let the agent guess this — read the numbers yourself.

---

## Phase 3 — Live Inference Service

**Goal:** Build `EnsembleInferenceService` — the centerpiece component. This runs two real forward passes on a live uploaded image and combines them per `ensemble_config.json`, with a startup safety check that fails loudly on any class-mapping inconsistency.

**Inputs the agent needs:** the ONNX files and `ensemble_config.json` from Phases 1–2, and the training preprocessing code from the session context.

**Prompt block:**
```
Before writing any code, open ml/models/ensemble_config.json fresh and
tell me its combination_method field — confirm this matches what we
concluded in Phase 2 before proceeding, since this determines how
predict() below needs to branch.

Using the files at ml/models/efficientnet.onnx, ml/models/mobilenet.onnx,
and ml/models/ensemble_config.json, build the live inference layer.

First, write app/services/preprocessing.py containing a single function
that replicates EXACTLY the preprocessing used during training, based on
the training preprocessing code from the session context. This must produce numerically
identical output to what the models were trained on — same resize
dimensions, same normalization values, same color channel order. Do not
guess these values; extract them directly from the training code.

Then write app/services/inference.py containing an EnsembleInferenceService
class with this behavior:

1. __init__(self, config_path, effnet_onnx_path, mobilenet_onnx_path):
   - Load ensemble_config.json
   - Create onnxruntime.InferenceSession for both models
   - Run a startup validation check: confirm the number of output classes
     from both ONNX models' output shape matches len(class_to_idx) in the
     config. If they don't match, raise a clear, loud exception immediately
     at startup — do NOT let the service start in a broken state. This
     mirrors a safety check my existing ensemble.py script does before
     combining any two models' outputs; port that same "verify before
     combining" discipline here.

2. predict(self, raw_image_bytes) -> a result object/dict containing:
   - predicted_class (string, human-readable disease name)
   - confidence (float)
   - combination_method (copied from config)
   - all_class_probabilities (dict of class_name -> probability, all classes)
   - solo_predictions: what EfficientNet alone predicted, what MobileNet
     alone predicted, and whether they agreed
   - is_low_support_class (bool, True if predicted_class is in the
     config's low_support_classes list)
   - inference_time_ms (measured, not estimated)

   Inside predict():
   - Call the preprocessing function from step 1 on the raw image
   - Run both ONNX sessions to get softmax probabilities from each
   - Combine them according to config["combination_method"]:
     - "weighted_average": weighted sum using config["weights"]
     - "simple_average": plain average
     - "efficientnet_solo" or "mobilenet_solo": use only that model's
       output, skip combination entirely
   - Compute all the result fields above from the combined probabilities

Then write tests/test_inference.py with at least these test cases:
- A test that intentionally constructs a broken ensemble_config.json
  (mismatched class count) and confirms EnsembleInferenceService raises
  an exception at construction time, not silently continuing
- A test that runs predict() on a real sample image and confirms the
  response has all the expected fields with sensible types/ranges
  (confidence between 0 and 1, all_class_probabilities sums to ~1.0)
- If combination_method is a "_solo" variant, a test confirming the
  service actually only uses that one model's output and ignores the other

Run the tests and show me the results before moving on.
```

**Verification gate:** All tests in `test_inference.py` pass, including the deliberately-broken-config test raising an exception. Manually run `predict()` on one real image and sanity-check the output makes sense (right disease type for a known test image, confidence in a reasonable range).

---

## Phase 4 — Grad-CAM for Both Backbones

**Goal:** Port your existing Grad-CAM code to work with the live ONNX inference path (or the loaded `.pth` models directly, whichever is cleaner — see note below), for both EfficientNet and MobileNet.

**Important note before starting this phase:** Grad-CAM requires gradient access to intermediate layers, which plain ONNX Runtime inference does not give you easily. Decide with the agent up front whether Grad-CAM should run against the original `.pth` PyTorch models (loaded separately, alongside the ONNX inference path, just for the heatmap) or whether you'll use an ONNX-compatible Grad-CAM approach. Running it on the original PyTorch models is usually simpler and is a legitimate, explainable design choice — say so explicitly in your documentation rather than treating it as a compromise.

**Inputs the agent needs:** your existing Grad-CAM code and the `.pth` checkpoints, both from the session context.

**Prompt block:**
```
Using the existing working Grad-CAM code from the session context (it
currently works for EfficientNet), I need to:
1. Port this into app/services/gradcam.py as a clean, reusable function
   that takes a loaded PyTorch model, a hook target layer, and a
   preprocessed image tensor, and returns a heatmap overlay image
   (base64-encoded PNG, overlaid on the original image with jet colormap,
   ~0.4 alpha blend)
2. Confirm the correct hook layer for EfficientNet (likely already known
   from the existing code) and get the equivalent working for MobileNet
   — MobileNet's architecture is different (depthwise-separable conv
   blocks), so identify its correct final convolutional layer for hooking;
   do not assume the same layer name/index as EfficientNet works here
3. Note: Grad-CAM needs gradient access, so this should load the original
   .pth models directly (not the ONNX exports) for this specific purpose.
   Write this as a small separate loading path in gradcam.py, clearly
   commented as to why it's separate from the ONNX inference path in
   inference.py.
4. Write a quick manual test script (does not need to be a formal pytest
   test) that runs Grad-CAM on a few known Glaucoma and Diabetic
   Retinopathy test images for BOTH models, saves the heatmap overlays
   as PNG files I can open and look at, and prints which image/class each
   one corresponds to.

Generate the heatmaps for both models on at least 3 Glaucoma images and
3 Diabetic Retinopathy images so I can manually check whether the
heatmaps are actually highlighting the optic disc / lesion regions
correctly, rather than lighting up random or irrelevant parts of the image.
```

**Verification gate:** You have personally opened the generated heatmap PNGs and visually confirmed they highlight clinically relevant regions (optic disc for Glaucoma, hemorrhage/microaneurysm areas for DR) rather than borders or irrelevant background. This is a manual, visual check — do not skip it or take the agent's word that it "looks right."

---

## Phase 5 — FastAPI Application

**Goal:** Wire `EnsembleInferenceService` and Grad-CAM into a working FastAPI app with the three scoped endpoints.

**Inputs the agent needs:** everything from Phases 1–4 (`app/services/inference.py`, `app/services/gradcam.py`, `app/services/preprocessing.py`, the `ml/models/` artifacts).

**Prompt block:**
```
Build the FastAPI application using the services already created in
app/services/inference.py, app/services/gradcam.py, and
app/services/preprocessing.py.

Write app/schemas/prediction.py with Pydantic models for the response
shape:

{
  "prediction": {
    "disease": str,
    "confidence": float,
    "combination_method": str,
    "class_probabilities": dict[str, float]
  },
  "model_agreement": {
    "efficientnet_prediction": str,
    "mobilenet_prediction": str,
    "agree": bool
  },
  "reliability_flag": {
    "is_low_support_class": bool,
    "note": str | None
  },
  "explainability": {
    "gradcam_overlay_base64": str
  },
  "meta": {
    "inference_time_ms": float,
    "model_version": str
  },
  "disclaimer": "Research prototype — not a certified diagnostic tool."
}

Note: if combination_method is a "_solo" variant (from Phase 2), the
model_agreement section should still show what each model individually
predicted (compute both even though only one is used for the final
answer), since that's still useful diagnostic information.

Write app/main.py with:
- FastAPI app initialization
- A lifespan event handler that loads EnsembleInferenceService ONCE at
  startup (not per-request) and stores it in app state
- Basic request logging (log method, path, response time — inline,
  does not need to be a separate middleware module)
- A global exception handler that catches unexpected errors and returns
  a clean JSON error response instead of a raw stack trace

Write app/routes/health.py with:
- GET /health — returns 200 if both ONNX models loaded successfully at
  startup, 503 otherwise
- GET /model/info — returns the contents of ensemble_config.json
  (combination_method, solo_test_macro_f1, ensemble_test_macro_f1) so
  the real evaluation numbers are visible through the API

Write app/routes/predict.py with:
- POST /predict — accepts an uploaded image file, validates it's a real
  image (reject non-image files, reject corrupted images, reject images
  below a reasonable minimum resolution, all with clean 4xx errors not
  crashes), runs it through preprocessing, EnsembleInferenceService, and
  Grad-CAM, and returns the full response schema above

Write app/utils/validators.py with the image validation logic referenced
above (file type check, corruption check via attempting to open with
PIL, minimum resolution check).

Write tests/test_api.py that:
- Tests /health returns 200 when models are loaded
- Tests /model/info returns the expected fields
- Tests /predict with a real valid test image returns 200 and a
  correctly-shaped response
- Tests /predict with a non-image file returns a clean 4xx error, not
  a 500
- Tests /predict with a corrupted/truncated image file returns a clean
  4xx error, not a 500

Run the full test suite and the app itself (uvicorn app.main:app) and
show me it starts cleanly and /health returns 200.
```

**Verification gate:** `uvicorn app.main:app` starts without errors, `/health` returns 200, `/model/info` shows your real evaluation numbers, `/predict` on a real image returns a complete, correctly-shaped response, and all tests in `test_api.py` pass including the error-handling cases.

---

## Phase 6 — Docker

**Goal:** Containerize the backend as a single service with a health check that fails loudly on missing model artifacts.

**Prompt block:**
```
Write a Dockerfile for this FastAPI backend (retinascan-backend/) that:
- Uses python:3.11-slim as the base image
- Installs dependencies from requirements.txt (make sure requirements.txt
  is complete and accurate based on everything actually imported across
  app/ — check this rather than assuming, since packages added during
  earlier phases may not all be listed yet)
- Copies the app/ directory and the ml/models/ directory (containing the
  two .onnx files and ensemble_config.json) into the image
- Exposes port 8000
- Includes a HEALTHCHECK instruction that calls the /health endpoint
- Runs via uvicorn on 0.0.0.0:8000

Also write a .dockerignore that excludes tests/, docs/, streamlit_app/,
the ml/export/ scripts, and any .pth checkpoint files (only the exported
.onnx files should go into the image, not the original PyTorch
checkpoints — the image shouldn't need PyTorch as a heavy dependency if
inference only uses onnxruntime; check whether Grad-CAM's separate .pth
loading path from Phase 4 needs to be excluded from the container image
entirely, or whether it needs torch included as a dependency — clarify
this with me before deciding, since it affects image size significantly).

Build the image (docker build -t retinascan-backend .) and run it
(docker run -p 8000:8000 retinascan-backend), then confirm:
1. The container starts successfully
2. docker ps shows the health check passing (not just "running", the
   actual HEALTHCHECK status)
3. curl http://localhost:8000/health from outside the container returns 200
4. Deliberately test the failure case: temporarily rename or remove one
   of the .onnx files from the image build context, rebuild, and confirm
   the container's health check FAILS clearly rather than the container
   silently running in a broken state — then restore the file and confirm
   it passes again
```

**Verification gate:** Image builds, container runs, health check shows passing status via `docker ps`, and the deliberate-failure test in step 4 actually fails the health check as expected (not silently succeeding).

---

## Phase 7 — Documentation

**Goal:** Write `README.md` and `docs/model_evaluation.md` — the second of which is genuinely the best interview material in this project, so don't let the agent write it as generic boilerplate.

**Prompt block:**
```
Write two documentation files.

1. README.md for the retinascan-backend repository, covering:
   - What this project is (one paragraph, plain language)
   - Architecture: EfficientNet + MobileNet ensemble, ONNX-exported,
     served via FastAPI, explained with Grad-CAM
   - Setup instructions: how to install dependencies, where model
     files need to go, how to run locally (uvicorn) and via Docker
   - API reference: the three endpoints (/health, /model/info,
     /predict), with a real example request/response for /predict
   - A short "Model Evaluation Summary" section that links to
     docs/model_evaluation.md for the full writeup, but states the
     headline result here (whichever combination_method actually won,
     from Phase 2's real numbers)

2. docs/model_evaluation.md — this should be an honest, specific writeup
   of the actual evaluation, not generic ML documentation boilerplate.
   Structure it as:
   - Setup: two backbones (EfficientNet, MobileNet), trained
     independently, 10-class fundus image classification, 5,000 images
   - Solo results: both models' individual test accuracy/macro-F1
     (pull the real numbers from eval_baseline_results.json)
   - Ensemble results: simple-average and weighted-average results
     (pull real numbers from ensemble_results.json), and clearly state
     whether either ensemble variant beat the better solo model
   - If it did NOT beat the better solo model: explain why, using the
     actual confusion matrix evidence — specifically, that both models
     share the same Glaucoma-confused-with-Healthy/Myopia failure
     direction, meaning there were no complementary errors for
     ensembling to average out. State plainly that this is why
     ensembling didn't help here, and that [whichever model/method
     actually won] was what got shipped, with the reasoning.
   - If it DID beat the better solo model: report the specific
     macro-F1 improvement and which classes benefited most, with
     numbers, not just "the ensemble performed better"
   - A short note on per-class reliability: list which classes had
     fewer than 25 test samples (from ensemble_config.json's
     low_support_classes) and state clearly that their per-class
     metrics should be treated with lower confidence
   - ONNX export verification: state that both models were verified to
     produce numerically matching outputs after export (reference
     Phase 1's verification results)

Do not write generic statements like "the model performs well" anywhere
in model_evaluation.md — every claim should be backed by a specific
number pulled from the actual JSON result files. Show me the real numbers
you pulled before finalizing the document, so I can confirm they're
accurate.
```

**Verification gate:** You have checked that every number in `model_evaluation.md` traces back to a real value in your actual `eval_baseline_results.json` / `ensemble_results.json` files — not a plausible-sounding placeholder the agent generated.

---

## After Phase 7: Resume Bullets

Once Phase 2's real winner is confirmed and Phase 7's documentation is written, fill in your resume bullets using the actual outcome:

**AI Engineer:**
> Built and evaluated an ensemble of EfficientNet and MobileNet for 10-class eye disease classification on 5,000 fundus images; designed a macro-F1-weighted combination strategy, rigorously evaluated it against both solo models, and found [your real result] due to correlated failure modes on Glaucoma; exported both models to ONNX with numerically verified parity against the original PyTorch evaluation.

**SDE:**
> Designed and containerized a FastAPI inference service serving a dual-model ONNX ensemble, including a startup-time class-mapping consistency check to prevent silent misprediction from model mismatch, and per-class reliability flagging surfaced directly through the API response; ported Grad-CAM explainability across two distinct CNN architectures.

Use your actual Phase 2 result in the bracket — whichever it turned out to be, it's the more defensible thing to say in an interview than a guessed number.
