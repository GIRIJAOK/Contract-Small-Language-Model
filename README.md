# ContractIQ-SLM

**Track B — SLM Fine-Tuning**  
**Scenario S2 — Gen AI for Enterprise Documents**

ContractIQ-SLM is a domain-adapted small language model for **contract clause verification and grounded evidence extraction**.

Given a target legal clause, short clause guidance, and a contract passage, the model predicts whether the clause is present and returns the supporting evidence in structured JSON.

```json
{
  "present": true,
  "evidence": "supporting text from the contract"
}
```

If the clause is not present:

```json
{
  "present": false,
  "evidence": null
}
```

## Walkthrough Video

[Watch the 5-minute technical walkthrough](https://drive.google.com/file/d/17OomiKY2zdaZ3eOa4TKz0DppcP-ZEjGO/view?usp=drive_link)

---

## 1. Problem Definition

Commercial contracts contain important clauses written in many different ways.

A general instruction model can understand the text but may still miss a clause when the wording is indirect or domain-specific.

The goal of this project is to improve a compact instruction model for two tasks:

- decide whether a target legal clause is present
- return supporting evidence directly from the supplied contract passage

The main question is:

> Can a compact 7B instruction model be adapted to improve contract clause detection and evidence extraction while keeping the output grounded in the source text?

---

## 2. Why Fine-Tuning?

I first evaluated the original **Qwen2.5-7B-Instruct** model as the baseline.

| Metric | Baseline Qwen |
|---|---:|
| Accuracy | 0.8289 |
| Precision | 0.9429 |
| Recall | 0.3976 |
| F1 | 0.5593 |

The baseline model had high precision, but recall was low. It was generally careful when predicting a clause, but it missed many clauses that were actually present.

This showed that prompting alone was not enough for the target task.

RAG was also considered. However, in this experiment the relevant contract passage is already supplied to the model, so retrieval is not the main problem.

Fine-tuning was therefore used to improve:

- legal clause understanding
- clause presence/absence decisions
- evidence extraction
- structured JSON output

---

## 3. Dataset and SFT Data Preparation

The project uses **CUAD v1 — Contract Understanding Atticus Dataset**.

CUAD contains:

- 510 commercial contracts
- 41 legal clause categories
- clause-level annotations
- supporting evidence spans

For this experiment, I selected eight clause categories:

1. Cap On Liability
2. Audit Rights
3. Termination For Convenience
4. Exclusivity
5. Renewal Term
6. Change Of Control
7. Uncapped Liability
8. Notice Period To Terminate Renewal

### Contract-Level Split

The dataset was split at the **contract level** so the same contract does not appear across training, validation, and test sets.

| Split | Contracts | Examples |
|---|---:|---:|
| Train | 357 | 2,856 |
| Validation | 76 | 608 |
| Test | 77 | 616 |

The test set was kept untouched until the final model configuration was fixed.

### SFT Example Format

Each supervised fine-tuning example contains:

1. target clause
2. short clause guidance
3. contract passage
4. expected JSON answer

For positive examples, the passage contains the annotated evidence.

For negative examples, another realistic legal passage from the same contract was used where possible. This makes the task harder because the model still sees legal language, but it must decide whether the **specific target clause** is present.

Training loss is calculated only on the assistant answer. The system and user prompt tokens are masked from the loss.

The maximum sequence length was set to **3,072 tokens** after checking the token-length distribution of the SFT examples.

---

## 4. Base Model Selection

The base model used in this project is:

**Qwen/Qwen2.5-7B-Instruct**

**Mistral-7B-Instruct** was also considered as an alternative.

Qwen was selected because it provides a good balance of:

- compact 7B model size
- instruction following
- structured JSON generation
- Hugging Face and PEFT support

The main experimental comparison is between the **original Qwen baseline** and the **DoRA fine-tuned Qwen model**.

---

## 5. Why DoRA?

I selected **DoRA — Weight-Decomposed Low-Rank Adaptation** for parameter-efficient fine-tuning.

DoRA adapts the model without updating all of the base-model parameters. It separates the weight update into magnitude and direction, allowing efficient adaptation while keeping the number of trainable parameters small.

For this project, the practical advantages were:

- only about **0.28%** of the model parameters were trainable
- the full 7B model did not need full fine-tuning
- the adapter can be merged with the base model for inference
- the method is suitable for focused domain adaptation

### Why DoRA with BF16 Instead of QLoRA?

QLoRA is useful when GPU memory is limited because the base model is loaded in 4-bit quantized form.

For this experiment, an **NVIDIA A100 80GB GPU** was available, so the additional memory saving was not necessary.

I therefore used DoRA with BF16 because it:

- avoided 4-bit quantization during training
- kept the training setup simpler
- avoided adding quantization as another experimental variable
- still kept the number of trainable parameters very small

QLoRA would be a practical option for the same experiment on a smaller GPU.

---

## 6. Training Configuration

The final model was trained with the following configuration:

| Parameter | Value |
|---|---|
| Base model | Qwen2.5-7B-Instruct |
| Fine-tuning method | DoRA |
| Rank | 8 |
| Alpha | 16 |
| Dropout | 0.05 |
| Precision | BF16 |
| Quantization | None |
| Max sequence length | 3072 |
| Learning rate | 1e-4 |
| Epochs | 2 |
| Batch size | 1 |
| Gradient accumulation | 8 |
| Effective batch size | 8 |
| Trainable parameters | 21,575,680 |
| Trainable percentage | 0.2825% |

DoRA was applied to the attention and MLP projection layers:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Training was performed on an **NVIDIA A100-SXM4-80GB GPU**.

Training summary:

- final training loss: **0.0198**
- validation loss: **0.0100**
- training time: approximately **3 hours 19 minutes**

---

## 7. Baseline vs Fine-Tuned Results

The baseline and fine-tuned model were evaluated on the same **608-example validation set**.

| Metric | Baseline Qwen | DoRA Fine-Tuned |
|---|---:|---:|
| Accuracy | 0.8289 | **0.9622** |
| Precision | **0.9429** | 0.9387 |
| Recall | 0.3976 | **0.9217** |
| F1 | 0.5593 | **0.9301** |
| Macro F1 | 0.7266 | **0.9521** |
| Evidence Token F1 | 0.2885 | **0.7751** |
| Grounded Evidence Rate | 0.9286 | **0.9939** |
| Unsupported Evidence Rate | 0.0714 | **0.0061** |
| Valid JSON Rate | 0.9770 | **1.0000** |

The largest improvement was in **recall**.

The baseline model missed many positive clauses. After fine-tuning, recall increased from **0.3976 to 0.9217**, while precision remained high.

---

## 8. Qualitative Example

### Target Clause

**Termination For Convenience**

### Supporting Evidence

```text
Either party may terminate this Agreement without cause at any time effective upon thirty (30) days' written notice.
```

### Baseline Qwen Output

```json
{
  "present": false,
  "evidence": null
}
```

The baseline model missed the clause.

### DoRA Fine-Tuned Output

```json
{
  "present": true,
  "evidence": "Either party may terminate this Agreement without cause at any time effective upon thirty (30) days' written notice."
}
```

The fine-tuned model detected the clause and returned the supporting contract text.

---

## 9. Final Test Results

After validation was complete, the final configuration was frozen and evaluated once on the untouched test set.

The test set contains **616 examples from 77 unseen contracts**.

| Metric | Test Result |
|---|---:|
| Accuracy | **0.9448** |
| Precision | **0.9451** |
| Recall | **0.8776** |
| F1 | **0.9101** |
| Macro F1 | **0.9351** |
| Evidence Token F1 | **0.7054** |
| Grounded Evidence Rate | **0.9834** |
| Unsupported Evidence Rate | **0.0166** |
| Valid JSON Rate | **0.9951** |
| Valid Schema Rate | **0.9935** |
| Average inference latency | **1.039 s/example** |

These results were obtained without further tuning on the test set.

---

## 10. Error Analysis

The final test set had:

```text
False negatives: 24
False positives: 10
```

The most difficult category was **Notice Period To Terminate Renewal**, with a recall of **0.625**.

The remaining errors mainly involved:

- indirect legal wording
- closely related legal concepts
- renewal and termination conditions
- longer evidence spans

Additional observations:

- 3 unsupported-evidence cases
- 3 invalid JSON outputs
- 4 schema-invalid outputs
- 3 outputs reached the 256-token generation limit

The model was not changed after reviewing the final test results.

---

## 11. General Capability Check

A small general-capability sanity check was used to look for obvious signs of catastrophic forgetting.

The test contained 15 prompts covering:

- arithmetic and reasoning
- factual QA
- instruction following
- structured JSON generation
- classification
- information extraction
- summarization

| Model | Passed |
|---|---:|
| Base Qwen | 15 / 15 |
| DoRA Fine-Tuned Model | 15 / 15 |
| Observed Regressions | **0** |

This is a limited diagnostic rather than a complete general-capability benchmark, but no obvious regression was observed on these prompts.

Detailed results are stored in:

```text
results/general_capability/general_capability_results.json
```

---

## 12. Evidence Grounding

Correct classification alone is not enough for this task.

When the model predicts that a clause is present, the returned evidence should also come from the supplied contract passage.

The final test model achieved a **98.34% grounded evidence rate**.

For a production system, both the JSON structure and the supporting evidence should be validated before the result is shown to the user.

---

## 13. Production Direction

A simple production flow could be:

```text
Contract / passage
      ↓
Backend API
      ↓
Fine-tuned Qwen model
      ↓
JSON and evidence validation
      ↓
Result shown to the user
```

For full contracts, a retrieval or passage-selection step would be added before the model.

For deployment, the merged model can also be evaluated with lower-cost inference options such as quantization or an optimized serving runtime.

---

## 14. Limitations and Next Steps

Current limitations:

- only 8 of the 41 CUAD clause categories were used
- the model receives a relevant passage instead of searching the full contract
- the general-capability check contains only 15 prompts
- the model has not yet been evaluated on an external contract dataset

With more time, I would:

- extend the experiment to more clause categories
- add full-document retrieval
- evaluate on an external contract dataset
- test lower-cost deployment options
- add a lightweight API and document-review interface

---

## 15. Environment

Repository target environment:

- **Python:** 3.12
- **Transformers:** 5.17.0
- **PEFT:** 0.21.0

Training was executed in a managed Google Colab environment using:

- **GPU:** NVIDIA A100-SXM4-80GB
- **Precision:** BF16
- **Colab runtime Python:** 3.13

Project dependencies are pinned in `requirements.txt`.

---

## 16. Reproducing the Main Pipeline

Install dependencies:

```bash
pip install -r requirements.txt
```

Build the SFT dataset:

```bash
python -m src.data.build_sft_dataset
```

Run the baseline:

```bash
python -m src.benchmark.run_qwen_baseline
```

Train the DoRA model:

```bash
python -m src.training.train_dora
```

Run validation inference:

```bash
python -m src.evaluation.run_dora_inference
```

Run final test inference:

```bash
python -m src.evaluation.run_dora_test
```

Run test error analysis:

```bash
python -m src.evaluation.analyze_test_errors
```

Run the general-capability check:

```bash
python -m src.evaluation.check_general_capability
```

---

## 17. Project Structure

```text
avathon-contractiq-slm/
├── configs/
├── data/
│   └── processed/
├── results/
│   ├── baselines/
│   ├── dataset_analysis/
│   ├── error_analysis/
│   ├── finetuned/
│   └── general_capability/
├── src/
│   ├── benchmark/
│   ├── data/
│   ├── evaluation/
│   └── training/
├── write-up/
│   └── ContractIQ_SLM_Technical_Writeup.pdf
├── .gitignore
├── README.md
└── requirements.txt
```

Large model checkpoints and adapter weights are intentionally not committed to Git.

The repository contains the code, evaluation results, analysis, and technical write-up needed to understand and reproduce the experiment.

---

## 18. Technical Write-Up

The technical write-up is available in:

```text
write-up/ContractIQ_SLM_Technical_Writeup.pdf
```

---

## Key Takeaway

The original Qwen baseline had high precision but low recall for contract clause detection.

After DoRA fine-tuning:

- validation recall improved from **0.3976 to 0.9217**
- validation F1 improved from **0.5593 to 0.9301**
- final test F1 was **0.9101**
- final grounded evidence rate was **0.9834**

The results show that a small parameter-efficient update can substantially improve domain-specific contract clause understanding while keeping the output strongly grounded in the supplied source passage.
