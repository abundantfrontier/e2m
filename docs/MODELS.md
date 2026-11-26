# Models

Adapters (LoRA / QLoRA) are not stored in git. Weights are on Hugging Face at [AbundantFrontierInstitute/e2m](https://huggingface.co/AbundantFrontierInstitute/e2m).
Each adapter applies to the listed base model from Hugging Face (`Qwen/<base>`).

| Adapter | Base model | Round | Description | Adapter size | Config | Weights |
|---|---|---|---|---:|---|---|
| lora1 | [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) | 1 | First LoRA: English to AT&T assembly, 2,910 scaled tasks. | 151.8 MB | [configs/lora1.json](../configs/lora1.json) | [lora1](https://huggingface.co/AbundantFrontierInstitute/e2m/tree/main/lora1) |
| qlora7b | [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) | 1 | 4-bit QLoRA: English to assembly, same 2,910 tasks as lora1. | 319.0 MB | [configs/qlora7b.json](../configs/qlora7b.json) | [qlora7b](https://huggingface.co/AbundantFrontierInstitute/e2m/tree/main/qlora7b) |
| lora2 | [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) | 1 | LoRA on the 6,781-task mix (rejection-sampled loop programs plus rewritten specs). | 151.8 MB | [configs/lora2.json](../configs/lora2.json) | [lora2](https://huggingface.co/AbundantFrontierInstitute/e2m/tree/main/lora2) |
| dpo | [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) | 2 | DPO from lora2 (beta 0.1); generation collapsed. Kept for the record. | 151.8 MB | [configs/dpo.json](../configs/dpo.json) | not published (collapsed) |
| qlora7b-v2 | [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) | 2b | 4-bit QLoRA on the 6,781-task mix, English to assembly. | 319.0 MB | [configs/qlora7b-v2.json](../configs/qlora7b-v2.json) | [qlora7b-v2](https://huggingface.co/AbundantFrontierInstitute/e2m/tree/main/qlora7b-v2) |
| dpo2 | [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) | 3 | DPO from lora2 with an RPO-style supervised term (beta 0.2, 1,500 steps). | 151.8 MB | [configs/dpo2.json](../configs/dpo2.json) | [dpo2](https://huggingface.co/AbundantFrontierInstitute/e2m/tree/main/dpo2) |
| c-then-asm | [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) | 3 | 4-bit QLoRA that writes C, then assembly for that C. | 319.0 MB | [configs/c-then-asm.json](../configs/c-then-asm.json) | [c-then-asm](https://huggingface.co/AbundantFrontierInstitute/e2m/tree/main/c-then-asm) |
| c-only-7b | [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) | 4 | 4-bit QLoRA that writes C only; the C model of the gcc hybrid. | 319.0 MB | [configs/c-only-7b.json](../configs/c-only-7b.json) | [c-only-7b](https://huggingface.co/AbundantFrontierInstitute/e2m/tree/main/c-only-7b) |
| asm-distill-7b | [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) | 5 | 4-bit QLoRA, English to assembly distilled from gcc lowerings of the model's own C. | 319.0 MB | [configs/asm-distill-7b.json](../configs/asm-distill-7b.json) | coming |

Training settings for every run are in `configs/`; scores are in `results/` and the README table.
