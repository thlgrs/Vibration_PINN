# Run on Google Colab (GPU)

Use notebook `notebooks/06_colab_exp01_gpu.ipynb` to run Experiment 01 on a GPU runtime.

## Steps

1. Push this repository to GitHub.
2. Open the notebook in Colab.
3. Set runtime to GPU: `Runtime -> Change runtime type -> Hardware accelerator = GPU`.
4. Edit `REPO_URL` in the notebook.
5. Run all cells in order.

## Notes

- The notebook installs `requirements-colab.txt` (no `torch`) so Colab keeps its preinstalled CUDA-enabled PyTorch build.
- For fast validation, run the smoke command first:

```bash
python experiments/exp01_clean_data.py --device cuda --phase1-epochs 30 --phase2-epochs 120 --hidden-features 64 --hidden-layers 3
```
