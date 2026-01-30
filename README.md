# Zonkey LLM

![Zonkey](zonkey_image.png)

**Zonkey: A Hierarchical Diffusion Language Model with Differentiable Tokenization and Probabilistic Attention**

Zonkey is a fully differentiable hierarchical diffusion language model that learns adaptive, probabilistic tokenization directly from raw characters. It replaces fixed, non-differentiable tokenizers (e.g., BPE) with a trainable **Segment Splitter** that emerges linguistically meaningful boundaries (words, sentences) without explicit supervision. The model uses **Probabilistic Attention** to handle variable-length sequences softly, a multi-vector **Compressor**, a **Denoising Diffusion Mixed Model (DDMM)** for stable latent-space denoising, and a differentiable **Stitcher** for overlap-invariant reassembly.

Trained end-to-end on Wikipedia, Zonkey generates coherent text with emergent word- and sentence-level hierarchies.


### Repository Status
This is a research prototype / proof-of-concept implementation. It demonstrates coherent sentence-level generation with emergent hierarchies (levels 0–1). Scaling to deeper hierarchies or larger datasets is left for future work.

### Installation

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains pinned versions of all dependencies (PyTorch 2.8.0, PyTorch Lightning 2.5.2, Hugging Face libraries, etc.).

### Data
The model uses character-level Wikipedia data via the provided `data/wiki_chars.py` dataloader (streams preprocessed Wikipedia text). No additional download is required for basic runs.

### Training
To start training:

```bash
python3 scripts/train.py --param_file configs/single_gpu_config.json
```

Other configurations are available in the `configs/` directory.

To resume from a checkpoint:
```bash
python3 scripts/train.py --param_file configs/single_gpu_config.json --resume path/to/checkpoint.ckpt
```

Training logs and checkpoints are saved automatically. TensorBoard is launched on port 6006.

### Generation
Basic unconditional generation examples are printed during training for debugging/monitoring. Dedicated generation scripts will be added in future updates.

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Contact / Issues
For questions or issues, please open a GitHub issue or contact alonzorz1@gmail.com.

Enjoy experimenting with differentiable tokenization and DDMM diffusion!