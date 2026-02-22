# Generative AI: Convolutional VAE on Fashion-MNIST

A generative modeling experiment using a Convolutional Variational Autoencoder (VAE) to learn the latent structure of the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset and generate novel fashion item images. The model is evaluated through reconstruction quality, random generation, latent space visualization, interpolation, and failure analysis.

| | |
|---|---|
| **Author** | Tim Wilcoxson |
| **Course** | Project 5 — Generative AI |
| **Date** | February 2026 |

## Key Findings

- **Task:** Unsupervised generative modeling with a Convolutional VAE
- **Dataset:** Fashion-MNIST — 70,000 28x28 grayscale images, 10 clothing categories
- **Framework:** PyTorch with MPS (Apple Silicon GPU) acceleration
- **Latent space (32-dim) captures meaningful class structure**, with smooth interpolation between categories
- **Reconstructions preserve overall shape and category identity**, though fine texture details are lost (characteristic VAE blurriness)

## Project Structure

```
project5_generative_ai/
├── generative_model.ipynb                           # Complete VAE workflow notebook
├── generate_report.py                               # Script to regenerate the PDF report
├── module_summary.pdf                               # Generative AI analysis report (PDF)
├── Generative_AI_Analysis_Report.pdf                # Generative AI analysis report (identical copy)
├── requirements.txt                                 # Python dependencies (pip freeze)
├── README.md                                        # This file
├── .gitignore
├── data/                                            # Fashion-MNIST auto-downloaded by torchvision
└── figures/
    ├── fig1_sample_images.png                       # Fashion-MNIST sample images (one per class)
    ├── fig2_training_loss.png                       # Training & validation loss curves (3 panels)
    ├── fig3_reconstruction_comparison.png            # Original vs. reconstructed images
    ├── fig4_generated_samples.png                   # Random samples from latent space
    ├── fig5_latent_space_tsne.png                   # t-SNE visualization of latent space
    ├── fig6_latent_interpolation.png                # Interpolation between class pairs
    ├── fig7_dimension_traversal.png                 # Latent dimension traversal
    ├── fig8_class_generation.png                    # Class-conditional prototype generation
    └── fig9_failure_analysis.png                    # Best and worst reconstructions
```

## Setup and Reproduction

```bash
git clone https://github.com/trwilcoxson/udacity-nd608-project5-generative-ai.git
cd project5_generative_ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the notebook (trains VAE, ~3-5 min on Apple Silicon MPS)
jupyter notebook generative_model.ipynb

# Regenerate the PDF report
python generate_report.py
```

## Technologies

- **Python 3.14** — PyTorch, Torchvision, NumPy, Pandas, Matplotlib, Seaborn
- **Generative modeling** — Convolutional VAE, reparameterization trick, beta-VAE loss
- **Evaluation** — Reconstruction quality, latent space t-SNE, interpolation, dimension traversal
- **Report generation** — fpdf2
- **Environment** — Jupyter Notebook, venv, MPS (Apple Silicon GPU)

## Dataset

Fashion-MNIST (Xiao et al., 2017): 70,000 28x28 grayscale images in 10 classes (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot). Split into 60,000 training and 10,000 test images. Auto-downloaded by `torchvision.datasets.FashionMNIST`.

## References

- Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. *Proceedings of the 2nd ICLR*.
- Higgins, I., et al. (2017). beta-VAE: Learning basic visual concepts with a constrained variational framework. *Proceedings of the 5th ICLR*.
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. *arXiv preprint arXiv:1708.07747*.
- Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *NeurIPS*, 32, 8024-8035.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. https://www.deeplearningbook.org/
