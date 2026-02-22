"""
Generate the PDF report for the Generative AI project.

Produces both 'module_summary.pdf' and 'Generative_AI_Analysis_Report.pdf'
(identical content) to satisfy rubric criteria that reference each filename.

Usage:
    python generate_report.py
"""

import shutil
from fpdf import FPDF

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_DIR = "."
FIGURES_DIR = f"{PROJECT_DIR}/figures"
OUTPUT_PRIMARY = f"{PROJECT_DIR}/module_summary.pdf"
OUTPUT_COPY = f"{PROJECT_DIR}/Generative_AI_Analysis_Report.pdf"

TITLE = "Generative AI: Convolutional VAE on Fashion-MNIST"
AUTHOR = "Tim Wilcoxson"
DATE = "February 2026"
COURSE = "Project 5 -- Generative AI"
DATASET = "Fashion-MNIST (Xiao et al., 2017)"

# Page geometry
PAGE_W = 210  # A4 width in mm
MARGIN = 20
CONTENT_W = PAGE_W - 2 * MARGIN

# Fonts
FONT_BODY = ("Helvetica", "", 11)
FONT_BOLD = ("Helvetica", "B", 11)
FONT_H2 = ("Helvetica", "B", 14)
FONT_H3 = ("Helvetica", "B", 12)
FONT_SMALL = ("Helvetica", "", 9)
FONT_ITALIC = ("Helvetica", "I", 10)

# ---------------------------------------------------------------------------
# Metrics from notebook execution (updated after training)
# ---------------------------------------------------------------------------
TOTAL_PARAMS = "384,577"
LATENT_DIM = "32"
NUM_EPOCHS = "50"
TRAINING_TIME = "210.7s (3.5 min)"
BEST_VAL_LOSS = "237.74"
TEST_RMSE = "0.1170"
KL_FINAL = "16.8"
RECON_FINAL = "220.9"


# ---------------------------------------------------------------------------
# Report PDF class
# ---------------------------------------------------------------------------
class ReportPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font(*FONT_SMALL)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, TITLE, align="L")
        self.ln(6)
        self.set_draw_color(180, 180, 180)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font(*FONT_SMALL)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    # ---- Helpers ----------------------------------------------------------

    def section_heading(self, number, title):
        self.ln(4)
        self.set_font(*FONT_H2)
        self.set_text_color(30, 60, 120)
        self.cell(0, 10, f"{number}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(30, 60, 120)
        self.line(MARGIN, self.get_y(), MARGIN + CONTENT_W, self.get_y())
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def subsection(self, title):
        self.ln(2)
        self.set_font(*FONT_H3)
        self.set_text_color(50, 80, 140)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)
        self.set_text_color(0, 0, 0)

    def body_text(self, text):
        self.set_font(*FONT_BODY)
        self.multi_cell(CONTENT_W, 6, text)
        self.ln(2)

    def bold_text(self, text):
        self.set_font(*FONT_BOLD)
        self.multi_cell(CONTENT_W, 6, text)
        self.ln(1)

    def italic_text(self, text):
        self.set_font(*FONT_ITALIC)
        self.multi_cell(CONTENT_W, 5, text)
        self.ln(1)

    def add_figure(self, path, caption, width=CONTENT_W):
        est_h = width * 0.6 + 15
        if self.get_y() + est_h > 270:
            self.add_page()
        x = (PAGE_W - width) / 2
        self.image(path, x=x, w=width)
        self.ln(2)
        self.set_font(*FONT_ITALIC)
        self.set_text_color(80, 80, 80)
        self.multi_cell(CONTENT_W, 5, caption, align="C")
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def bullet(self, text):
        self.set_font(*FONT_BODY)
        self.cell(6, 6, "-")
        self.multi_cell(CONTENT_W - 6, 6, text)
        self.ln(1)


# ---------------------------------------------------------------------------
# Build the report
# ---------------------------------------------------------------------------
def build_report():
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(MARGIN, MARGIN, MARGIN)

    # =======================================================================
    # TITLE PAGE
    # =======================================================================
    pdf.add_page()
    pdf.ln(50)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(30, 60, 120)
    pdf.multi_cell(CONTENT_W, 12, TITLE, align="C")
    pdf.ln(10)
    pdf.set_draw_color(30, 60, 120)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(CONTENT_W, 8, AUTHOR, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(CONTENT_W, 8, DATE, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(CONTENT_W, 8, COURSE, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 11)
    pdf.cell(CONTENT_W, 8, f"Dataset: {DATASET}", align="C",
             new_x="LMARGIN", new_y="NEXT")

    # =======================================================================
    # 1. REPORT OVERVIEW
    # =======================================================================
    pdf.add_page()
    pdf.section_heading(1, "Report Overview")
    pdf.body_text(
        "This report presents a generative modeling experiment using a "
        "Convolutional Variational Autoencoder (VAE) on the Fashion-MNIST "
        "dataset. The VAE learns a compressed latent representation of "
        "28x28 grayscale fashion item images and generates novel samples "
        "by decoding points from the learned latent space. The model is "
        "implemented in PyTorch (Paszke et al., 2019) and trained on "
        "Apple Silicon MPS hardware."
    )
    pdf.body_text(
        f"The architecture contains {TOTAL_PARAMS} trainable parameters "
        f"with a {LATENT_DIM}-dimensional latent space. Training completed "
        f"in {NUM_EPOCHS} epochs ({TRAINING_TIME}) with a best validation "
        f"loss of {BEST_VAL_LOSS}. The model is assessed through "
        "reconstruction quality, random sample generation, latent space "
        "visualization, interpolation between classes, latent dimension "
        "traversal, class-conditional generation, and failure analysis."
    )

    # =======================================================================
    # 2. DATASET DESCRIPTION
    # =======================================================================
    pdf.section_heading(2, "Dataset Description")
    pdf.body_text(
        "Fashion-MNIST (Xiao et al., 2017) is a dataset of Zalando's "
        "article images consisting of 70,000 28x28 grayscale images in "
        "10 categories: T-shirt/top, Trouser, Pullover, Dress, Coat, "
        "Sandal, Shirt, Sneaker, Bag, and Ankle boot. The dataset is "
        "split into 60,000 training images and 10,000 test images, with "
        "exactly 7,000 images per class in the full dataset."
    )
    pdf.body_text(
        "For this experiment, the 60,000 training images were further "
        "split into 54,000 for training and 6,000 for validation using "
        "a fixed random seed (42) for reproducibility. No data "
        "augmentation or normalization was applied, as the VAE decoder "
        "uses a Sigmoid activation to output values in [0, 1], matching "
        "the raw pixel range from ToTensor()."
    )

    pdf.add_figure(
        f"{FIGURES_DIR}/fig1_sample_images.png",
        "Figure 1. Representative Fashion-MNIST images, one per class, "
        "showing the 28x28 resolution and visual diversity of the dataset.",
        width=CONTENT_W - 10,
    )

    # =======================================================================
    # 3. MODEL DESIGN AND TRAINING APPROACH
    # =======================================================================
    pdf.section_heading(3, "Model Design and Training Approach")

    pdf.subsection("Why a VAE?")
    pdf.body_text(
        "Several generative model families exist -- GANs, VAEs, diffusion "
        "models, autoregressive models, and normalizing flows. We chose a "
        "Convolutional VAE for this project because: (1) VAEs provide a "
        "principled probabilistic framework with a well-defined loss function "
        "(the evidence lower bound, or ELBO), unlike GANs which require "
        "delicate adversarial balancing and suffer from mode collapse; "
        "(2) the learned latent space supports rich downstream analysis -- "
        "interpolation, dimension traversal, class-conditional generation, "
        "and t-SNE visualization -- that would be difficult or impossible "
        "with other architectures; (3) VAEs train stably and reproducibly "
        "with standard gradient descent, making them ideal for educational "
        "and experimental settings; and (4) the architecture scales well to "
        "Fashion-MNIST's 28x28 resolution while training in minutes on "
        "consumer hardware."
    )

    pdf.subsection("VAE Architecture")
    pdf.body_text(
        "The Convolutional VAE consists of a symmetric encoder-decoder "
        "architecture connected through a stochastic latent bottleneck. "
        "The encoder uses three convolutional blocks (Conv2d with stride 2, "
        "BatchNorm2d, LeakyReLU with slope 0.2) to progressively "
        "downsample the input from 28x28 to 4x4, increasing channel "
        "depth from 1 to 128. The flattened features are projected to "
        "two vectors: the mean (mu) and log-variance (log_var) of the "
        "approximate posterior distribution in the 32-dimensional latent "
        "space."
    )
    pdf.body_text(
        "The reparameterization trick (Kingma & Welling, 2014) enables "
        "gradient flow through the stochastic sampling step by computing "
        "z = mu + sigma * epsilon, where epsilon is drawn from N(0, I). "
        "The decoder mirrors the encoder using transposed convolutions "
        "(ConvTranspose2d, BatchNorm2d, ReLU) to upsample from 4x4 back "
        "to 28x28, with a final Sigmoid activation to constrain outputs "
        "to [0, 1]."
    )

    pdf.subsection("Loss Function")
    pdf.body_text(
        "The VAE loss combines two terms: (1) binary cross-entropy (BCE) "
        "reconstruction loss, summed over pixels and averaged over the "
        "batch, which measures how faithfully the decoder reproduces the "
        "input; and (2) KL divergence between the learned posterior "
        "q(z|x) and the standard normal prior N(0, I), which regularizes "
        "the latent space (Kingma & Welling, 2014). The KL term is "
        "weighted by beta=1.0 (Higgins et al., 2017), balancing "
        "reconstruction quality against latent space smoothness."
    )

    pdf.subsection("Training Configuration")
    pdf.body_text(
        "The model was trained with Adam optimizer (learning rate 1e-3) "
        f"for {NUM_EPOCHS} epochs with early stopping (patience=10). "
        f"Training completed in {TRAINING_TIME} on Apple Silicon MPS "
        f"with batch size 128. The best validation loss was "
        f"{BEST_VAL_LOSS} (reconstruction: {RECON_FINAL}, KL divergence: "
        f"{KL_FINAL}). The non-zero KL term "
        "confirms the model avoids posterior collapse and actively uses "
        "the latent space."
    )

    pdf.add_figure(
        f"{FIGURES_DIR}/fig2_training_loss.png",
        "Figure 2. Training and validation loss curves: total loss (left), "
        "reconstruction loss (center), and KL divergence (right). Both "
        "losses converge smoothly with minimal train-val gap.",
        width=CONTENT_W - 10,
    )

    # =======================================================================
    # 4. OUTPUT EVALUATION AND INTERPRETATION
    # =======================================================================
    pdf.section_heading(4, "Output Assessment and Interpretation")

    pdf.subsection("Reconstruction Quality")
    pdf.body_text(
        f"The model achieves a test set per-pixel RMSE of {TEST_RMSE}, "
        "indicating that reconstructions are close to the originals in "
        "aggregate. Visual inspection (Figure 3) confirms that the VAE "
        "preserves overall shape, silhouette, and category identity in "
        "reconstructions. However, fine-grained details (textures, "
        "patterns, sharp edges) are smoothed out -- a well-known "
        "limitation of VAEs with pixel-wise reconstruction losses "
        "(Kingma & Welling, 2014)."
    )

    pdf.add_figure(
        f"{FIGURES_DIR}/fig3_reconstruction_comparison.png",
        "Figure 3. Original test images (top) vs. VAE reconstructions "
        "(bottom). Category identity is preserved, with characteristic "
        "VAE blurriness on fine details.",
        width=CONTENT_W - 10,
    )

    pdf.subsection("Random Sample Generation")
    pdf.body_text(
        "Sampling from the prior z ~ N(0, I) and decoding produces novel "
        "fashion item images (Figure 4). The generated samples show "
        "diverse, identifiable clothing shapes across multiple categories, "
        "confirming that the learned latent space generalizes beyond the "
        "training data. Some samples are ambiguous or blend features from "
        "multiple classes, which is expected when sampling from regions "
        "between class clusters in the latent space."
    )

    pdf.add_figure(
        f"{FIGURES_DIR}/fig4_generated_samples.png",
        "Figure 4. 100 random samples generated by decoding z ~ N(0, I). "
        "The samples show diverse fashion items with recognizable shapes.",
        width=CONTENT_W - 20,
    )

    pdf.subsection("Latent Space Visualization")
    pdf.body_text(
        "The t-SNE projection of the 32-dimensional latent means (Figure 5) "
        "reveals meaningful class structure. Visually distinct categories "
        "(Trouser, Bag, Sandal) form tight, well-separated clusters, while "
        "visually similar categories (Pullover/Coat, T-shirt/Shirt) show "
        "more overlap. This organization emerges unsupervised -- the VAE "
        "was never given class labels during training."
    )

    pdf.add_figure(
        f"{FIGURES_DIR}/fig5_latent_space_tsne.png",
        "Figure 5. t-SNE visualization of VAE latent space (mu vectors), "
        "colored by class label. Clear class clustering emerges despite "
        "fully unsupervised training.",
        width=CONTENT_W - 20,
    )

    pdf.subsection("Latent Space Interpolation")
    pdf.body_text(
        "Linear interpolation between latent representations of different "
        "classes (Figure 6) produces smooth, gradual transitions. For "
        "example, interpolating from a Sneaker to an Ankle boot shows "
        "the shoe gradually gaining height. These smooth transitions "
        "confirm that the latent space is continuous and well-organized, "
        "without abrupt discontinuities between classes."
    )

    pdf.add_figure(
        f"{FIGURES_DIR}/fig6_latent_interpolation.png",
        "Figure 6. Latent space interpolation between four class pairs, "
        "showing smooth transitions through 10 equally-spaced steps.",
        width=CONTENT_W - 10,
    )

    pdf.subsection("Latent Dimension Traversal")
    pdf.body_text(
        "Traversing individual latent dimensions from -3 to +3 (Figure 7) "
        "reveals that different dimensions capture different visual "
        "attributes. Some dimensions control overall shape (e.g., sleeve "
        "length, item width), while others affect texture or category "
        "identity. The top 8 highest-variance dimensions capture the most "
        "meaningful variation in the data."
    )

    pdf.add_figure(
        f"{FIGURES_DIR}/fig7_dimension_traversal.png",
        "Figure 7. Traversal of the top 8 highest-variance latent "
        "dimensions from -3 to +3, starting from the mean latent vector.",
        width=CONTENT_W - 10,
    )

    pdf.subsection("Class-Conditional Generation")
    pdf.body_text(
        "Decoding the per-class mean latent vectors produces class "
        "prototypes (Figure 8) -- the 'average' item in each category "
        "as understood by the model. These prototypes are recognizable "
        "and distinct, confirming that the latent space has learned "
        "meaningful class-specific regions."
    )

    pdf.add_figure(
        f"{FIGURES_DIR}/fig8_class_generation.png",
        "Figure 8. Class prototypes generated by decoding the per-class "
        "mean latent vectors. Each image represents the model's 'average' "
        "understanding of that category.",
        width=CONTENT_W - 10,
    )

    pdf.subsection("Failure Analysis")
    pdf.body_text(
        "The best-reconstructed images (Figure 9, top rows) tend to be "
        "simple, high-contrast items with uniform shapes (e.g., trousers, "
        "sneakers), with per-image BCE losses as low as ~90-100. The worst "
        "reconstructions (bottom rows, BCE losses exceeding 400) involve "
        "items with fine details, unusual poses, or ambiguous category "
        "membership."
    )
    pdf.body_text(
        "Per-class average reconstruction loss (BCE, summed over pixels) "
        "on the test set reveals a nearly 2x spread: Pullover (286.6), "
        "Shirt (281.9), Bag (273.7), T-shirt/top (258.7), Coat (258.0), "
        "Ankle boot (207.4), Dress (189.6), Sandal (161.2), Sneaker "
        "(153.8), Trouser (152.2). Visually simple, structurally consistent "
        "classes (Trouser, Sneaker) achieve the lowest reconstruction "
        "error, while highly variable classes with complex internal "
        "structure (Pullover, Shirt) are the hardest to reconstruct. "
        "This pattern suggests the VAE allocates latent capacity toward "
        "capturing broad shape categories rather than fine within-class "
        "variation."
    )

    pdf.add_figure(
        f"{FIGURES_DIR}/fig9_failure_analysis.png",
        "Figure 9. Best (top two rows) and worst (bottom two rows) "
        "reconstructions by per-image BCE loss. Numbers above images "
        "indicate reconstruction loss.",
        width=CONTENT_W - 10,
    )

    # =======================================================================
    # 5. ETHICAL CONSIDERATIONS AND RESPONSIBLE USE
    # =======================================================================
    pdf.section_heading(5, "Ethical Considerations and Responsible Use")
    pdf.body_text(
        "While Fashion-MNIST is a benign academic benchmark, the "
        "generative modeling techniques demonstrated here have broader "
        "ethical implications when applied to real-world fashion data. "
        "This section examines specific ethical concerns, how design "
        "choices in this project relate to each, and concrete mitigations "
        "for responsible deployment."
    )

    pdf.subsection("Cultural Representation Bias")
    pdf.body_text(
        "Fashion-MNIST reflects a narrow, Western-centric view of clothing. "
        "The 10 categories (T-shirt, Trouser, Dress, etc.) map primarily "
        "to Western fashion norms and exclude culturally significant "
        "garments such as saris, hanbok, dashiki, kimono, and many other "
        "traditional clothing forms. A generative model trained on this "
        "data can only produce outputs within this limited cultural scope, "
        "risking the reinforcement of Western fashion as the default or "
        "norm."
    )
    pdf.italic_text(
        "Mitigation: Future work should train on culturally diverse "
        "datasets with balanced representation across global fashion "
        "traditions, and audit generated outputs for cultural coverage gaps."
    )

    pdf.subsection("Differential Output Quality Across Categories")
    pdf.body_text(
        "Our per-class analysis reveals that the model reconstructs some "
        "categories significantly better than others (Trouser: 152.2 BCE "
        "vs. Pullover: 286.6 BCE -- a nearly 2x gap). If deployed at "
        "scale, such differential quality could systematically disadvantage "
        "certain product categories, creating biased representations in "
        "e-commerce or design tools. Categories with higher visual "
        "complexity receive lower-fidelity outputs, which could lead to "
        "unfair treatment in downstream applications."
    )
    pdf.italic_text(
        "Mitigation: Audit per-class generation quality before deployment "
        "and consider class-weighted loss functions or specialized "
        "sub-models for underperforming categories."
    )

    pdf.subsection("Counterfeit and Misuse Potential")
    pdf.body_text(
        "More capable generative models trained on higher-resolution "
        "fashion data could be misused to produce counterfeit product "
        "images for fraudulent marketing, fake e-commerce listings, or "
        "intellectual property theft. As generative quality improves, "
        "the barrier to creating convincing fake product imagery decreases, "
        "raising concerns about consumer deception and brand integrity."
    )
    pdf.body_text(
        "Design choices in this project inherently limit misuse risk: "
        "the 28x28 grayscale resolution, inherent VAE blurriness, and "
        "beta=1.0 (which prioritizes latent regularity over pixel-perfect "
        "reconstruction) all ensure that outputs cannot pass as real "
        "product photography. However, the techniques demonstrated here "
        "are directly transferable to higher-resolution settings."
    )
    pdf.italic_text(
        "Mitigation: Watermark generated images, restrict access to "
        "high-resolution models, and implement provenance tracking for "
        "AI-generated content."
    )

    pdf.subsection("Creative Ownership")
    pdf.body_text(
        "The question of who owns AI-generated fashion designs remains "
        "legally and ethically unresolved. If a VAE trained on existing "
        "designs produces novel outputs, are those outputs derivative "
        "works? Can they be copyrighted? These questions become practical "
        "concerns as generative models are increasingly used in the "
        "fashion industry for design ideation and rapid prototyping."
    )
    pdf.italic_text(
        "Mitigation: Clearly document training data provenance, disclose "
        "AI involvement in design pipelines, and establish licensing "
        "frameworks for AI-assisted creative outputs."
    )

    pdf.subsection("Environmental Cost")
    pdf.body_text(
        "Training generative models, particularly at scale, requires "
        "significant computational resources and associated energy "
        "consumption. While the model in this project is small (384,577 "
        "parameters, 3.5 min training), production-scale generative models "
        "(e.g., diffusion models, large GANs) can require orders of "
        "magnitude more compute. Responsible AI development should consider "
        "the environmental footprint of model training and deployment."
    )
    pdf.italic_text(
        "Mitigation: Track carbon footprint of training runs, use "
        "efficient architectures and early stopping to minimize unnecessary "
        "computation, prefer fine-tuning over training from scratch, and "
        "document energy consumption in model cards."
    )

    pdf.subsection("Positive Applications")
    pdf.body_text(
        "Generative models like VAEs also enable responsible innovation: "
        "rapid prototyping for sustainable fashion design (reducing "
        "physical sample waste), accessibility tools for visually impaired "
        "designers, personalized virtual try-on experiences, and "
        "educational tools for understanding visual similarity and style. "
        "The latent space structure demonstrated in this project -- smooth "
        "interpolation, class prototypes, dimension traversal -- provides "
        "interpretable, controllable generation that supports human "
        "creativity rather than replacing it."
    )

    # =======================================================================
    # 6. LIMITATIONS AND FUTURE IMPROVEMENTS
    # =======================================================================
    pdf.section_heading(6, "Limitations and Future Improvements")

    pdf.subsection("Current Limitations")
    pdf.bullet(
        "VAE blurriness: The pixel-wise BCE reconstruction loss averages "
        "over plausible outputs, producing blurry reconstructions that "
        "lack sharp edges and fine textures. This is a fundamental "
        "limitation of VAEs with simple reconstruction objectives "
        "(Kingma & Welling, 2014)."
    )
    pdf.bullet(
        "Low resolution: Fashion-MNIST images are only 28x28 grayscale, "
        "far below the resolution needed for practical fashion design "
        "applications. The architectural choices (e.g., three conv layers, "
        "32-dim latent space) are tailored to this small scale."
    )
    pdf.bullet(
        "No conditional generation: The model is trained without class "
        "labels, so it cannot generate items of a specific category on "
        "demand. Class-conditional generation is approximated by decoding "
        "per-class mean vectors, but true conditional VAEs would allow "
        "more precise control."
    )
    pdf.bullet(
        "Single beta value: The experiment uses beta=1.0 (standard VAE). "
        "Systematic tuning of beta (Higgins et al., 2017) could improve "
        "the trade-off between reconstruction quality and latent space "
        "disentanglement."
    )

    pdf.subsection("Future Improvements")
    pdf.bullet(
        "Perceptual or adversarial losses: Replacing or supplementing "
        "BCE with a perceptual loss (feature matching) or adversarial "
        "loss (VAE-GAN hybrid) could produce sharper, more realistic "
        "outputs (Goodfellow et al., 2016)."
    )
    pdf.bullet(
        "Architecture upgrades: ResNet-style skip connections, attention "
        "mechanisms, or VQ-VAE (discrete latent spaces) could improve "
        "representation quality and generation fidelity."
    )
    pdf.bullet(
        "Conditional VAE (CVAE): Incorporating class labels as conditioning "
        "information would enable targeted generation of specific clothing "
        "categories with finer control."
    )
    pdf.bullet(
        "Higher-resolution datasets: Applying the same approach to "
        "DeepFashion or similar high-resolution datasets would test "
        "scalability and practical applicability."
    )

    # =======================================================================
    # 7. REFERENCES
    # =======================================================================
    pdf.section_heading(7, "References")
    pdf.set_font(*FONT_BODY)

    references = [
        (
            "Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep "
            "Learning. MIT Press. https://www.deeplearningbook.org/"
        ),
        (
            "Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., "
            "Botvinick, M., Mohamed, S., & Lerchner, A. (2017). beta-VAE: "
            "Learning Basic Visual Concepts with a Constrained Variational "
            "Framework. Proceedings of the 5th International Conference on "
            "Learning Representations (ICLR)."
        ),
        (
            "Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational "
            "Bayes. Proceedings of the 2nd International Conference on "
            "Learning Representations (ICLR)."
        ),
        (
            "Paszke, A., et al. (2019). PyTorch: An Imperative Style, "
            "High-Performance Deep Learning Library. Advances in Neural "
            "Information Processing Systems (NeurIPS), 32, 8024-8035."
        ),
        (
            "Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: "
            "A Novel Image Dataset for Benchmarking Machine Learning "
            "Algorithms. arXiv preprint arXiv:1708.07747."
        ),
    ]

    for ref in references:
        pdf.multi_cell(CONTENT_W, 5.5, ref)
        pdf.ln(3)

    # =======================================================================
    # OUTPUT
    # =======================================================================
    pdf.output(OUTPUT_PRIMARY)
    shutil.copy2(OUTPUT_PRIMARY, OUTPUT_COPY)
    print(f"Generated: {OUTPUT_PRIMARY}")
    print(f"Copied to: {OUTPUT_COPY}")


if __name__ == "__main__":
    build_report()
