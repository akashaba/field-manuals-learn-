# Convolutional Neural Networks (CNNs) — Master Study Guide

> **Track:** Deep Learning · **Module:** 07
> **Prerequisites:** Modules 01–06.
> **Time budget:** ~12–15 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** CNNs were the workhorse that kicked off the deep-learning era. AlexNet's 2012 ImageNet win — CNNs by a huge margin — showed the world that neural networks could actually work. Even in the transformer age, CNNs dominate:

- Real-time inference on mobile / edge (they're small, fast).
- Medical imaging.
- Object detection and segmentation backbones.
- Any vision task where compute or latency matters.

More broadly, CNNs teach the concept of **inductive bias**: baking assumptions about the data (local structure, translation invariance) into the architecture. This principle underlies every modern architecture — the reason a transformer works well on language is because its attention mechanism has the right *inductive bias for tokens*. Understanding CNNs is understanding what inductive bias looks like in practice.

**Fundamental principles you must own:**

1. **Convolution is a linear operation** that shares parameters across spatial positions.
2. **Local receptive fields + weight sharing** are why CNNs are efficient — they encode the assumption that useful features are local and translationally invariant.
3. **Pooling downsamples spatial dimensions** while retaining important features.
4. **Depth builds a hierarchy of features** — edges → parts → objects.
5. **Residual connections** (ResNet) enable training very deep CNNs by giving gradients a shortcut path.
6. **Batch normalization + ReLU + max-pooling + residual connections** are the CNN's four horsemen.

If you retain nothing else: **a CNN is a stack of local, shared-weight linear operations followed by pointwise nonlinearities, engineered for translation-invariant hierarchies of features.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 The Convolution Operation

**Discrete 2D convolution** of an input $I$ with a filter (kernel) $K$:

$$O(i, j) = \sum_{m}\sum_{n} I(i + m, j + n) \cdot K(m, n)$$

(Machine-learning "convolution" is technically cross-correlation — no filter flip — but everyone uses the term "convolution".)

**In practice** the operation is done with **PyTorch** as:

```python
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
# input shape: (batch, in_channels, height, width)
# output shape: (batch, out_channels, height', width')
```

**Key parameters:**

- **In-channels** — feature maps in the input (3 for RGB image, or N for N feature maps from a previous layer).
- **Out-channels** — number of filters = number of output feature maps.
- **Kernel size** — spatial extent (typical: 3×3, 5×5, 1×1).
- **Stride** — how far the filter shifts per step (typical: 1 or 2).
- **Padding** — pixels added around the input (typical: `same` padding to preserve spatial size).
- **Dilation** — spacing between kernel points; dilation=2 skips every other input pixel. Used in segmentation and audio.

**Output spatial dimensions** for a 2D conv:

$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2p - d(k - 1) - 1}{s} \right\rfloor + 1$$

Where:
- **$k$** = kernel size.
- **$p$** = padding.
- **$s$** = stride.
- **$d$** = dilation.

Same formula for width.

**Number of parameters** per Conv2d layer:

$$\text{params} = (k_h \cdot k_w \cdot C_{\text{in}}) \cdot C_{\text{out}} + C_{\text{out}}$$

A 3×3 conv from 64 to 128 channels: $(3 \cdot 3 \cdot 64) \cdot 128 + 128 = 73{,}856$ parameters. Compare to a fully-connected layer of the same dimensions on a 32×32 image ($64 \cdot 32 \cdot 32 = 65{,}536$ inputs → $128 \cdot 32 \cdot 32 = 131{,}072$ outputs), which would need ~8.6 billion parameters. **Weight sharing is the difference.**

**Why 3×3 dominates.** Stacking two 3×3 convs has an effective receptive field of 5×5 with fewer parameters and more nonlinearity. Three 3×3 → 7×7. Larger kernels are rarely worth it.

**1×1 convolutions** — surprisingly powerful. They mix channels at a single spatial position, changing the channel dimension. Used to reduce channels ("bottleneck") before an expensive 3×3 conv, or to project back after.

---

### 2.2 Pooling and Downsampling

**Max pooling** — take the max in each $k \times k$ window:

$$O(i, j) = \max_{m, n \in \text{window}} I(i \cdot s + m, j \cdot s + n)$$

**Average pooling** — take the mean. Less common in modern nets; used in the final "global average pooling" layer.

**Global average pooling (GAP)** — average over the entire spatial extent, producing one scalar per feature map. Replaces the flatten + dense layers at the end of a CNN. Reduces parameters and often improves generalization.

```python
pool = nn.AdaptiveAvgPool2d(1)   # produces (B, C, 1, 1)
```

**Stride-2 convolution** — an alternative to pooling. Instead of a separate pooling layer, use `stride=2` in a conv. Trainable downsampling; often preferred in modern architectures (VGG-style pooling → ResNet-style strided conv).

**Why we downsample:**
- Reduce compute for deeper layers.
- **Enlarge receptive field** — each unit "sees" more of the input.
- Extract translation-invariance from small spatial shifts.

**Receptive field** — the region of the input that influences a given output unit. Grows with depth:

$$RF_L = RF_{L-1} + (k_L - 1) \cdot \prod_{i=1}^{L-1} s_i$$

Simplified for stride-1 3×3 convs: $RF_L = 2L + 1$. Stride-2 convs and pooling grow it multiplicatively.

**The point of receptive field:** each layer's units see progressively larger regions of the image. Early layers see edges; deep layers see whole objects. This mirrors visual cortex hierarchy — the biological inspiration for CNNs.

---

### 2.3 Building Blocks: Convolution + BN + Activation

The canonical CNN block:

```python
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
```

**Why bias=False when followed by BN:** BatchNorm has its own learnable shift; a preceding bias is redundant. Same reasoning applies elsewhere.

**A minimal CNN architecture** (LeNet-style, ~1998):

```
Input (1, 28, 28)
→ Conv 5x5, 32 filters + ReLU + MaxPool 2x2      → (32, 14, 14)
→ Conv 5x5, 64 filters + ReLU + MaxPool 2x2      → (64, 7, 7)
→ Flatten                                         → (3136,)
→ Linear(3136, 128) + ReLU                        → (128,)
→ Linear(128, 10)                                 → logits
```

**A modern CNN architecture** (VGG-style, ~2014):

```
Conv-Conv-Pool blocks (repeated)
→ Global Average Pool
→ Linear → logits
```

Deeper and simpler; more expressive per parameter due to the small (3×3) filters.

**AlexNet (2012)** — 8-layer CNN that won ImageNet. First big DL success.

**VGG (2014)** — 16 or 19 layers, all 3×3 convs, uniform structure. Elegant but heavy.

**GoogLeNet / Inception (2014)** — parallel branches with different kernel sizes; 1×1 convs as "bottlenecks."

**ResNet (2015)** — see next section. The template for every modern CNN.

---

### 2.4 Residual Connections and Modern Architectures

**The problem before ResNet.** Beyond ~20 layers, deeper networks *underperformed* shallower ones — even on training data. This wasn't overfitting; it was a training difficulty. Vanishing gradients + optimization difficulty.

**ResNet's insight** (He et al., 2015): instead of learning $F(\mathbf{x})$, learn $F(\mathbf{x}) - \mathbf{x}$. Add a **skip connection** that carries the input around a block:

$$\mathbf{y} = F(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

Where $F$ is a "residual" block (a few conv + BN + ReLU layers). Every block has to only learn a **correction** to its input; if no correction is needed, $F \to 0$ and the block is an identity.

**Why it works:**
- **Gradients flow directly** through the skip, bypassing the block. Depth doesn't kill gradient signal.
- **Identity is a natural default** — the block starts as identity plus small perturbation.
- Empirically enables training networks with hundreds or thousands of layers.

**Residual block structure:**

```
Input
   ↓
Conv 3x3 → BN → ReLU
   ↓
Conv 3x3 → BN
   ↓
   + ←────── (skip connection from input)
   ↓
ReLU
```

**Bottleneck block** (ResNet-50+):

```
Conv 1x1 (reduce channels) → BN → ReLU
Conv 3x3                    → BN → ReLU
Conv 1x1 (restore channels) → BN
+ skip
ReLU
```

Cheaper than three 3×3 convs; same expressive power.

**Modern CNNs building on ResNet:**

- **DenseNet** — every layer connects to every subsequent layer within a block.
- **MobileNet / EfficientNet** — depthwise-separable convolutions for efficient mobile inference.
- **RegNet, NASNet** — architecture search finds ResNet-descended designs.
- **ConvNeXt (2022)** — a modernized CNN that matches Vision Transformers by absorbing their design choices (LayerNorm, GELU, fewer normalization layers, larger kernels).

---

### 2.5 Training a CNN — Practicalities

**Data pipeline:**
- Use `torchvision.transforms` for standard augmentations.
- Normalize using dataset statistics (mean, std per channel).
- `torchvision.datasets` has CIFAR, MNIST, ImageNet, Places, and many others.

**Optimizer & schedule:**
- **SGD + Nesterov momentum (0.9)** + weight decay 5e-4 + LR ~0.1 with step or cosine decay was the classic ResNet recipe.
- **AdamW** + weight decay 0.05 + cosine schedule is a modern default.

**Batch size:** 128–512 is typical for ImageNet. Larger with linear LR scaling.

**Augmentations for image classification:**
- **Basic:** `RandomCrop`, `RandomHorizontalFlip`, `ColorJitter`, `Normalize`.
- **Advanced:** `RandAugment`, `AutoAugment`, `MixUp`, `CutMix`, `TrivialAugment`.
- **Test-time:** center-crop + normalize. No random augmentations.

**Common pitfalls:**
- Forgetting to `.eval()` before validation → BatchNorm and Dropout misbehave.
- Not normalizing inputs → activations at layer 1 wildly different from later stages.
- Learning rate too high → NaN.
- Learning rate too low → training plateaus at bad accuracy.
- Insufficient augmentation → overfits fast.

**Transfer learning workflow** (the highest-value CNN skill):

1. Load a pretrained model (`torchvision.models.resnet50(weights='DEFAULT')`).
2. **Freeze all but the classifier**: `for p in model.parameters(): p.requires_grad = False`.
3. Replace the classifier: `model.fc = nn.Linear(2048, N_CLASSES)`.
4. Fine-tune with a small LR (1e-3 → 1e-5).
5. Optionally unfreeze upper blocks after warming up the classifier.

Transfer learning has been the single biggest ROI in practical deep learning for a decade. On any real vision task with modest data (< 10k images), start with a pretrained backbone.

---

## 3. Mental Models & Analogies

### 3.1 The "Sliding Cookie Cutter" Model

A conv filter is a **cookie cutter**. It slides across the image, and wherever the cutter matches the image's local pattern, the output feature map is high. The cutter has a specific shape (learned during training) — one detector fires on horizontal edges, another on green blobs, another on dog ears.

- **Weight sharing** = the same cookie cutter used at every position. If "dog ear" appears at any spot in the image, it fires.
- **Multiple filters per layer** = a whole set of cookie cutters, each detecting a different pattern.
- **Multiple channels stacked** = each filter operates on all input channels; outputs become inputs for the next layer's cutters. Deep filters match "combinations of combinations" of earlier features.
- **Pooling** = zoom out; details of exactly where the pattern was matter less; presence matters.
- **Deep networks** = a stack of cutters where deep cutters are shaped like "assemblages of earlier cutters' fires."

This model is exact. It explains:
- Why CNNs have translation invariance: the cutter fires wherever the pattern is.
- Why they're parameter-efficient: one cutter, applied everywhere.
- Why 1×1 convs mix channels: a cutter that "matches" specific channel patterns at a single position.

### 3.2 The "Hierarchy of Detectives" Model

Imagine a police investigation with layered detectives:

- **Junior detectives** (early conv layers) each specialize in one clue: fingerprints, tire treads, shoe imprints.
- **Middle detectives** combine junior reports: "if fingerprint AND tire tread, it's Vehicle A near the fingerprint."
- **Senior detectives** integrate mid-level findings into scenarios: "car chase from Vehicle A to Vehicle B, suspect Y matches ."
- **Chief inspector** (final layer) says: "Suspect Y did it, 78% probability."

**Skip connections** = information from junior detectives can bypass middle levels to reach seniors directly, ensuring even highly synthesized reports don't lose access to raw clues.

**Pooling** = generalization; senior detectives don't care exactly *which* pixel the fingerprint was on, only that it was.

**Transfer learning** = a police force that already trained detectives on generic crimes (ImageNet) sends them to solve your specific fraud case; only the chief inspector needs re-training on the new evidence types.

![IMG-CNN-01](/3%20—%20Deep%20Learning/images/IMG-CNN-01.jpg)

> **Caption:** A CNN alternates convolutional feature extraction with spatial pooling; each layer's receptive field grows.
> **Placement:** Section 2.3.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "CNNs Only Work for Images"

Not true. Convolutions are useful anywhere the input has:
- **Local structure** — neighboring positions are related.
- **Translation invariance** — pattern position matters less than pattern shape.

Applications: **1D convs** for audio, EEG, sensor data, protein sequences, time series. **3D convs** for MRI/CT scans, video (temporal + 2D spatial). **Graph convolutions** generalize to non-Euclidean domains (social networks, molecules).

### 4.2 "Deeper Is Always Better"

Without residual connections + BatchNorm + proper initialization, deep CNNs fail to train (see ResNet's motivation). Even with those, past ~200 layers, compute cost outpaces accuracy gains for most tasks. The right depth depends on your data size — a 152-layer ResNet is overkill on a 1000-image dataset.

### 4.3 "Feature Maps Learned by CNNs Are Interpretable Detectors"

Kind of, but not as cleanly as popular science suggests. Early layers do learn edge detectors of various orientations. But deeper layers learn **entangled combinations** — hard to name any single filter as "detects wheels" without picking a curated example. Feature visualization (Olah et al.) shows a spectrum from clean to hopelessly mixed. Don't over-interpret CNN filter visualizations; they're a rough sketch.

---

## 5. Self-Assessment Bank (CNNs)

### Questions

**Q1 (Short answer).** In one sentence, what makes a convolutional layer more parameter-efficient than a fully-connected layer for images?

**Q2 (Multiple choice).** Given a 32×32 input, a 3×3 conv with `stride=1`, `padding=1`, the output spatial dimensions are:
A. 30×30
B. 32×32
C. 34×34
D. 16×16

**Q3 (Short answer).** Compute the number of parameters in a Conv2d with in_channels=64, out_channels=128, kernel=3, bias enabled.

**Q4 (Multiple choice).** The purpose of a residual (skip) connection is:
A. To reduce parameters.
B. To provide a shortcut path for gradients, enabling very deep networks.
C. To add regularization.
D. To speed up inference.

**Q5 (Short answer).** Why do we sometimes use `bias=False` in Conv2d layers?

**Q6 (Multiple choice).** Global average pooling (GAP) at the end of a CNN:
A. Increases the number of parameters.
B. Averages each feature map to a single scalar, producing a fixed-size representation regardless of input size.
C. Only works for square inputs.
D. Is equivalent to max pooling.

**Q7 (Short answer).** Why stack multiple 3×3 convs rather than using a single 7×7 conv?

**Q8 (Multiple choice).** In transfer learning with a pretrained ResNet on a new small dataset, the standard first move is to:
A. Fine-tune all parameters with the same LR.
B. Freeze the backbone; retrain only the final classifier.
C. Retrain from scratch.
D. Prune half the filters.

**Q9 (Short answer).** What is a 1×1 convolution useful for?

**Q10 (Multiple choice).** During inference, if you forget to call `model.eval()`, the CNN's BatchNorm layers will:
A. Behave the same.
B. Use current batch statistics rather than running averages — predictions will vary with batch composition.
C. Throw an error.
D. Skip normalization.

---

### Answer Key & Detailed Explanations

**A1.** **Weight sharing across spatial positions**: a single filter with a small number of parameters (e.g., 9 for a 3×3 filter) is applied at every position of the image. A fully-connected layer would need a distinct weight for every (input position, output position) pair, which for high-resolution images is billions of parameters.

**A2. B.** Padding of 1 with a 3×3 kernel and stride 1 preserves spatial dimensions. Formula: $(32 + 2 \cdot 1 - 3)/1 + 1 = 32$.

**A3.** Parameters = $(k_h \cdot k_w \cdot C_{\text{in}}) \cdot C_{\text{out}} + C_{\text{out}} = (3 \cdot 3 \cdot 64) \cdot 128 + 128 = 73{,}728 + 128 = 73{,}856$.

**A4. B.** Residual connections give gradients an unimpeded backward path from later layers to earlier ones. Without them, very deep networks suffer vanishing gradients. Skip connections also make identity a "natural default" for each block, which stabilizes training. They don't reduce parameters or serve as regularization directly.

**A5.** When a Conv2d is immediately followed by BatchNorm (or GroupNorm), the BN layer has its own learnable shift ($\beta$) that subsumes any bias the conv would add — the net effect is redundant. Setting `bias=False` saves parameters and slightly speeds up computation.

**A6. B.** GAP averages each feature map spatially to a scalar. For a $(B, C, H, W)$ input, output is $(B, C, 1, 1)$ regardless of $H, W$. This makes the CNN input-size-agnostic (as long as the classifier layer takes the correct channel count) and reduces parameters compared to a big flatten + dense.

**A7.** Two 3×3 convs stacked have the same receptive field as one 5×5 conv (5×5), but use fewer parameters ($2 \cdot 3 \cdot 3 = 18$ vs $5 \cdot 5 = 25$ per channel pair) and have more nonlinearity in between. Three 3×3 → 7×7 with 27 vs 49 params. The parameter/expressivity ratio is better for stacked small convs, and empirical results consistently favor them.

**A8. B.** Freeze the pretrained backbone (huge, well-trained features you shouldn't perturb without gradients balanced by lots of new data); train only the final classifier on your small dataset. Optionally, once the classifier converges, unfreeze upper blocks and fine-tune with a small LR.

**A9.** A 1×1 conv is a **pointwise linear projection across channels** at each spatial location. Uses: (1) **Channel reduction** — reduce a 512-channel feature map to 128 before an expensive 3×3 (bottleneck). (2) **Channel expansion** — after a bottleneck. (3) **Combining features from parallel branches** (Inception). (4) **Injecting nonlinearity** with fewer parameters than a spatial conv.

**A10. B.** In training mode (`.train()`), BatchNorm uses the current batch's statistics. In eval mode (`.eval()`), it uses running EMA statistics computed during training. Forgetting `.eval()` means predictions depend on which other samples are in the current batch — unstable and wrong for inference.

---

## 6. Practice Prompts

1. **From scratch (small).** Implement a 4-layer CNN for MNIST from scratch (no pretrained anything). Achieve >99% test accuracy.
2. **ResNet-18 from scratch.** Implement a `BasicBlock`, stack them into ResNet-18, train on CIFAR-10 to at least 90% test accuracy.
3. **Receptive field calculator.** Write a function that, given a stack of (kernel_size, stride, padding) layers, returns the receptive field of a final output unit. Verify against a hand-computed 3-layer example.
4. **Transfer learning.** Fine-tune `torchvision.models.resnet50` (ImageNet-pretrained) on a small (e.g., 200-image) custom dataset. Compare (a) linear probing (freeze backbone), (b) full fine-tuning.
5. **Feature-map viz.** For a trained CNN on CIFAR, visualize the activation of one filter in layer 1 (edges) and one filter deep in the network (mysterious combinations). Observe the hierarchy.

---

## 7. References

- LeCun et al., ["Gradient-Based Learning Applied to Document Recognition"](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) (1998) — LeNet.
- Krizhevsky, Sutskever & Hinton, ["ImageNet Classification with Deep Convolutional Neural Networks"](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) (2012) — AlexNet.
- Simonyan & Zisserman, ["Very Deep Convolutional Networks for Large-Scale Image Recognition"](https://arxiv.org/abs/1409.1556) (2014) — VGG.
- He et al., ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385) (2015) — ResNet.
- Huang et al., ["Densely Connected Convolutional Networks"](https://arxiv.org/abs/1608.06993) (2017) — DenseNet.
- Liu et al., ["A ConvNet for the 2020s"](https://arxiv.org/abs/2201.03545) (2022) — ConvNeXt.
- Chris Olah et al., ["Feature Visualization"](https://distill.pub/2017/feature-visualization/) — Distill.pub.
- PyTorch docs: [torch.nn — Convolution Layers](https://pytorch.org/docs/stable/nn.html#convolution-layers).
