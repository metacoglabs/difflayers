# Analysis: Uniform Memory Retrieval with Larger Capacity for Modern Hopfield Models

## TECHNICAL SUMMARY

The paper proposes `U`-`Hop`, a two-stage retrieval/training scheme for modern Hopfield models. The method introduces a learnable feature map \(\Phi\) to define a kernel similarity \(K(u,v)=\langle \Phi(u),\Phi(v)\rangle\) in place of the standard dot product. In Section 2.1, the feature map is instantiated as a linear map \(\Phi(x)=Wx\) (Eq. (2.1), text: “we consider the linear affine feature map”), so \(K(u,v)=u^\top W^\top W v\). The kernelized Hopfield energy is then defined in Eq. (2.2), using this kernel inside the overlap term and a convex conjugate \(\Psi_\alpha^\star\) associated with Tsallis-entropy-based separations (Section 2.1). Retrieval dynamics are given in Theorem 2.1 / Eq. (2.4), and Lemma 2.1 states that iterates of \(T_K\) converge to stationary points of the kernelized energy.

Stage I learns \(W\) by minimizing a “separation loss” over the stored memory set \(\Xi=\{\xi_\mu\}_{\mu=1}^M\). In Section 2.2, they define an RBF kernel over features and the “Average Separation Loss” \(L_\Phi\) (Definition 2.2), described as “the logarithm of average Gaussian separation of \(\Phi\) vector pairs over \(\Xi\).” Algorithm 1 performs \(N\) gradient steps on \(W\) with step size \(\gamma \le 1/G\), then row-normalizes \(W\), and then runs \(T\) retrieval updates with \(T_K\). For deep learning, Section 3 defines a kernelized Hopfield layer and a deep-learning version of the separation loss (Definition 3.1). Algorithm 2 alternates SGD on the separation loss and SGD on task loss.

Datasets are standard public datasets only: MNIST (60k/10k), CIFAR10 (50k/10k), CIFAR100 (50k/10k), TinyImageNet (100k images, 200 classes), ETTh1, ETTm1, and WTH (Section F.1). No LLM-generated data, synthetic labeling, or special filtering pipeline is described. For retrieval, queries are created either by masking 50% of pixels (Section 4.1, F.2) or adding Gaussian noise with varying “noise level” (Section 4.1, F.3). Metrics are sum-of-square pixel differences for retrieval (Section 4.1), test/max-train accuracy for image classification (Table 1), and MSE/MAE for time series (Table 8).

Experimental setup: retrieval compares against Modern Hopfield, Sparse Modern Hopfield, Dense Associative Memory (10th-order polynomial), and L2/Manhattan similarity baselines (Section 4.1). Classification replaces ViT attention with Hopfield variants and trains for 25 epochs (Tables 5–6). Time series uses STanHop-Net with/without `U`-`Hop` (Section F.5, Table 8). Quantitatively, Table 1 reports CIFAR10 test accuracy improvements from 52.2% to 55.2% (MHM) and 52.0% to 55.4% (Sparse MHM), CIFAR100 from 26.3% to 28.7% and 26.0% to 29.0%, and TinyImageNet from 12.2% to 12.7% and 12.3% to 12.5%. Table 8 shows mixed but often small time-series gains, e.g. ETTh1 horizon 720 MSE 0.631→0.572, while WTH horizon 192 worsens in MSE 0.513→0.528.

## CORE CLAIM

The paper claims that learning a kernelized similarity via a separation-loss pre-stage “enhanc[es] memory capacity across all modern Hopfield models” and that empirically “`U`-`Hop` outperforms all existing modern Hopfield models and SOTA similarity measures” on retrieval and downstream learning tasks (Abstract; also Section 1: “This allows Hopfield models under `U`-`Hop` to distinguish different memory patterns with larger separation and hence achieve larger memory capacity.”).

## MAIN RISKS

1. **The optimization target is not aligned with the claimed capacity quantity.**  
   Threat: the core claim is about larger memory capacity / reduced confusion, but the paper concedes that its loss does not optimize the relevant minimum separation. Section 5 states: “the optimality of separation loss (Definition 2.2) does not guarantee maximal separation for \(R := \frac12 \min_{\mu,\nu\ne\mu}\|\xi_\mu-\xi_\nu\|\)” and earlier Section 1 says it “does not provably guarantee enlarging \(R\).”  
   Why decision-relevant: if Stage I improves average pairwise separation while the nearest-neighbor bottleneck remains unchanged, retrieval failures from close/confusable memories may persist in practice despite lower loss.

2. **Theoretical guarantees rely on assumptions and proof steps that are not justified as written.**  
   Threat: the convergence/fixed-point theory underpins the method’s novelty claim, but Theorem 2.1’s proof sketch says “By Assumption 2.1 and the convexity of \(K\), there exists an inverse map that transforms the CCCP results in kernel space back to the state space” (Section 2.1), while \(K(u,v)=u^\top W^\top W v\) is bilinear in \((u,v)\), not obviously a convex map in the joint arguments. In Appendix E.2 they also write “Since the lse function is non-decreasing and convex, and \(K\) is convex, the composited function lse(\(K(\Xi,x)\)) is convex.”  
   Why decision-relevant: if the proof assumptions are incorrect or incomplete, the claimed monotonic decrease and convergence of retrieval may not hold for the proposed kernelized model, weakening trust in both correctness and reproducibility.

3. **Experimental evidence is weak on statistical rigor and variance reporting.**  
   Threat: the empirical claim of superiority is based on results with effectively no uncertainty characterization. Table 1 says, “We omit variance as all variance are ≤ 0.03%.” Table 8 says “variance omitted as they are all ≤ 2%.” Retrieval experiments in Sections F.2/F.3 say they repeat “20 times for each baseline,” but no confidence intervals or error bars are shown in the text excerpted around Figure 3/Figure 5.  
   Why decision-relevant: adoption decisions depend on whether gains such as TinyImageNet 12.2%→12.7% or CIFAR100 26.3%→28.7% are stable across seeds and tuning; omitted uncertainty makes these deltas hard to trust.

4. **Compute and deployment cost are confounded with the method.**  
   Threat: `U`-`Hop` adds a separate optimization stage before retrieval/training. Algorithm 1 explicitly performs \(N\) extra gradient steps on \(W\), and Section 5 admits “Algorithm 1 has a time complexity of \(O(N+T)\)” and “Algorithm 2 has a time complexity of \(O(N_oN_i)\). Although this increases the standard supervised learning training time by a factor of \(N_i\)…”  
   Why decision-relevant: if improvements come from extra optimization budget rather than an intrinsically better retrieval rule, practitioners may prefer simply spending that compute on baseline tuning/training.

5. **Downstream gains are small and inconsistent relative to broad claims.**  
   Threat: the paper claims in Section 1 and 5 that learning tasks improve by “an average 3% margin” and that generalization improves significantly, but Table 1 shows TinyImageNet gains of only 0.5 points (12.2→12.7 and 12.3→12.5), and Table 8 contains regressions, e.g. WTH horizon 192 MSE worsens from 0.513 to 0.528 and ETTm1 horizon 192/336 MSE worsens from 0.351 to 0.355 and 0.391 to 0.392.  
   Why decision-relevant: practitioners need robust gains across tasks; inconsistent or marginal improvements reduce confidence that the method is generally useful beyond selected settings.

## DOMAIN-SPECIFIC CONCERNS

1. **Associative-memory capacity is governed by worst-case geometry, but the proposed loss optimizes average geometry.**  
   In Hopfield-style retrieval, failures are often triggered by the closest competing memory, not by average pairwise distances. The paper itself defines \(R\) as the minimal pairwise separation in Definition 1.1 and discusses retrieval error in terms of \(\Delta_\mu - 2mR\) in Section 1, yet Definition 2.2 optimizes “Average Separation Loss,” and Section G.2 concedes: “the average loss does not guarantee maximizing \(R_\Phi\), nor does it ensure an optimal \(\Delta_{\Phi,\mu}\).” This is a specialist-level mismatch between the quantity used in theory and the quantity optimized in practice.

2. **Single-step retrieval evaluation is not sufficient for modern Hopfield dynamics claims.**  
   Retrieval dynamics are introduced as iterative energy minimization in Theorem 2.1 and Algorithm 1, but the retrieval experiments in Sections F.2 and F.3 evaluate “a single-step update with various Hopfield models.” In this subfield, iterative convergence behavior and basin-of-attraction size matter; evaluating only one-step reconstruction can miss instability, oscillation, or degradation over multiple steps.

3. **The exact-retrieval theory is restricted to sparse variants, but empirical claims are made broadly across “all modern Hopfield models.”**  
   Section 2.3 states exact retrieval is obtained when using “\(\alpha\)-EntMax as separation when \(\alpha>1\)” and Theorem 2.2 is specifically for \(T_{\text{sparse}}\). However, the abstract and Section 1 repeatedly claim improvement “across all modern Hopfield models.” A specialist would immediately ask for separate evidence that dense/softmax variants benefit from the same exact-retrieval mechanism, since the paper itself says “In the standard modern Hopfield model … the inability of Softmax to satisfy (2.6) results in a lack of exact retrieval.”

4. **The deep-learning story relies on patch/token separation, but the architecture is unusually weak and may not reflect realistic transformer deployments.**  
   Section F.4 states that for CIFAR10/CIFAR100 they “use a fully connected layer right after the encoder” and send patches into a single Hopfield layer with the CLS token as query. Table 5 uses patch size 32 for 32×32 images, meaning each image is effectively one patch plus CLS. For TinyImageNet, Table 6 uses patch size 64 for 64×64 images, again effectively one patch. A domain specialist would question whether “patch/token separation” claims in Section 4.3 are meaningful when there is effectively one image patch rather than a sequence of multiple tokens.

5. **The representation theorem is only meaningful in a restrictive regime that is glossed over in main claims.**  
   Section 3 states Theorem 3.1 requires \(K(u,v)=0\) for distinct memories and that “it is only possible when \(d \ge M\). In the context of deep learning, the patch size must not be larger than the hidden dimension to realize this result.” This is a strong dimensionality constraint. In realistic transformers, sequence length often exceeds hidden dimension constraints in ways not addressed here; yet the paper presents this as explaining broad “low-rank bottleneck” improvements (Figure 4, Section 4.2).

## STRENGTHS

- **The method is clearly specified as a concrete two-stage algorithm.** Algorithm 1 gives the exact procedure: \(N\) gradient steps on \(W\) using \(\nabla_W L_\Phi(\Xi)\), row normalization, then \(T\) retrieval iterations using \(T_K\) from Theorem 2.1 (Section 2.2).

- **The paper attempts to connect the modified retrieval rule to modern Hopfield theory rather than presenting it as pure engineering.** Section 2.1 provides Theorem 2.1 and Lemma 2.1 to argue monotonic energy decrease and convergence to stationary points for the kernelized energy, and Section 2.3 states explicit conditions for exact retrieval under sparse variants (Theorem 2.2, Corollary 2.2.1).

- **Experiments cover both associative-memory retrieval and downstream tasks.** Section 4 evaluates retrieval on MNIST/CIFAR10, classification on CIFAR10/CIFAR100/TinyImageNet, and time-series forecasting on ETTh1/ETTm1/WTH, rather than only a single benchmark family.

- **Some retrieval gains appear large in the figures and are supported by correlation analyses.** Figure 1 and Appendix G.1 explicitly report a “strong correlation between low separation loss and low retrieval error,” and Figure 5 shows retrieval error decreasing as Stage-I iteration count \(N\) increases.

- **The paper includes implementation details and hyperparameter tables.** Appendix F specifies datasets, optimizers, learning rates, epochs, batch sizes, patch sizes, memory set sizes, and kernel epochs (Tables 4–7), which is better than a purely high-level experimental description.

## WEAKNESSES

- **The central optimization objective is admitted not to guarantee the claimed capacity quantity.** Section 5: “the optimality of separation loss (Definition 2.2) does not guarantee maximal separation for \(R\),” and Section G.2 reiterates that average loss “does not guarantee maximizing \(R_\Phi\), nor does it ensure an optimal \(\Delta_{\Phi,\mu}\).”

- **The paper overstates theoretical optimality despite that limitation.** Section 2 says Algorithm 1 is “a two-stage algorithm for the kernel learning with optimal theoretical guarantees,” while Section 5 later states the method does not guarantee maximal \(R\) and the optimal-capacity result is deferred to follow-up work (“Hu et al., 2024d”). These statements are in tension.

- **The proof logic for convexity/convergence is questionable as written.** Appendix E.2 claims “Since the lse function is non-decreasing and convex, and \(K\) is convex, the composited function lse(\(K(\Xi,x)\)) is convex,” but with \(K(u,v)=u^\top W^\top W v\) from Section 2.1, convexity is not straightforward. Theorem 2.1’s proof sketch in Section 2.1 also relies on “the convexity of \(K\)” and the existence of “an inverse map” without precise conditions.

- **Retrieval evaluation uses only single-step updates despite an iterative algorithm and convergence claims.** Sections F.2 and F.3 both state they perform “a single-step update with various Hopfield models,” whereas Algorithm 1 includes \(T\) retrieval iterations and Theorem 2.1/Lemma 2.1 concern iterative energy minimization.

- **No compute-matched baseline is provided despite extra Stage-I optimization.** Algorithm 1 adds \(N\) gradient steps, and Section 5 acknowledges increased time complexity, but no baseline receives comparable extra optimization/tuning budget to isolate whether gains are due to the specific method versus more compute.

- **Variance reporting is insufficient and in some places simply omitted.** Table 1: “We omit variance as all variance are ≤ 0.03%.” Table 8: “variance omitted as they are all ≤ 2%.” This omits confidence intervals and makes small gains hard to interpret.

- **Some downstream gains are too small to support broad generalization claims.** Table 1 shows TinyImageNet gains of only 0.5 and 0.2 points in test accuracy, while Section 4.2 describes “significant” improvements in generalization and convergence.

- **Time-series improvements are not consistent.** Table 8 includes degradations for ETTm1 at horizons 192 and 336 in MSE/MAE and for WTH at horizon 192 in both metrics, conflicting with the strong cross-domain performance narrative in Section 4.2.

- **The image-classification setup is unusually limited and may confound claims about token-level separation.** Table 5 uses patch size 32 on CIFAR10/CIFAR100 images of size 32×32, and Table 6 uses patch size 64 on TinyImageNet images of size 64×64, implying one patch per image. This weakens Section 4.3’s discussion of patch/token geometry and representation learning.

- **Key theorem validation is indirect.** Table 1 caption says “the improvement on Max. Training accuracy is a validation of Theorem 3.1,” but Theorem 3.1 is an existence/expressiveness statement under conditions including \(M\le d\) and pairwise orthogonality in kernel space (Section 3, Appendix E.4), not directly tested by train accuracy alone.

## FORENSIC DEEP-DIVE

### Math & Logic Errors

#### 1. The convergence proof leans on an unsubstantiated convexity property of the kernelized overlap.
- **Citation:** Section 2.1 proof sketch for Theorem 2.1: “By Assumption 2.1 and the convexity of \(K\), there exists an inverse map…” Appendix E.2: “Since the lse function is non-decreasing and convex, and \(K\) is convex, the composited function lse(\(K(\Xi,x)\)) is convex.”
- **Issue:** The paper defines \(K(u,v)=u^\top W^\top W v\) from Eq. (2.1). This is bilinear in \((u,v)\), and when viewed as a function of \(x\) with fixed \(\Xi\), each component \(K(\xi_\mu,x)=\xi_\mu^\top W^\top W x\) is affine/linear in \(x\), not a generally convex kernel map in the sense invoked by the proof. The text never states the domain/argument in which \(K\) is convex, yet the CCCP argument depends on it.
- **Why it matters:** The core claim depends on Theorem 2.1 and Lemma 2.1 to justify that the proposed kernelized retrieval still behaves like a valid modern Hopfield model. If the convexity argument is not correct, the monotonic-decrease and convergence guarantees are not established as claimed.

#### 2. The proof introduces an “inverse map” from kernel space to state space without a precise construction.
- **Citation:** Section 2.1 proof sketch: “there exists an inverse map that transforms the CCCP results in kernel space back to the state space.”
- **Issue:** Assumption 2.1 only states \(W\in \mathbb{R}^{D_\Phi\times d}\) is full rank with \(D_\Phi \gg d\). This implies injectivity of the linear map \(x\mapsto Wx\), but the paper never defines the inverse used in the algorithmic retrieval derivation, nor analyzes how the inverse interacts with the energy minimization in Eq. (2.2).
- **Why it matters:** This is the bridge between the learned kernel representation and actual retrieval in the original state space. Without a precise inverse argument, the retrieval rule may be a formal manipulation rather than a rigorously derived dynamics.

### Eval Gaps

#### 3. The retrieval experiments do not test the iterative retrieval dynamics that the theory is about.
- **Citation:** F.2: “using the masked image as a query for a single-step update”; F.3: “performed a single-step update with different Hopfield models.” Algorithm 1, in contrast, includes “for \(t=1,\dots,T\) do \(x \leftarrow T_K(x)\)” and Theorem 2.1/Lemma 2.1 discuss monotonic energy decrease over iterations.
- **Issue:** The main theoretical selling point is the retrieval dynamics under the kernelized energy. But the empirical protocol only evaluates one update step.
- **Why it matters:** A practitioner adopting this as an associative-memory retrieval rule would care about full iterative convergence, basin behavior, and whether repeated application helps or hurts. The paper’s experiments do not validate that.

#### 4. No ablation isolates whether gains come from kernelization, row-normalization, or extra optimization.
- **Citation:** Algorithm 1 includes three changes relative to baselines: Stage-I gradient descent on \(L_\Phi\), row normalization (“Normalize the rows of \(W\)”), and Stage-II retrieval with \(T_K\). Section B further says “the removal of outliers is achieved by the row-wise normalization in `U`-`Hop` (see line 4 of Algorithm 1).”
- **Issue:** There is no ablation comparing (i) kernelized retrieval without Stage I, (ii) Stage I without row normalization, (iii) row normalization only, or (iv) extra baseline optimization budget.
- **Why it matters:** Without these controls, the paper cannot attribute gains to the proposed separation objective rather than to normalization or compute.

### Confounds

#### 5. The deep-learning experiments use one-patch images, making patch-separation claims doubtful.
- **Citation:** Table 5: CIFAR patch size = 32; F.1 says CIFAR images are 32×32. Table 6: TinyImageNet patch size = 64; F.1 says TinyImageNet images are 64×64. Section 4.3 argues `U`-`Hop` “maximizes the pairwise distance between patches” and that “token/patch level” separation improves generalization.
- **Issue:** With patch size equal to image size, each image contributes effectively one patch token (plus CLS). Then the mechanism is not separating many intra-image patches/tokens in the usual transformer sense.
- **Why it matters:** This directly weakens the paper’s explanatory narrative for why downstream generalization improves and limits transfer of the conclusions to realistic ViT settings with many patches.

#### 6. Claimed broad superiority is undermined by mixed downstream results.
- **Citation:** Abstract: “outperforms all existing modern Hopfield models and SOTA similarity measures.” Section 5: “learning tasks by an average 3% margin.” Table 8 shows regressions, e.g. ETTm1 horizon 192 MSE 0.351→0.355, horizon 336 MSE 0.391→0.392, WTH horizon 192 MSE 0.513→0.528.
- **Issue:** The “outperforms” claim reads universal, but Table 8 is mixed.
- **Why it matters:** This is not just wording: practitioners in time-series would see that the method is not reliably better, so the claim is overstated for cross-domain adoption.

### Scope

#### 7. The paper uses broad language about “all modern Hopfield models” although exact-retrieval theory only covers sparse variants.
- **Citation:** Abstract: “enhance memory capacity across all modern Hopfield models.” Section 2.3: “we show `U`-`Hop` achieves exact memory retrieval when \(\alpha>1\)… we study the application of `U`-`Hop` with \(\alpha\)-EntMax as separation when \(\alpha>1\).”
- **Issue:** The key exact-retrieval theorem is variant-specific, not universal.
- **Why it matters:** The strongest theory does not cover the full scope of the empirical/core claim, so the paper overgeneralizes.

## MISSING EVALUATIONS

1. **Multi-step retrieval curves and convergence diagnostics.**  
   Missing experiment: evaluate retrieval error and energy across multiple Stage-II iterations \(T\), not only single-step updates.  
   Claim tested: Theorem 2.1 / Lemma 2.1 and the core claim that the retrieval dynamics improve memory retrieval.  
   Why decision-relevant: practitioners need to know whether repeated application converges stably and whether the gains persist or vanish after more than one step.

2. **Compute-matched baselines.**  
   Missing experiment: give baseline models equivalent extra optimization budget (e.g., extra training epochs, extra retrieval tuning, or learning an alternative projection) matching Stage-I compute in Algorithm 1 / Algorithm 2.  
   Claim tested: that gains are due to the `U`-`Hop` method, not simply more optimization.  
   Why decision-relevant: if equal compute closes the gap, the practical value of the proposed method is much lower.

3. **Ablation of Stage-I loss vs row normalization vs kernelization.**  
   Missing experiment: compare (a) learned kernel without separation loss, (b) row-normalization only, (c) separation loss without row normalization, and (d) fixed random \(W\).  
   Claim tested: Section 1/2’s claim that separation-loss minimization is the mechanism reducing metastable states and increasing capacity.  
   Why decision-relevant: without this, implementers do not know which components are necessary.

4. **Worst-case separation metrics.**  
   Missing experiment: report \(R_\Phi\), nearest-neighbor margins, and distribution of per-memory \(\Delta_{\Phi,\mu}\) before/after Stage I.  
   Claim tested: larger memory capacity and reduced confusion.  
   Why decision-relevant: Section 5 and G.2 admit average loss may not improve \(R_\Phi\); measuring the actual worst-case geometry would validate or refute the central mechanism.

5. **Realistic ViT patch settings with multiple patches per image.**  
   Missing experiment: classification with standard patch sizes smaller than the image (e.g. 4, 8, 16 on CIFAR/TinyImageNet).  
   Claim tested: Section 4.3’s token/patch-level separation story and the connection to attention/low-rank bottlenecks.  
   Why decision-relevant: current results use one patch per image, so they do not show the method helps in realistic multi-token transformer regimes.

6. **Seed-level statistical reporting with confidence intervals.**  
   Missing experiment: report mean ± std/CI over multiple random seeds for all downstream tasks and retrieval figures.  
   Claim tested: the empirical superiority statements in the abstract, Table 1, and Table 8.  
   Why decision-relevant: small gains like 12.2→12.7 on TinyImageNet are not actionable without uncertainty.

7. **Scaling with memory-set size and feature dimension \(D_\Phi\).**  
   Missing experiment: vary \(D_\Phi\), retrieval/training memory size, and show whether gains persist or saturate.  
   Claim tested: “enhanced memory capacity” and broad applicability “across all modern Hopfield models.”  
   Why decision-relevant: users need to know the compute-memory tradeoff and whether the method only helps at one selected scale.

## SHARPEST FLAW

The sharpest flaw is that the paper’s optimization target is not the quantity its theory and motivation say matters for memory capacity. The method optimizes the “Average Separation Loss” in Definition 2.2, but the paper’s own retrieval discussion in Section 1 is based on the minimum-separation-dependent quantities \(R\) and \(\Delta_\mu\), and the authors explicitly admit in Section 5 that “the optimality of separation loss (Definition 2.2) does not guarantee maximal separation for \(R\)” and in Section G.2 that average loss “does not guarantee maximizing \(R_\Phi\), nor does it ensure an optimal \(\Delta_{\Phi,\mu}\).” Since capacity failures in associative memory are driven by worst-case confusable pairs, this mismatch directly undermines the core claim that the proposed Stage-I optimization increases memory capacity rather than merely improving an average geometric proxy.

## ACCEPTANCE RECOMMENDATION

**Reject**

**Reasoning:** The paper’s central loss is explicitly acknowledged not to optimize the worst-case separation quantity tied to memory capacity (Section 5, Section G.2), while the theoretical justification for kernelized convergence relies on insufficiently supported convexity claims (Section 2.1, Appendix E.2).

## DATASET & DEPLOYMENT AUDIT

### DATASETS

- **Scale/distribution mismatch for retrieval evaluation.**  
  The retrieval benchmarks use artificially corrupted in-distribution image memories rather than realistic associative-memory workloads. Section 4.1 says queries are generated by “randomly masking 50% of pixels in the target image,” and Sections F.2/F.3 use either masking or additive Gaussian noise. This means the evaluation distribution is synthetic corruption of stored examples, not unseen or naturally varying queries. Results may therefore overstate practical retrieval robustness.

- **Synthetic query construction may bias outcomes toward pixel-level reconstruction metrics.**  
  For retrieval, the paper uses “Sum-of-Square pixel differences between the ground truth image and the retrieved image” (Section 4.1) after masking/noising stored images (Sections F.2/F.3). This metric and setup favor low-level image fidelity, but do not test semantic retrieval or robustness to realistic perturbations.

- **No label-quality discussion for downstream time-series targets.**  
  Section F.1 lists ETTh1/ETTm1/WTH sources and target variables, but there is no discussion of missing values, preprocessing, splits, or noise handling. For forecasting benchmarks, such preprocessing strongly affects results; the omission makes the data pipeline under-specified.

- **No evidence of train/test contamination checks in retrieval or classification.**  
  The paper uses standard datasets (Section F.1) and says retrieval experiments iterate over “every image in the memory set” (Section F.2), but it does not state whether memory sets for retrieval are sampled exclusively from train or test portions, or whether downstream kernels are learned using only training data in all cases. This is especially relevant because Stage I learns directly on the stored memory set.

### DEPLOYMENT / PRODUCTIONIZATION

- **Inference-time dependence on a separately learned kernel / extra component.**  
  Algorithm 1 requires a learned matrix \(W\) from Stage I before retrieval, then row normalization, then Stage-II retrieval with \(T_K\). This means deployment needs both the memory set and the learned kernel parameters, not just a standard Hopfield update.

- **Additional training-time infrastructure and optimization complexity.**  
  Algorithm 2 alternates two optimization loops with separate batch sizes and step sizes; Section 5 states “Algorithm 2 has a time complexity of \(O(N_oN_i)\)” and “increases the standard supervised learning training time by a factor of \(N_i\).” This adds integration complexity versus standard attention/Hopfield layers.

- **Potential latency concerns due to pre-optimization on memory set.**  
  For retrieval, Algorithm 1 requires \(N\) gradient steps over the memory set before using the model, and F.2/F.3 use up to 1000 kernel epochs (Table 4). If memories change online, this retraining burden could be substantial; the paper does not discuss update frequency or amortization.

- **Versioning / drift sensitivity to the stored memory set.**  
  The abstract says the kernel “utilizes the stored memory patterns as learning data,” and Section 2.2 defines \(L_\Phi\) over the memory set \(\Xi\). This implies that changing the stored memory bank changes the learned similarity itself. In production, memory-bank drift would require re-learning \(W\), but the paper does not discuss this operational issue.

- **Under-specified hardware/compute assumptions.**  
  The acknowledgments mention use of the “Quest high performance computing facility at Northwestern University,” but the paper gives no GPU counts, wall-clock time, or hardware budget in the main experimental sections or Appendix F. This limits reproducibility and deployment planning.

## PRODUCTIONIZABILITY SCORECARD

| Dimension                   | Score 1-5 | Evidence from paper                  |
|-----------------------------|-----------|--------------------------------------|
| Reproducibility             | 3 | Appendix F gives datasets and hyperparameters (Tables 4–7), but hardware, seeds, and full preprocessing details are missing; variance often omitted (Table 1, Table 8). |
| Data availability           | 4 | All named datasets are public standard benchmarks in Section F.1 (MNIST, CIFAR10/100, TinyImageNet, ETT, WTH). |
| Compute accessibility       | 2 | Section 5 adds extra \(N\) / \(N_i\) optimization loops; acknowledgments mention HPC usage; no concrete hardware budget is provided. |
| Implementation completeness | 3 | Algorithm 1 and 2 are provided and Appendix F lists many hyperparameters, but some mathematical definitions/equations are incomplete in the text extract and several operational details are unspecified. |
| Generalization evidence     | 3 | Evidence spans retrieval, image classification, and time series (Section 4, Table 8), but gains are mixed/inconsistent and some setups are narrow (one-patch images via Tables 5–6). |
| Claim-to-evidence ratio     | 2 | Broad claims like “across all modern Hopfield models” and “outperforms all existing” exceed the restricted theory and mixed empirical results (Abstract, Section 2.3, Table 8). |
| Statistical rigour          | 2 | Variance is omitted in Table 1 and Table 8; no confidence intervals; retrieval repeats are mentioned in F.2/F.3 but not fully reported. |

Overall productionizability: 2.7/5

## POINTERS

- Section 5 explicitly admits that Definition 2.2 “does not guarantee maximal separation for \(R\),” undermining the claim that the method improves memory capacity by enlarging the relevant separation quantity.
- Section G.2 further states the average loss “does not guarantee maximizing \(R_\Phi\), nor does it ensure an optimal \(\Delta_{\Phi,\mu}\),” showing the training objective is misaligned with the paper’s own retrieval theory from Section 1.
- Theorem 2.1’s proof sketch in Section 2.1 relies on “the convexity of \(K\),” but Eq. (2.1) defines \(K(u,v)=u^\top W^\top W v\), for which the needed convexity property is not established.
- Appendix E.2 claims “lse(\(K(\Xi,x)\)) is convex” because lse is convex and “\(K\) is convex,” but no valid convexity argument is given for the proposed bilinear kernel form.
- Sections F.2 and F.3 evaluate retrieval using “a single-step update,” despite Algorithm 1 and Lemma 2.1 being about iterative retrieval dynamics and convergence.
- Algorithm 1 includes both Stage-I optimization and row normalization, but the paper provides no ablation isolating the effect of row normalization from the separation-loss optimization.
- Section B attributes “removal of outliers” to the row-wise normalization in Algorithm 1, yet there is no experiment comparing with/without this normalization step.
- Section 5 acknowledges that `U`-`Hop` “increases the standard supervised learning training time by a factor of \(N_i\),” but no compute-matched baseline is presented.
- Table 1 omits variance entirely (“all variance are ≤ 0.03%”), leaving small gains like TinyImageNet 12.2%→12.7% without uncertainty estimates.
- Table 8 also omits variance (“≤ 2%”) while showing regressions such as WTH horizon 192 MSE 0.513→0.528 and ETTm1 horizon 192 MSE 0.351→0.355.
- Section 2.3’s exact-retrieval result is specifically for sparse variants with “\(\alpha>1\),” yet the abstract claims enhanced capacity “across all modern Hopfield models.”
- Table 5 uses patch size 32 for 32×32 CIFAR images, and Table 6 uses patch size 64 for 64×64 TinyImageNet images, so the classification setup effectively uses one patch per image.
- Because of those patch sizes, Section 4.3’s explanation that Stage I improves “patch/token level” geometry is not well supported by the actual image-classification experiments.
- Section 3 states Theorem 3.1 is only possible when “\(d \ge M\),” but the paper still uses it to motivate broad claims about overcoming attention/Hopfield low-rank bottlenecks.
- The Table 1 caption claims max train accuracy validates Theorem 3.1, but Theorem 3.1 is an existence theorem under specific orthogonality assumptions, not a direct prediction about train accuracy.
- Section 4.1 states retrieval robustness is tested using masked or Gaussian-noised images, which is a narrow synthetic corruption regime and may not reflect real associative-memory queries.