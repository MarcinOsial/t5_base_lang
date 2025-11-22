# EFFICIENT MULTI-SOURCE KNOWLEDGE TRANSFER BY MODEL Merging

Anonymous authors

Paper under double-blind review

# ABSTRACT

While transfer learning is an advantageous strategy, it overlooks the opportunity to leverage knowledge from numerous available models online. Addressing this multi-source transfer learning problem is a promising path to boost adaptability and cut re-training costs. However, existing approaches are inherently coarse-grained, lacking the necessary precision for granular knowledge extraction and the aggregation efficiency required to fuse knowledge from either a large number of source models or those with high parameter counts. We address these limitations by leveraging Singular Value Decomposition (SVD) to first decompose each source model into its elementary, rank-one components. A subsequent aggregation stage then selects only the most salient components from all sources, thereby overcoming the previous efficiency and precision limitations. To best preserve and leverage the synthesized knowledge base, our method adapts to the target task by fine-tuning only the principal singular values of the merged matrix. In essence, this process only recalibrates the importance of top SVD components. The proposed framework allows for efficient transfer learning, is robust to perturbations both at the input level and in the parameter space (e.g., noisy or pruned sources), and scales well computationally. Our code is provided in the supplementary.

# 1 INTRODUCTION

The increasing complexity of models and the immense computational costs associated with their training necessitate the efficient utilization of existing resources. Transfer learning Zhuang et al. (2020), which involves initializing networks with weights from a pretrained model, has emerged as a standard practice. This practice relies on foundational models, such as large-scale vision transformers Awais et al. (2025) and self-supervised models Caron et al. (2021), which learn robust and generalized representations from vast, general-purpose datasets (e.g., ImageNet, LAION-5B). By effectively leveraging this broad pre-existing knowledge, transfer learning significantly reduces the demand for extensive task-specific data, accelerates training, and enhances overall model performance across a wide range of computer vision tasks.

However, the wealth of specialized knowledge residing in other fine-tuned models remains largely untapped. Each model represents a valuable knowledge asset, with hundreds of thousands of versions publicly available on platforms like Hugging Face. Each new adaptation typically requires training from its original, pre-trained state, neglecting the specialized knowledge already acquired by previously fine-tuned models for distinct tasks. This gap has sparked considerable interest in developing methods for combining multiple models into a unified model Shu et al. (2021); Yang et al. (2022). Among these is model merging Yang et al. (2024), which presents a notable opportunity to fuse capabilities at low cost, with an example of an aTLAS method Zhang et al. (2024), which addresses the multi-source knowledge transfer for a new target task. It learns to scale and combine task vectors anisotropically Ilharco et al. (2022), which are the weight differences between fine-tuned models and their pre-trained state. The method operates by learning a distinct coefficient for each of the  $T$  tasks, across each of the  $L$  layers, and for each of  $P$  partitions within a weight matrix. These coefficients collectively form a learned tensor with dimensions  $T \times L \times P$ , allowing for adjustments to the model's behavior for new tasks. While holding significant promise, aTLAS lacks mechanisms for granular parameter selection, which restricts the precision of knowledge fusion. Furthermore, aTLAS's memory footprint scales linearly with the number of added sources due to its reliance on using full task vectors. This design prevents the aggregation of larger models or a

greater number of source models. As a result, its training is confined to multi-GPU environments, undermining its parameter-efficient benefits. This coarse-grained approach lacks a robust knowledge composition mechanism, making it susceptible to perturbations from both corrupted or pruned parameters and degraded inputs.

In this paper, we present a unified method that efficiently combines specialized knowledge from multiple fine-tuned source models in the parameter space to facilitate transfer to a new, unseen target task. We depart from the methodology proposed in the aTLAS paper, which assumes that the entire set of full-rank task vectors is used throughout the entire training process. Instead, we propose a more scalable approach that first aggregates knowledge and then allows for its efficient refinement during adaptation. First, we leverage Singular Value Decomposition (SVD) to decompose each task vector into its elementary, rank-one components. This allows us to identify and isolate granular patterns learned for each source task. A subsequent combination stage aggregates these components from all source models, performing a joint ranking to retain only a small, fixed number of the most significant ones. We term this strategy AXIS, as it embodies the principle of Aggregation by eXtraction of Important Singular components.

Such selective aggregation ensures a stable memory usage and constant wall-time footprint during training, irrespective of the number of source models or original task matrix sizes (see Figure 5). Consequently, the proposed design is not only more parameter-efficient, but it also proves to be more robust. Our key contributions include:

- We introduce a scalable approach, AXIS, which outperforms the state-of-the-art method, aTLAS, across a wide spectrum of evaluation conditions, including 21 distinct tasks and various parameter budgets.  
- The computational efficiency of AXIS is a key advantage, allowing for the scaling of knowledge transfer from a large number of source tasks and larger models.  
- We demonstrate that AXIS exhibits robustness against degradations at both the parameter and input levels.  
- Through ablation studies, we offer insights into the underlying structure of knowledge composition and how it can be leveraged.

# 2 RELATED WORKS

![](images/a168143ea6b95e7a7b6a4619806b35537934ba7832726683d56037b6dea88c9d.jpg)  
Figure 1: Accuracy versus the number of trainable parameters for our method and aTLAS, averaged over all target tasks with ViT-B-32 architecture. Each data point corresponds to a fine-tuning parameter budget defined by the top N singular values  $(N = 10\%, 20\%,$  and  $40\%)$ . The solid line denotes the mean accuracy, while the shaded area represents the standard deviation. The variation is calculated over all source task vectors.

Model merging is gaining traction as a promising approach to leverage fine-tuned models without requiring access to training data or incurring increased model size and inference costs. The merging stage itself demands low computational resources and could be entirely training-free. While numerous works explore combining models' weights with diverse architectures Du et al. (2025) or those trained without a shared initialization Rinaldi et al. (2025); Stoica et al. (2023); Ainsworth et al. (2022), these often prove less effective than approaches that assume all considered models originate from the same base model Akiba et al. (2025); Yang et al. (2023); Yadav et al. (2023). This greater effectiveness is largely built upon the concept of a task vector, introduced by Ilharco et al. (2022), which operates on full-rank weight matrices, in contrast to merging low-rank approximations, such as LoRA modules Zhao et al. (2024). Model merging can enhance single-task performance Wortsman et al. (2022a); Ramé et al. (2023); Jang et al. (2024) or be utilized in the creation of multitask models Marczak et al. (2025); Gargiulo et al. (2025). While merged models for multitask performance show limited promise for cross-domain compositional generalization Tam et al. (2024), we focus on explicitly reusing weights for

distinct, new target tasks. Other prior works focus on merging reasoning skills with Chains-of-Thought Yin et al. (2025) for better zero-shot knowledge composition.

Singular Value Decomposition (SVD) offers a valuable approach for parameter-efficient fine-tuning (PEFT), allowing effective modifications within the eigenspectrum of pre-trained weights Wang et al. (2024); Bałazy et al. (2024); Peng et al. (2024); Meng et al. (2024). While many of these strategies achieve parameter efficiency by focusing on the singular values, diverse approaches exist Lingam et al. (2024). Others leverage SVD with reinforcement learning at inference time, adapting to unseen target tasks Sun et al. (2025). We introduce a unique adaptation strategy that diverges from prior work in two critical ways. First, we apply SVD to a multi-source merged model. Second, departing from the more varied heuristics seen before, our adaptation is guided exclusively by the largest singular values.

# 3 METHOD

# 3.1 PROBLEM STATEMENT

Let the parameters of the base, pre-trained model be denoted by  $\theta_{\mathrm{pre}}$ . We consider a set of  $T$  distinct tasks. For a given task  $i$ , the model is fine-tuned on a corresponding dataset  $D_{i}$ . The parameters of this fine-tuned model are denoted as  $\theta_{i}$ . Finally, the parameters for a specific layer  $l$  within this model are represented by  $\theta_{i}^{(l)}$ . A task vector is the element-wise difference between the parameters of a fine-tuned model and its pre-trained counterpart. Building on this concept, we define a per-layer task difference to capture these modifications with greater granularity. Denoting the parameters of the base model for layer  $l$  as  $\theta_{\mathrm{pre}}^{(l)}$  and the fine-tuned parameters for task  $i$  at layer  $l$  as  $\theta_{i}^{(l)}$ , we define task vectors  $\tau_{i}^{(l)}$  as:

$$
\tau_ {i} ^ {(l)} = \theta_ {i} ^ {(l)} - \theta_ {\mathrm {p r e}} ^ {(l)}
$$

For all other modules (e.g., biases, normalization), we retain the term  $\tau_i^{(l)}$ . For these non-matrix parameters, we simply compute their element-wise average across all source tasks, similar to other works. The entire procedure, from decomposition to adaptation, is performed independently for each relevant layer in the model. For brevity, we will generally omit the layer index  $(l)$ . While non-parametric operations, such as activation functions, are applied during the model's forward pass, they do not have learnable weights and are therefore not represented in the task vector.

# 3.2 DECOMPOSING TASK MATRICES

To capture the structured modifications introduced by fine-tuning, we perform a granular analysis of each task matrix,  $\Delta_{i}$ , using Singular Value Decomposition (SVD). For a given task matrix  $\Delta_{i}$  at any generic layer, we consider its SVD:

$$
\Delta_ {i} = \boldsymbol {U} _ {i} \boldsymbol {\Sigma} _ {i} \boldsymbol {V} _ {i} ^ {\top}
$$

where  $U_{i}\in \mathbb{R}^{m\times r_{i}}$  and  $V_{i}\in \mathbb{R}^{n\times r_{i}}$  are the matrices of left and right singular vectors, respectively, and  $\Sigma_{i}\in \mathbb{R}^{r_{i}\times r_{i}}$  is a diagonal matrix containing the singular values  $\sigma \in \mathbb{R}^{r_i}$ . The value  $r_i$  denotes the rank of the matrix  $\Delta_{i}$  and corresponds to the number of its singular components.

Given a pre-trained model, parameterized by  $\theta_{\mathrm{pre}}$ , and a library of  $T - 1$  source task vectors,  $\{\Delta_i\}_{i = 1}^{T - 1}$ , our objective is to synthesize this knowledge to effectively adapt the model for a new, unseen target task. The original training datasets for these source tasks, i.e.,  $\{D_1,\dots,D_{T - 1}\}$ , are not available. For the target task, we only have access to its labeled dataset, which is partitioned into a training set  $D_{\mathrm{t}}^{\mathrm{train}}$  and a test set  $D_{\mathrm{t}}^{\mathrm{test}}$ .

# 3.3 OUR TWO-STAGE COMPOSITION FRAMEWORK

# STAGE 1: KNOWLEDGE EXTRACTION AND AGGREGATION.

Our core hypothesis is that the most transferable useful knowledge for the target task, encoded across diverse source tasks  $\{\Delta_i\}_{i=1}^{T-1}$ , is within the principal singular components, which represent the most dominant structural patterns in the parameter space.

![](images/4c9c6afedf926169a35ab2cdf6c02a2b27861af90a0074d986d3e041c79f0437.jpg)

![](images/b58aab298239e23f43ea95968860f461aafcbaa3c256dde4481cc0886afa15a3.jpg)  
Figure 2: An overview of the AXIS framework. The process consists of two stages: (1) Extraction and aggregation: Each source task matrix  $(\Delta_1, \Delta_2, \ldots)$  is decomposed into its elementary singular components using SVD. The most salient components from all sources are selected based on a global Top-K ranking of their singular values. These K components are then summed to synthesize the merged task matrix,  $\Delta_m$ . For clarity, the diagram illustrates this with  $\mathrm{K} = 2$ . (2) Adaptation: To form a stable and decorrelated transfer basis,  $\Delta_m$  is re-parameterized via a final SVD. The model is then adapted to the target task by fine-tuning only a small subset (Top-N) of the most principal singular values of the resulting matrix  $\Sigma_t$  in each layer.

# Algorithm 1 AXIS

1: Initialize SVD components:  $\mathcal{C}\gets \emptyset$  
2: for each source task  $i \in \{1, \dots, T - 1\}$  do  
3: Compute the SVD of  $\Delta_{i} = U_{i}\Sigma_{i}V_{i}^{T}$  
4:  $\mathcal{C} \gets \mathcal{C} \cup \{(\mathbf{u}_j, \sigma_j, \mathbf{v}_j^\top)\}_{j=1}^{r_i}$  
5: end for  
6: Select the top-K components to form  $\mathcal{B}$  
7:  $\operatorname{Sort}_{\sigma_k \downarrow}(\mathcal{C}) \to \mathcal{B}$  
8: Assemble non-orthogonal vectors:  
9:  $U_{m}\gets [u_{1}|u_{2}|\ldots |u_{K}]$  
10:  $\Sigma_{m}\gets \mathrm{diag}(\sigma_{1},\sigma_{2},\dots ,\sigma_{K})$  
11:  $V_{m}\gets [v_{1}|v_{2}|\ldots |v_{K}]$  
12: Reconstruct from components:  
13:  $\Delta_{m}\gets U_{m}\Sigma_{m}V_{m}^{\top}$  
14: Re-orthogonalize the basis via SVD:  
15:  $\Delta_{m} = U_{\mathrm{t}}\Sigma_{\mathrm{t}}V_{\mathrm{t}}^{\top}$  
16: Define the set of learnable parameters  $\Lambda$  as the top- $N$  singular values from  $\Sigma_{\mathrm{t}}$ :  
17:  $\Lambda \gets [s_1, \ldots, s_N]$ .  
18: Define frozen singular values:  
19:  $\mathbf{s}_{\mathrm{frozen}}\gets \mathrm{diag}(\Sigma_t)\setminus \Lambda$  
20: Reconstruct with learned values:  
21:  $\Delta_{\mathrm{t}}\gets U_{\mathrm{t}}\mathrm{diag}(\Lambda ,\mathbf{s}_{\mathrm{frozen}})V_{\mathrm{t}}^{\top}.$  
22: return  $\Delta_{\mathrm{t}}$

Therefore, for each source task matrix  $\Delta_{i}$ , we perform SVD to decompose it into a set of orthogonal components. Each component is a triplet  $(\mathbf{u}_{i,j},\sigma_{i,j},\mathbf{v}_{i,j}^{\top})$ , where  $j$  is the component index for a given task  $i$ . Consequently, we propose an aggregation strategy based on a global ranking of all components from all source task matrices. We then select the Top-K components with the highest singular values to construct the transfer basis:

$$
\mathcal {B} = \left\{\left(\mathbf {u} _ {k}, \sigma_ {k}, \mathbf {v} _ {k} ^ {\top}\right) \right\} _ {k = 1} ^ {K}, \text {w h e r e} \sigma_ {k} \geq \sigma_ {k + 1}, \forall k
$$

Finally, the merged task matrix,  $\Delta_{m}$ , is synthesized by summing the Top- $K$  selected rank-one components:

$$
\Delta_ {m} = \sum_ {k = 1} ^ {K} \mathbf {u} _ {k} \sigma_ {k} \mathbf {v} _ {k} ^ {\top}.
$$

By prioritizing these high-magnitude components, we aim to build a new, effective pre-trained state for any unknown downstream task. We empirically validate the quality of the merged model and the component selection strategy against alternatives in our ablation studies.

# STAGE 2: TARGET TASK ADAPTATION.

In the second stage, the merged knowledge  $\Delta_{m}$  is adapted to the specific target task. We define the final target task vector  $\Delta_{t}$  as a function of  $\Delta_{m}$  and a small set of learnable parameters  $\Lambda$  that minimize the cross-entropy loss  $\mathcal{L}$  on the target dataset:

$$
\Lambda^ {*} = \underset {\Lambda} {\operatorname {a r g m i n}} \mathbb {E} _ {(x, y) \in D _ {\mathrm {t}}} [ \mathcal {L} (f (x; \theta_ {\text {p r e}} + \Delta_ {\mathrm {t}} (\Lambda)), y) ]
$$

For a parameter-efficient adaptation, we apply gradient-based learning exclusively to the top-  $N$  singular values of  $\Delta_t$ , which constitute the set  $\Lambda$ . The remaining singular vectors and less significant components are kept frozen. The resulting full model parameters for the target task are  $\theta_{\mathrm{t}} = \theta_{\mathrm{pre}} + \Delta_{\mathrm{t}}(\Lambda)$  and the full, step-by-step process is formalized in Algorithm ?? and Figure 2.

The synthesized matrix  $\Delta_{m}$  represents a rich but intermediate consolidation of knowledge from multiple source tasks. To transform this aggregation into a computationally stable and effective basis for adaptation, we re-parameterize it using a final SVD. This procedure,  $\Delta_{m} \rightarrow U_{t}\Sigma_{t}V_{t}^{\top}$ , serves a dual purpose. First, it constructs a new set of orthogonal vectors,  $U_{t}$  and  $V_{t}$ , creating a decorrelated basis that optimally represents the merged transformation in the sense of the Frobenius norm. Second, it yields a new diagonal matrix  $\Sigma_{t}$ , whose values reflect the true importance of the components within the combined matrix  $\Delta_{m}$  and also serve as the isolated set of learnable parameters,  $\Lambda$ , for the subsequent fine-tuning.

# 4 RESULTS

# 4.1 EXPERIMENTAL SETUP

To evaluate the performance, scalability, and robustness of our method, we benchmark it against the recent state-of-the-art method, aTLAS, which serves as our baseline. The experimental framework is based on diverse image classification tasks, including texture recognition (DTD), satellite imagery (EuroSAT), and fine-grained visual categorization (Flowers102). The experimental setup employs a leave-one-out protocol. For each target task, we incrementally aggregate knowledge assets by varying the number of source task vectors from one up to the maximum of  $T - 1$  in a fixed, predefined sequence. By default, we use the pre-trained Vision Transformer (ViT-B-32) variant of the CLIP model Radford et al. (2021). Our primary performance metric is the Top-1 accuracy evaluated on the test set of each target task. All results are presented under a matched number of trainable parameters and within the range used by aTLAS method. Our evaluation adapts the comprehensive benchmark, publicly released task vectors, and training protocols established by the authors of aTLAS to ensure a direct and fair comparison. For each target task adaptation, the fine-tuning process utilizes the complete, standard training set. To provide a one-to-one comparison, we adopted the same training configuration used for the aTLAS baseline and ran all its experiments within this consistent framework. Specifically, each adaptation runs for 10 epochs with a learning rate of  $10^{-1}$ . All setup details are provided in the Appendix.

# 4.2 PERFORMANCE AND EFFICIENCY GAINS OVER ATLAS

For each target task, we incrementally build the merged task vector,  $\Delta_{\text{target}}$ , by aggregating an increasing number of source task vectors. For example, a single model synthesized from 16 source vectors is then independently fine-tuned 21 times - once for each distinct target task as illustrated in Figure 10. This entire process is repeated for every aggregation level, and the outcomes are averaged to produce the final performance curves. The parameter budgets  $N$  of  $10\%$ ,  $20\%$ , and  $40\%$  are determined by the percentage of trainable singular values selected from each task matrix; their sum across all matrices results in total trainable parameter counts of approximately 3.6k, 7.3k, and 14.7k, respectively, in the ViT-B-32 version. The results demonstrate that our approach outperforms aT-LAS across the entire spectrum of source task quantities on both the ViT-B-32 (illustrated in Figure 4) and ViT-L-14 architectures (see Figure 12 in the Appendix).

![](images/8702c6908aa3fd8e567e11e24961ce041e8e10e54a57dd579ca29602b12502b3.jpg)  
Figure 3: The comparison of the merged models, AXIS and aTLAS, utilizing 16 task vectors across all target tasks yielded an average of  $78.42\%$  for AXIS and  $75.13\%$  for aTLAS.

![](images/073f635177cec20657e3cdb21e5f30b50f183259694e964b616735da06526176.jpg)  
Figure 4: Performance comparison with the aTLAS varying the number of trainable parameters with the ViT-B-32 architecture. Each point represents a model configuration that was independently adapted to all target tasks. The plotted value is the mean performance across these tasks.

![](images/e1189fe2cb51aec2e81282dccd5960e6eb4da9b14f2cedacc4cb38bc05b1863a.jpg)  
Figure 5: Scalability analysis for ViT-L-14 architecture with  $N = 10\%$  trainable parameters. As the number of source task vectors increases, the runtime and memory costs of aTLAS scale near-linearly. In contrast, our AXIS framework maintains a constant computational footprint.

Our method shows higher parameter efficiency, as illustrated in Figure 1. The figure compares AXIS with aTLAS, showing that for any given parameter budget, our approach yields higher average accuracy. Furthermore, the noticeably smaller shaded area for AXIS indicates a lower standard deviation, highlighting that our aggregation mechanism is more stable and less sensitive to variations in the number of source task vectors used.

Memory and Runtime Scalability. A key advantage of our method is its significantly lower computational overhead compared to baselines like aTLAS. The memory and runtime costs of aTLAS scale near-linearly with the number of source models, as it learns a distinct coefficient for each of the  $T$  source tasks across every layer and parameter partition  $P$  during the fine-tuning process. This means that all source task vectors must be present in memory throughout the entire adaptation phase for a new target task.

In stark contrast, AXIS decouples the process into two distinct stages. The first stage, knowledge aggregation, is a fast, one-time operation. It efficiently processes all  $T - 1$  source task vectors using SVD and consolidates them into a single, fixed-size merged matrix,  $\Delta_{m}$ . The subsequent, and most computationally intensive, fine-tuning stage operates only on this compact  $\Delta_{m}$ . As a result, the memory footprint and runtime of the adaptation phase remain constant, regardless of the number of source models initially aggregated. This design choice not only makes our approach more scalable but also significantly reduces the resources required for fine-tuning, as is illustrated in Figure 5

# 4.3 ROBUSTNESS TO NOISE AND SPARSITY IN SOURCE PARAMETERS

To evaluate the robustness of our method with unreliable, uncertain Li et al. (2025) or compressed Iurada et al. (2025); Li et al. (2025) source task vectors, we designed two specific scenarios. The first simulates contamination from a single, low-quality source, for instance, due to training instabilities. The second scenario evaluates how effectively these approaches leverage knowledge when all source task vectors are heavily pruned. Both investigations explore the method's capacity to merge a more diverse and challenging spectrum of models, expanding its practical applicability.

We formed aggregations of source task vectors of varying sizes, ranging from three to eight, to demonstrate the effect of a single faulty source. In each aggregation, one task vector was intentionally corrupted, while the others remained intact. The corruption was applied by adding zero-mean Gaussian noise to the weights of an original task vector. To ensure a significant level of disruption, the standard deviation of the noise was scaled to  $50\%$  of the Frobenius norm of that task matrix  $(\sigma = 0.5 \cdot ||\Delta_i||_F)$ . The results illustrated in Figure 6 demonstrate that while both methods experience some performance degradation in the presence of a corrupted source, the impact on our method

![](images/e50f93fae47d7974175a47bd071db1b5908f8aed61382e0150138f8d5805ff7d.jpg)  
Figure 6: Robustness to altered source task vectors. The plot compares performance under two distinct perturbation scenarios, with results averaged across all 21 target tasks. Our method AXIS demonstrates substantially higher resilience to both scenarios compared to aTLAS.

![](images/3409c2e7b26c2363e2304da05b8428cbfc384072997f5a7ce0dacf7602bb09b7.jpg)  
Figure 7: The chart illustrates the average accuracy across all target tasks. Results indicate that our approach, AXIS, outperforms the baselines even under challenging conditions where input information is partially hidden, with up to  $50\%$  of patches masked.

is significantly less pronounced. This indicates a more robust knowledge transfer mechanism. We observe that our SVD-based selection process, by focusing on components with the highest singular values, is less susceptible to the unstructured perturbations introduced into a single source vector.

To assess the robustness of our method from a compression perspective, each of the source task vectors underwent magnitude-based pruning. We applied a high-level ratio, ensuring that specialized knowledge was not catastrophically degraded. The subsequent analysis in Fig 6 suggests that our approach can more effectively leverage the knowledge contained within highly sparse task vectors, showcasing a distinct advantage in utilizing compressed knowledge.

# 4.4 ROBUSTNESS TO INPUT DATA DEGRADATION

Building on findings that merging models fine-tuned with distinct hyperparameters on the same task leads to greater stability under distribution shifts Wortsman et al. (2022a;b), we explore whether aggregating knowledge from multiple, diverse models, each fine-tuned with the same set of hyperparameters, can similarly construct a more robust representation. For this experiment, the AXIS and aTLAS models were built by aggregating the complete set of  $T - 1$  source task vectors and fine-tuning them for each target task.

The model's accuracy on images with randomly omitted patches can serve as a direct test, which was previously used to measure model robustness Paul & Chen (2022) or ability to perform prediction with partial information Pardyl et al. (2025), providing unique insight into a model's internal representation, as this form of robustness is often less correlated with baseline model performance than other image perturbations Malik et al. (2025). To ensure a fair comparison, a fixed seed guarantees that all methods are evaluated using the same masked patches for each dropout level. In Figure 7, AXIS shows resilience when almost all complete information is available, and degrades more slowly as input degradation becomes more severe. This capability is essential for real-world scenarios with incomplete data and follows prior research aimed at improving model resilience to partial visual information Liu et al. (2023); Tang et al. (2022) (see Table 4). Additionally, we demonstrate better robustness capabilities of AXIS than aTLAS against a set of 12 common image corruptions Hendrycks & Dietterich (2019) with five severity levels in the Appendix.

# 5 ANALYSIS

To provide a deeper understanding of our method's mechanics, we conduct a series of ablation studies targeting its key elements.

![](images/83eca7acbca42175043c5964f55fbd6d0f08ca348e944aa3c0733256ade275db.jpg)  
Figure 8: Performance comparison with competing methods, including PEFT variants. The proposed merge-and-tune paradigm achieves a more efficient performance-parameter trade-off.

<table><tr><td>Method</td><td>N = 10%</td><td>N = 20%</td><td>N = 40%</td></tr><tr><td>DARE + Stage 2</td><td>78.09 ± 0.06</td><td>79.69 ± 0.04</td><td>80.77 ± 0.09</td></tr><tr><td>Average + Stage 2</td><td>78.19 ± 0.15</td><td>79.45 ± 0.15</td><td>79.43 ± 0.70</td></tr><tr><td>TIES + Stage 2</td><td>77.39 ± 0.03</td><td>78.99 ± 0.05</td><td>80.27 ± 0.05</td></tr><tr><td>TSV-M + Stage 2</td><td>76.41 ± 0.05</td><td>78.69 ± 0.07</td><td>80.41 ± 0.11</td></tr><tr><td>aTLAS</td><td>75.50 ± 0.03</td><td>75.93 ± 0.44</td><td>77.66 ± 0.05</td></tr><tr><td>AXIS</td><td>78.46 ± 0.04</td><td>79.93 ± 0.11</td><td>81.13 ± 0.07</td></tr></table>

Table 1: Performance comparison with aTLAS and merging methods when followed by our Stage 2 adaptation. While the best results are obtained by AXIS, the adaptation mechanism itself is a potent and versatile tool for refining diverse multi-capability models. All results are averaged over 3 seeds.

# 5.1 BROAD COMPARISON

To demonstrate the advantages of our approach, we compare it with different finetuning methods, in particular with PEFT methods. This includes the widely-adopted LoRA Hu et al. (2022) and its enhanced variant LoRA-XS Balazy et al. (2024). Additionally we try to further adapt the pre-trained weights as a single task vector (TV). As the Fig 8 shows, our method efficiently outperforms these techniques, effectively reusing already finetuned weights.

We further take inspiration from model merging techniques and ask the question whether a general, multi-task model serve as an effective knowledge base for our Stage 2 adaptation? To test this hypothesis, we substitute our AXIS aggregation with several established multi-task merging techniques, such as DARE Yu et al. (2024), TIES-Merging Yadav et al. (2023), TSV-M Gargiulo et al. (2025) and simple averaging, treating their merged weights as alternative initializations. As the results in Table 1 demonstrate, these multi-task models indeed form a potent foundation for our adaptation mechanism, however slightly below the performance of the AXIS method. This suggests that the Stage 2 is not rigidly dependent on a single aggregation method but can effectively refine knowledge from various merged, multi-capability models.

# 5.2 SCALABILITY AND PARAMETER SENSITIVITY OF AXIS

To assess the sensitivity of our method to the size of the transfer basis, we conducted an ablation study on the number of selected components,  $K$ . This sole hyperparameter directly controls the dimensionality of the aggregated knowledge consolidated into the merged task matrix,  $\Delta_{m}$ . In this experiment, we varied the value of  $K$  used in our top components aggregation strategy, where components from all source tasks are globally ranked by their singular values before the top  $K$  are selected to form the transfer basis. Our default choice of  $K = 76$  (approximating  $10\%$  of each layer's rank) proves to be a robust heuristic. The plot demonstrates that performance remains high, with the drop being less than  $1.5\%$  even for large  $K$  (Figure 9). overall, we find that limiting the  $K$  to be less than  $20\%$  of total parameters provides robust results. We hypothesise that including additional components may introduce more task-specific details, which are not necessarily important for the target task.

Additionally, we evaluate how scaling the total number of trainable parameters  $(N)$  affects model performance (Fig. 10). Overall, we observe that increasing the number of parameters leads to higher final accuracy. However, the improvements begin to diminish once  $N$  exceeds  $60\%$ . Thus, N serves as a control parameter that balances computational requirements and final performance.

# 5.3 COMPONENTS SELECTIONS STRATEGY

To evaluate the quality of component aggregation, we test three selection criteria from a global pool of all aggregated SVD components. We compare the impact of selecting components with the highest singular values (top components), the lowest (bottom components), and those chosen arbitrarily (arbitrary components). The results of this comparison are presented in Table 2, which

![](images/e8765dccf41bdb7f55402d0f700b2a84bb42c06f1db0a13d0f7576fb9c62f590.jpg)  
Figure 9: Performance sensitivity to the number of aggregated components  $K$ . We vary the number of globally top-ranked SVD components used to construct the transfer basis and report the average accuracy. AXIS is robust to changes in K and the best results are obtained for K being smaller than 20% of total number of components - our default choice is using 10%, K = 76.

![](images/403660c299e30dba282d45318c298086c30608ba594fb7ecbc3127cd1e0af651.jpg)  
Figure 10: The AXIS scales consistently with the number of trained parameters  $(N\%)$ , showing improved performance as  $N$  increases, with gains tapering off beyond  $60\%$ .

<table><tr><td>N (%)</td><td>Top</td><td>Arbitrary</td><td>Bottom</td></tr><tr><td>10</td><td>78.46 ± 0.04</td><td>77.83 ± 0.04</td><td>77.56 ± 0.02</td></tr><tr><td>20</td><td>79.93 ± 0.11</td><td>79.79 ± 0.03</td><td>79.81 ± 0.05</td></tr><tr><td>40</td><td>81.13 ± 0.07</td><td>81.17 ± 0.08</td><td>81.13 ± 0.04</td></tr></table>

Table 2: Performance comparison of different SVD component selection strategies within the AXIS framework, demonstrating their comparable effectiveness.

![](images/cc7c6231f032ca5d68f6182a1871f90ee7f9925904b427a4d7384a2108704e73.jpg)  
Figure 11: Skipping the final SVD orthogonalization results in a decline in performance, especially when combining a moderate number of task vectors.

indicates that the top components strategy yields the best performance. While selecting the top components components yields the highest accuracy, this advantage is most pronounced at lower parameter budgets. As the number of trainable parameters increases, the performance of all three strategies converges, suggests that the importance of the initial component selection decreases as the model is given more trainable parameters.

# 5.4 STABILIZING THE TRANSFER BASIS

Instead of performing the final SVD re-parameterization, the layer's weights were reconstructed directly from the aggregated components  $\Delta_{m}$ . For our primary strategy of top component selection, this omission results in significant performance degradation when a moderate number of task vectors are aggregated (Figure 11). The results in Table 6 confirm that the final SVD orthogonalization step provides a performance uplift for the other selection criteria (see Table 6).

# 6 CONCLUSION

We presented AXIS, a framework that addresses multi-source knowledge transfer through the extraction, aggregation, and adaptation of useful knowledge for the target task. The resulting merged model provides a promising performance baseline in a zero-shot setting, confirming the high quality of the consolidated components. Furthermore, the framework enables efficient final adaptation while demonstrating robustness to degradations at both the parameter and input levels. The effectiveness of this entire process, however, relies on the fundamental assumption of a common architecture and a shared pre-trained origin.
