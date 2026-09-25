---
layout: distill
title: "Quantifying and Using Uncertainty in Deep Learning-based UAV Navigation"
date: 2025-11-03
tags: Autonomous Navigation Uncertainty BayesianDeepLearning Systems
description: "Quantifying and using uncertainty in Bayesian deep learning systems for robust UAV navigation"
# appendix: true
citation: true
tikzjax: true
pretty_table: true
published: true
thumbnail: assets/img/posts/2025-10-25-UQ-BDL-UAV-System/UAV-Nav-AirSim.gif
bibliography: 2025-10-25-UQBDLSys.bib
authors:
  - name: Fabio Arnez
    affiliations:
      name: Université Paris-Saclay, CEA, List
# toc:
#   sidebar: left
toc:
  - name: Introduction
  - name: The Navigation Task and Architecture Overview
    subsections:
      - name: Learning Perception Representations
      - name: Learning a Probabilistic Control Policy
      - name: Autonomous Navigation Overview
  - name: Quantifying Uncertainty in the DNN-based Navigation System
    subsections:
      - name: Uncertainty From Perception Representations
      - name: Handling Input Uncertainty In The Control Policy
  - name: Navigation Performance Evaluation
    subsections:
      - name: Navigation Models Setup
      - name: Navigation Performance Results
  - name: Leveraging System Uncertainty for Better Navigation Performance
    subsections:
      - name: Understanding the System Components' Predictive Uncertainty
      - name: Uncertainty-Aware Control Strategy
      - name: Why Does This Work?
      - name: Results With the Uncertainty-Aware Control Strategy
  - name: Conclusion

_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Introduction

Autonomous systems, like Unmanned Aerial Vehicles (UAVs) and self-driving cars, increasingly rely on Deep Neural Networks (DNNs) to handle critical functions within their navigation pipelines (perception, planning, and control). While DNNs are powerful, deploying them in safety-critical roles demands that they accurately express their confidence in predictions. This is where Bayesian Deep Learning (BDL) <d-cite key="gal2016dropout,lakshminarayanan2017simple"></d-cite> comes in, offering a principled framework to model and capture uncertainty.
However, if the Bayesian approach is followed, ideally all the components in the navigation pipeline (perception, planning, control) should use BDL to enable uncertainty propagation, so that the output of the system reflects the uncertainty of the system as a whole <d-cite key="mcallister2017concrete"></d-cite>. Uncertainty propagation is challenging, as it requires BDL components to admit uncertainty information as an input, to account for the uncertainty coming from the preceding components.

In this post, we describe how to capture, propagate, and use uncertainty along a navigation pipeline of BDL components, summarizing our work in <d-cite key="arnez2022towards,arnez2022quantifying,arnez2023navigation"></d-cite>. We assess how uncertainty quantification throughout the system impacts the navigation performance of a UAV that must fly autonomously through a set of gates disposed in a circle within a simulated environment (AirSim). The post is organized around three research questions: whether capturing uncertainty along the whole pipeline improves the navigation performance and robustness (<a href="#rq:rq1">RQ-1</a>); how this uncertainty shows up in the predictions of the fully Bayesian pipeline in challenging situations, and what limits its gains (<a href="#rq:rq2">RQ-2</a>); and whether the system's own uncertainty can be used at runtime to make better control decisions (<a href="#rq:rq3">RQ-3</a>).

## The Navigation Task and Architecture Overview

The goal of the autonomous agent (i.e., UAV) is to navigate through a set of gates with unknown locations disposed in a circular track in the AirSim simulator, as presented in <a href="#fig:uav-track">Figure 1</a>.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <a id="fig:uav-track"></a>
        {% include figure.liquid loading="eager" path="assets/img/posts/2025-10-25-UQ-BDL-UAV-System/circular_track.png" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>
<div class="caption">
    Figure 1: UAV circular track in AirSim.
</div>

We consider a minimalistic end-to-end deep learning-based navigation architecture to study uncertainty propagation and its use. Therefore, in our experiments, the autonomous navigation architecture consists of two neural network components, one for **perception** and the other for **control**, as presented in <a href="#fig:nav-arch">Figure 2</a>.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <a id="fig:nav-arch"></a>
        {% include figure.liquid loading="eager" path="assets/img/posts/2025-10-25-UQ-BDL-UAV-System/UAV-nav-arch.jpg" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>
<div class="caption">
    Figure 2: UAV autonomous navigation architecture.
</div>


To create an instance of the architecture above, we follow the approach presented in <d-cite key="bonatti2020learning"></d-cite>, where the perception component defines an encoder function  $$q_{\phi}:\mathcal{X} \rightarrow \mathcal{Z}$$ that maps the input image $$\mathbf{x}$$ to a rich low-dimensional representation  $$\mathbf{z} \in \mathbb{R}^{10}$$. Next, a control policy $$\pi_{w}: \mathcal{Z} \rightarrow \Upsilon$$ maps the compact representation $$\mathbf{z}$$ to velocity commands $$\Upsilon = \{\dot{x}, \dot{y}, \dot{z}, \dot{\psi}\} \in \mathbb{R}^{4}$$, corresponding to the desired linear and yaw velocities in the UAV body frame. These desired velocities are then sent to the UAV low-level controller, which is responsible for the UAV motion in the simulator. <a href="#fig:bonatti-nav-arch">Figure 3</a> shows the UAV navigation architecture proposed by Bonatti et al. <d-cite key="bonatti2020learning"></d-cite>, where the control policy is implemented using a multilayer perceptron (MLP), and the perception encoder is implemented using the encoder block of a variational autoencoder (VAE).

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <a id="fig:bonatti-nav-arch"></a>
        {% include figure.liquid loading="eager" path="assets/img/posts/2025-10-25-UQ-BDL-UAV-System/bonatti_nav_arch.png" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>
<div class="caption">
    Figure 3: The input image is encoded into a latent representation of the environment. A control policy acts on the lower-dimensional embedding to output the desired robot velocity commands.
</div>

In particular, Bonatti et al. <d-cite key="bonatti2020learning"></d-cite> employ a special type of VAE, a cross-modal VAE (CMVAE), that allows mixing two data modalities. In the CMVAE, besides reconstructing the input image, an additional network block is trained to predict the gate's pose (position and orientation in spherical coordinates) relative to the UAV camera, i.e., the additional network block performs a (supervised) regression task with the additional data modality (gate pose labels for each image), as presented in <a href="#fig:cmvae">Figure 4</a>.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
       <a id="fig:cmvae"></a>
        {% include figure.liquid loading="eager" path="assets/img/posts/2025-10-25-UQ-BDL-UAV-System/bonatti_percept_cmvae.png" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>
<div class="caption">
    Figure 4: Cross-Modal VAE: Each input image sample is encoded into a single latent space that can be decoded back into images, or transformed into another data modality such as the poses of gates relative to the UAV.
</div>

Moreover, the additional network block for predicting the gate pose is connected to the VAE in an unusual way. The regression block only uses the first four variables of the latent vector at the output of the CMVAE encoder, and each of these four latent variables is connected to a dedicated regressor for one of the predicted spherical coordinates (radius, polar, azimuth, yaw).
This imposes a stronger regularization on the latent space during training, since the first four latent variables receive an additional flow of error gradients corresponding to the pose prediction errors, forcing the disentanglement of these four latent variables, as presented in <a href="#fig:disentangled-representation">Figure 5</a>.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <a id="fig:disentangled-representation"></a>
        {% include figure.liquid loading="eager" path="assets/img/posts/2025-10-25-UQ-BDL-UAV-System/cm-vae-disentangled.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Figure 5: CMVAE disentangled representations: Changing the values from the first four latent vector variables allows us to control the gate attributes in the generated images.
</div>

### Learning Perception Representations

To build the perception component, we use the Dronet architecture <d-cite key="loquercio2018dronet"></d-cite> for the CMVAE encoder $$q_{\phi}$$. As mentioned before, additional constraints are imposed in the latent space to promote the learning of disentangled representations. For this purpose, the latent vectors $$\mathbf{z}$$ are treated differently by the decoder $$p_{\theta}$$ and the regression networks. The image decoder $$p_{\theta}$$ uses the whole latent vector $$\mathbf{z}$$, while the regression network $$p_{\rho}$$ uses only the first four elements $$z_{1:4}$$. Each element from $$z_{1:4}$$ has a dedicated network to predict a specific gate pose $$\xi$$ attribute. Moreover, the CMVAE loss function in Equation \eqref{eq:loss_cmvae_prob} has additional regularization coefficients that penalize the predictions of each network and the closeness to the imposed latent structure.

$$
\begin{equation}
    ({\phi^{*}, \theta^{*}, \rho^{*}}) = \underset{\phi, \theta, \rho}{\arg\min} \text{  } \mathcal{L}_{p}(\mathcal{D}_{p}; \phi, \theta, \rho)
\end{equation}
$$

$$
\begin{equation}
    \mathcal{L}_{p}(\mathcal{D}_{p}; \phi, \theta, \rho) = \mathcal{L}_{p}(\mathbf{x}, \xi; \phi, \theta, \rho)
\end{equation}
$$

$$
\begin{equation}
\label{eq:loss_cmvae_prob}
\begin{split}
\mathcal{L}_{p}(\mathcal{D}_{p}; \phi, \theta, \rho) = {} & \frac{\alpha_{p}}{2} {\big\Vert {\mathbf{x} - p_{\theta}(\hat{\mathbf{x}} \mid z)} \big\Vert}^{2} \\
& + \frac{\gamma_{p}}{2} {\big\Vert \xi - p_{\rho}(\hat{\xi}\mid z_{1:4}) \big\Vert}^{2} \\
& + \beta_{p} \; \mathbb{KL}\big( q_{\phi}(z \mid \mathbf{x}) \; \Vert \; \mathcal{N}(0, \mathbf{I})\big)
\end{split}
\end{equation}
$$

### Learning a Probabilistic Control Policy

Once the perception component is trained, we use only the encoder of the trained CMVAE to get a rich compact representation (latent vector) of the input image. The downstream control task (control policy $$\pi$$) uses an MLP network that operates on the latent vectors $$\mathbf{z}$$ at the output of the CMVAE encoder $$q_{\phi}$$ to predict UAV velocities. To this end, a probabilistic control policy network is added at the output of the perception encoder $$q_{\phi}$$, forming the UAV navigation stack. The probabilistic control policy network $$\pi_{w}(\Upsilon \mid \mathbf{z})$$ predicts the mean and the variance for each velocity command given a perception representation $$\mathbf{z}$$ from the encoder $$q_{\phi}$$, i.e., $$\Upsilon \sim \mathcal{N}\big(\mu_{w}(\mathbf{z}), \sigma^{2}_{w}(\mathbf{z})\big)$$, where
$$\Upsilon_{\mu} = \{\mu_{\dot{x}}, \mu_{\dot{y}}, \mu_{\dot{z}}, \mu_{\dot{\psi}}\}$$ and
$$\Upsilon_{\sigma^{2}} = \{\sigma^{2}_{\dot{x}}, \sigma^{2}_{\dot{y}}, \sigma^{2}_{\dot{z}}, \sigma^{2}_{\dot{\psi}}\}$$. For training the probabilistic control policy, we use imitation learning with a dedicated control dataset $$\mathcal{D}_c$$, and the heteroscedastic loss function <d-cite key="kendall2017uncertainties,lakshminarayanan2017simple"></d-cite> from Equation \eqref{eq:loss_ctrl_policy}.

$$
\begin{equation}
    {w}^{*} = \underset{w}{\arg\min} \text{  } \mathcal{L}_{c}(\mathcal{D}_{c}; w)
\end{equation}
$$

$$
\begin{equation}
    \mathcal{L}_{c}(\mathcal{D}_{c}; w) = \mathcal{L}_{\pi}(\Upsilon, z; w)
\end{equation}
$$

$$
\begin{equation}
\label{eq:loss_ctrl_policy}
\mathcal{L}_{c}(\mathcal{D}_{c}; w) = \frac{1}{2 \hat{\sigma}_{w}^{2}(z)} {\Vert \Upsilon_{i} - \hat{\mu}_{w}(z) \Vert}^{2} + \frac{1}{2} \log \hat{\sigma}_{w}^{2}(z)
\end{equation}
$$

Following the work by <d-cite key="bonatti2020learning"></d-cite>, during training, we freeze the weights of the perception encoder $$q_{\phi}$$ and update only the weights $$w$$ of the control policy network, as presented in the control component from <a href="#fig:uncertainty-nav-arch">Figure 7</a>.

### Autonomous Navigation Overview

After training the components of the minimalistic navigation architecture, we obtain an autonomous navigation flight that drives the UAV through the red gates, as shown in <a href="#fig:nav-airsim">Figure 6</a>.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <a id="fig:nav-airsim"></a>
        {% include figure.liquid loading="eager" path="assets/img/posts/2025-10-25-UQ-BDL-UAV-System/UAV-Nav-AirSim.gif" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>
<div class="caption">
    Figure 6: UAV autonomous navigation in AirSim simulator.
</div>

## Quantifying Uncertainty in the DNN-based Navigation System

### Uncertainty From Perception Representations

Although the CMVAE encoder $$q_{\phi}$$ employs Bayesian inference to obtain latent vectors $$\mathbf{z}$$, the CMVAE does not capture epistemic uncertainty since the encoder lacks a distribution over its parameters $$\phi$$. To capture uncertainty in the perception encoder, we follow prior work <d-cite key="daxberger2019bayesian,jesson2020identifying"></d-cite> that captures epistemic uncertainty in VAEs. We adapt the CMVAE to capture the posterior $$q_{\Phi}(\mathbf{z} \mid \mathbf{x}, \mathcal{D}_p)$$, as shown in Equation \eqref{eq:postEncoder}.

$$
\begin{equation}
\label{eq:postEncoder}
q_{\Phi}(\mathbf{z} \mid \mathbf{x}, \mathcal{D}_{p}) = \int{q(\mathbf{z} \mid \mathbf{x}, \phi) \; p(\phi \mid \mathcal{D}_{p}) \; d\phi}
\end{equation}
$$

To approximate Equation \eqref{eq:postEncoder}, we take a set $${\Phi = \{\phi_{m}\}^{M}_{m=1}}$$ of encoder parameter samples $$\phi_{m} \sim p(\phi \mid \mathcal{D}_{p})$$, to obtain a set of latent samples $$\{z_{m}\}^{M}_{m=1}$$ from the output of the encoder $$q_{\Phi}(\mathbf{z} \mid \mathbf{x}, \mathcal{D}_{p})$$. In practice, we modify the CMVAE by adding a dropout layer in the encoder. Then, we use Monte Carlo Dropout (MCD) <d-cite key="gal2016dropout"></d-cite> to approximate the posterior on the encoder weights $$p(\phi \mid \mathcal{D}_{p})$$, as shown for the perception component in <a href="#fig:uncertainty-nav-arch">Figure 7</a>. Finally, for a given input image $$\mathbf{x}$$, we perform $$M$$ stochastic forward passes (with dropout turned on) to compute a set of $$M$$ latent vector samples $$\mathbf{z}$$ at runtime.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <a id="fig:uncertainty-nav-arch"></a>
        {% include figure.liquid loading="eager" path="assets/img/posts/2025-10-25-UQ-BDL-UAV-System/uncertainty-drone-architecture.png" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>
<div class="caption">
    Figure 7: Uncertainty-aware UAV navigation architecture.
</div>

### Handling Input Uncertainty In The Control Policy

In BDL, downstream uncertainty propagation assumes that a neural network component is able to handle or admit uncertainty at the input. In our case, this implies that the neural network for control is able to handle the uncertainty coming from the perception component.
To do that, we use the BNN with latent variable inputs (BNN+LV) approach <d-cite key="henaff2018model,depeweg2017learning,depeweg2018decomposition"></d-cite> to propagate the uncertainty from perception to control in a principled way.
To capture the overall system uncertainty at the output of the controller, we compute the posterior predictive distribution for the target variable $$\Upsilon^{*}$$ associated with a new input image $$\mathbf{x}^{*}$$, as shown in Equation \eqref{eq:postPredDist} and Equation \eqref{eq:post_pred_dist_whole_system}:

$$
\begin{equation}
\label{eq:postPredDist}
p(\Upsilon^{*} \mid \mathbf{x}^{*}, \mathcal{D}_{c}, \mathcal{D}_{p}) =
        \iint{\pi(\Upsilon} \mid \mathbf{z}, \mathbf{w}) \;p(\mathbf{w} \mid \mathcal{D}_{c}) \;q_{\Phi}(\mathbf{z} \mid \mathbf{x}^{*}, \mathcal{D}_{p}) \;dz \;dw
\end{equation}
$$

$$
\begin{multline}
\label{eq:post_pred_dist_whole_system}
p(\Upsilon^{*} \mid \mathbf{x}^{*}, \mathcal{D}_{c},\mathcal{D}_{p}) = \\\int \int \int  \underbrace{\pi(\Upsilon^{*} \mid \mathbf{z}, \mathbf{w})}_\textit{control policy} \; p(\mathbf{w} \mid \mathcal{D}_{c}) \underbrace{q(\mathbf{z} \mid \mathbf{x}^{*}, \phi)}_{\textit{perception encoder}} p(\phi \mid \mathcal{D}_{p})\; d\phi \; dz \; dw
\end{multline}
$$

The integrals from the equations above are intractable, and we rely on approximations to obtain an estimation of the predictive distribution. The posterior $$p(\mathbf{w} \mid \mathcal{D}_{c})$$ is difficult to evaluate. Thus, we approximate the integral over the control policy weights $$\mathbf{w}$$ using an ensemble of neural networks <d-cite key="gustafsson2019evaluating"></d-cite>. As presented in <a href="#fig:uncertainty-nav-arch">Figure 7</a>, in practice, we train an ensemble of $$N$$ probabilistic control policies $${\pi}_{w_{n}}(\Upsilon \mid \mathbf{z}, w_{n})$$, with weights $$\{w_{n}\}^{N}_{n=1} \sim p(\mathbf{w} \mid \mathcal{D}_{c})$$, where each control policy $${\pi}_{w_{n}}$$ in the ensemble predicts the mean $$\mu_{w_{n}}(\mathbf{z})$$ and variance $$\sigma^{2}_{w_{n}}(\mathbf{z})$$ for each velocity command, i.e., $$\Upsilon \sim \mathcal{N}\big(\mu_{w_{n}}(\mathbf{z}), \sigma^{2}_{w_{n}}(\mathbf{z})\big)$$.

The integral over the latent representations $$\mathbf{z}$$ is approximated by taking a set of samples from the perception component latent space. In our previous work <d-cite key="arnez2021improving"></d-cite>, latent representation samples were drawn from the encoder output distribution $$\mathbf{z} \sim \mathcal{N}(\mu_{\phi},\sigma^{2}_{\phi})$$. Here, for simplicity, we directly use the $$M$$ samples obtained with MCD in the perception component, $$\{z_{m}\}^{M}_{m=1} \sim q_{\Phi}(\mathbf{z} \mid \mathbf{x}, \mathcal{D}_{p})$$, to take into account the epistemic uncertainty from the previous stage. Finally, the predictions we get from passing each latent vector $$\mathbf{z}$$ through each ensemble member are used to estimate the posterior predictive distribution in Equation \eqref{eq:postPredDist}. The predictive distribution $$p(\Upsilon^{*} \mid \mathbf{x}^{*}, \mathcal{D}_{c},\mathcal{D}_{p})$$ from Equation \eqref{eq:post_pred_dist_whole_system} takes into account the uncertainty from both system components.

From the control policy perspective, using multiple latent samples $$\mathbf{z}$$ can be seen as taking a better "picture" of the latent space (perception representation) to gather more information about the environment. Interestingly, we can also make a connection between our sampling approach and the works <d-cite key="tai2019visual,zhang2022memo"></d-cite> that sample the input space by performing translations and augmentations on input images to improve prediction robustness.

Finally, to control the UAV, we use the deep ensemble expected value of the predicted velocities, as suggested in the literature <d-cite key="lakshminarayanan2017simple,lee2019ensemble,nozarian2020uncertainty"></d-cite>. This means that we use $$\hat{\Upsilon}_{\mu} = \frac{1}{NM}\sum_{n=1}^{N}\sum_{m=1}^{M} \hat{\mu}_{w_{n}}(\mathbf{z}_{m})$$, i.e., the average of the predicted means $$\hat{\mu} = [\hat{\mu}_{\dot{x}}, \hat{\mu}_{\dot{y}}, \hat{\mu}_{\dot{z}}, \hat{\mu}_{\dot{\psi}}]$$ over all ensemble members and latent samples. These predicted velocities represent the desired (reference) velocities that are passed to AirSim's low-level control through its API. As we will see later, this seemingly innocuous choice turns out to be critical.

## Navigation Performance Evaluation

The goal of the UAV is to navigate through a set of gates with unknown locations, forming a circular track. In AirSim, a track is entirely defined by a set of gates, their poses in the space, and the agent navigation direction. For perception-based navigation, the complexity of a track resides in the _"gate-visibility"_ difficulty <d-cite key="madaan2020airsim,song2021autonomous"></d-cite>, i.e., how well the UAV camera Field-of-View (FoV) captures the target gate.

We evaluate the navigation system using a circular track with eight equally spaced gates positioned initially at a radius of 8 m and constant height, as shown in <a href="#fig:track-without-noise">Figure 8</a>. A natural way to increase track complexity is by adding a random displacement to the position of each gate in the track, i.e., introducing operational domain shift (a factor that influences model predictive uncertainty). A track without random displacement in the gates has a circular shape. Gate position randomness alters the shape of the track, affecting the gate visibility, as presented in <a href="#fig:track-with-noise">Figure 9</a>, and therefore, shifted images are more likely to be generated, e.g., gates may not be visible or may be only partially visible, or multiple gates may be captured in the UAV FoV.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <a id="fig:track-without-noise"></a>
        {% include figure.liquid loading="eager" path="assets/img/posts/2025-10-25-UQ-BDL-UAV-System/track_without_noise.png" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>
<div class="caption">
    Figure 8: UAV navigation circular track without noise. Bird's-eye view (left), and the UAV view perspective (right).
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <a id="fig:track-with-noise"></a>
        {% include figure.liquid loading="eager" path="assets/img/posts/2025-10-25-UQ-BDL-UAV-System/track_with_noise.png" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>
<div class="caption">
    Figure 9: UAV navigation circular track with noise. Bird's-eye view (left), and the UAV view perspective (right).
</div>

To assess the system performance and robustness to perturbations in the environment, we generate new tracks by adding an offset with random noise to each gate radius and height. We specify the Gate Radius Noise (GRN) and the Gate Height Noise (GHN) with two levels of track noise, as follows:

$$
\begin{align*}
    \text{Noise level 1}
    \begin{cases}
        GRN \sim \mathcal{U}[-1.0, 1.0)\\
        GHN \sim \mathcal{U}[0, 2.0)
    \end{cases} & \;\;\;\;\;
    \text{Noise level 2}
    \begin{cases}
        GRN \sim \mathcal{U}[-1.5, 1.5)\\
        GHN \sim \mathcal{U}[0, 3.0)
    \end{cases}
\end{align*}
$$

We measure the system performance by the average number of gates passed in six different noisy tracks, where each navigation model has two trials on each track. A UAV mission considers a maximum of 32 gates, which is equivalent to 4 laps (8 gates/lap), and a trial ends as soon as the UAV fails to pass a gate.

With this experimental setup, we seek to answer the following research question:

> **RQ-1<a id="rq:rq1"></a>**: Can we improve the UAV navigation performance and robustness to perturbations of the environment by capturing and propagating uncertainty along the whole DL-based navigation architecture?

### Navigation Models Setup

__Navigation Models Training Datasets.__ We use two independent datasets for each component in the navigation pipeline. The perception CMVAE uses a dataset ($$\mathcal{D}_p$$) of 300k images where a gate is visible and gate-pose annotations are available. The control component uses a dataset ($$\mathcal{D}_c$$) of 17k images with UAV velocity annotations. $$\mathcal{D}_c$$ is collected by flying the UAV in a circular track with gates, using traditional methods for trajectory planning and control. The perception dataset is divided into 80% for training and the remaining 20% for validation and testing. The control dataset uses a split of 90% for training and the remaining 10% for validation and testing. In both cases, the image size is 64x64 pixels.

__Navigation Models Baselines.__ Ideally, we would expect a fully uncertainty-aware navigation architecture that controls the UAV with the expected value of the predicted velocities <d-cite key="lakshminarayanan2017simple,lee2019ensemble,nozarian2020uncertainty"></d-cite> to have a more robust and stable navigation performance. To see if this premise holds, we compare the navigation performance of different uncertainty-aware navigation architectures. The navigation models are listed in <a href="#table:nav-models">Table 1</a>, detailing the type of perception component, the number of latent representation samples (LRS), the type of control policy, and the number of control prediction samples (CPS) at the output of the system:

<!-- |    Model     | Perception Encoder |    LR Samples   |         Control Policy       |       CPS       |
| :----------: | :----------------: | :-------------: | :--------------------------: | :-------------: |
|     M1.      |        CMVAE       |        32       |    Ensemble Prob. (N = 5)    |      160        |
|     M2.      |        CMVAE       |         1       |    Ensemble Prob. (N = 5)    |       5         |
|     M3.      |        CMVAE       |        32       |     Deterministic (N = 1)    |       32        |
|     M4.      |        CMVAE       |        1        |    Ensemble Prob. (N = 1)    |        1        |
|     M5.      |      CMVAE-MCD     |        32       |    Ensemble Prob. (N = 5)    |      160        | -->

<table
  data-toggle="table"
  data-url="{{ 'assets/json/posts/2025-10-25-UQ-BDL-UAV-System/table_models.json' | relative_url }}">
  <a id="table:nav-models"></a>
  <thead>
    <tr>
      <th data-field="Model">Navigation Model</th>
      <th data-field="Perception Encoder">Perception Encoder</th>
      <th data-field="Latent Representation Samples">LRS</th>
      <th data-field="Control Policy">Control Policy</th>
      <th data-field="Control Prediction Samples">CPS</th>
    </tr>
  </thead>
</table>
<div class="caption">
    Table 1: UAV navigation models.
</div>

In <a href="#table:nav-models">Table 1</a>, models $$\mathcal{M}_1$$ to $$\mathcal{M}_3$$ partially capture uncertainty in the pipeline since they use a deterministic perception component (CMVAE), i.e., the encoder weights have no distribution and the perception epistemic uncertainty is not captured; their latent representation samples are drawn from the CMVAE encoder output distribution $$\mathbf{z} \sim \mathcal{N}(\mu_{\phi},\sigma^{2}_{\phi})$$, as in our previous work <d-cite key="arnez2021improving"></d-cite>. For the control component, $$\mathcal{M}_1$$ and $$\mathcal{M}_2$$ take 32 and 1 LRS, respectively, and use the samples later with an ensemble of 5 probabilistic control policies capturing epistemic and aleatoric uncertainty. $$\mathcal{M}_3$$ uses 32 LRS, and its control component is completely deterministic. Finally, $$\mathcal{M}_4$$ represents our fully Bayesian navigation pipeline<d-footnote>In the ICRA 2022 workshop paper and in the PhD thesis, this fully Bayesian navigation pipeline is named M0; the model named M4 there (a single probabilistic control policy) is not included in this post.</d-footnote>, where the perception component captures epistemic uncertainty using MCD with 32 forward passes for each input to get 32 latent representation predictions. To ease the computation, perception predictions are directly used as latent variable samples in downstream control. The control component uses an ensemble of 5 probabilistic control policies, obtaining 160 control prediction samples. In this first experiment, all models control the UAV with the same decision strategy: the expected value (mean) of their control prediction samples.

### Navigation Performance Results

<a href="#table:nav-models-performance">Table 2</a> shows the navigation performance of each model. Moreover, the videos below show a qualitative performance comparison of the models in the UAV navigation task, on one of the noisy tracks.

<table
  data-toggle="table"
  data-url="{{ 'assets/json/posts/2025-10-25-UQ-BDL-UAV-System/table_performance.json' | relative_url }}">
  <a id="table:nav-models-performance"></a>
  <thead>
    <tr>
      <th data-field="Model" data-halign="center" data-align="center">Navigation Model</th>
      <th data-field="Noise Level 1" data-halign="center" data-align="center">Gates Passed @ Track Noise Level 1</th>
      <th data-field="Noise Level 2" data-halign="center" data-align="center">Gates Passed @ Track Noise Level 2</th>
    </tr>
  </thead>
</table>
<div class="caption">
    Table 2: Navigation performance of the UAV models (average number of gates passed, out of 32), using the expected value (mean) of each model's control prediction samples.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0 text-center">
        {% include video.liquid path="https://www.youtube.com/embed/e9oLOo0JJRg" class="img-fluid rounded z-depth-1" %}
        <div class="caption">UAV navigation model M1 (ensemble mean)</div>
    </div>
    <div class="col-sm mt-3 mt-md-0 text-center">
        {% include video.liquid path="https://www.youtube.com/embed/vepmyzaIvzE" class="img-fluid rounded z-depth-1" %}
        <div class="caption">UAV navigation model M2 (ensemble mean)</div>
    </div>
</div>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0 text-center">
        {% include video.liquid path="https://www.youtube.com/embed/m5nH3nARsL0" class="img-fluid rounded z-depth-1" %}
        <div class="caption">UAV navigation model M3 (mean)</div>
    </div>
    <div class="col-sm mt-3 mt-md-0 text-center">
        {% include video.liquid path="https://www.youtube.com/embed/j82R279Kbio" class="img-fluid rounded z-depth-1" %}
        <div class="caption">UAV navigation model M4 (ensemble mean)</div>
    </div>
</div>

Two observations stand out from <a href="#table:nav-models-performance">Table 2</a>. First, learning to predict uncertainty in the control component helps: at noise level 1, the models with an ensemble of probabilistic control policies ($$\mathcal{M}_1$$, $$\mathcal{M}_2$$, and $$\mathcal{M}_4$$) pass roughly twice as many gates as $$\mathcal{M}_3$$, whose control component is deterministic. Interestingly, at noise level 2, $$\mathcal{M}_3$$ passes more gates than $$\mathcal{M}_2$$ (5.0 vs. 4.0), since sampling 32 latent representations from the noisy perception encoding adds diversity to its control predictions <d-cite key="arnez2023navigation"></d-cite>. Second, the fully Bayesian architecture $$\mathcal{M}_4$$ is the best model at both noise levels, but its advantage at noise level 1 is small: 19.77 gates vs. 17.67 for $$\mathcal{M}_1$$ and 17.33 for $$\mathcal{M}_2$$, even though $$\mathcal{M}_4$$ requires 32 stochastic forward passes through the perception encoder and 160 control prediction samples per input image, while $$\mathcal{M}_2$$ requires a single perception forward pass and 5 control predictions. At noise level 2, its lead is larger in relative terms (9.22 gates vs. at most 6.0 for the other models), yet all models, including $$\mathcal{M}_4$$, pass fewer than a third of the 32 gates.

Based on these results, we can answer <a href="#rq:rq1">RQ-1</a> with a qualified yes: capturing and propagating uncertainty along the navigation architecture improves the UAV performance and robustness to perturbations of the environment, and the fully Bayesian architecture $$\mathcal{M}_4$$ is the most robust one. However, the gains are modest compared with its computational cost, and the performance of all models collapses under stronger perturbations. This situation leads us to question the benefit of the fully Bayesian architecture, and more precisely, whether something prevents it from exploiting the uncertainty it captures. Naturally, our next question is:

<!-- Naturally, the next question is: **What is the reason for this suboptimal behavior in the full uncertainty-aware (Bayesian) architecture $\mathcal{M}_4$?** -->
> **RQ-2<a id="rq:rq2"></a>**: How does the uncertainty propagated along the fully Bayesian architecture $$\mathcal{M}_4$$ manifest in its predictions when the UAV faces a challenging situation, and why does it translate into only modest performance gains?

## Leveraging System Uncertainty for Better Navigation Performance

### Understanding the System Components' Predictive Uncertainty

To understand what is going on during the UAV mission, let's probe the control predictions of the fully Bayesian architecture $$\mathcal{M}_4$$ when facing one of the situations that arise when adding noise to the tracks, as shown in <a href="#fig:track-with-noise">Figure 9</a>.
In particular, consider the double-gate situation from <a href="#fig:double-gate-situation">Figure 10</a>, where two gates are captured in the UAV FoV, and the corresponding predictions of the control component in <a href="#fig:double-gate-preds-vy-vyaw">Figure 11</a>:

<div class="row mt-3">
<a id="fig:double-gate-situation"></a>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/2025-10-25-UQ-BDL-UAV-System/UAV-nav-task.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/2025-10-25-UQ-BDL-UAV-System/double-gate-situation.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Figure 10: Double-gate situation in the UAV field-of-view caused by noisy circular tracks (left). UAV field-of-view and the corresponding input image for the DNN in the double-gate situation (right).
</div>

<div class="row mt-3">
<a id="fig:double-gate-preds-vy-vyaw"></a>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/2025-10-25-UQ-BDL-UAV-System/m4_ensemble_mem_pred_lateral_mean.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/2025-10-25-UQ-BDL-UAV-System/m4_ensemble_mem_pred_angular_mean.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Figure 11: Densities of the predicted means \(\hat{\mu}\) for the lateral velocity \(\dot{y}\) (left) and the yaw angular velocity \(\dot{\psi}\) (right), for each control ensemble member \(\pi\) of \(\mathcal{M}_4\), in the double-gate situation.
</div>

This controlled experiment reveals that the ambiguity introduced in the input image (two gates) is also reflected in the control actions predicted by the navigation architecture. The predicted control commands that move the UAV towards one of the two gates, i.e., $$\hat{\dot{y}}$$ (lateral velocity) and $$\hat{\dot{\psi}}$$ (yaw rate), present **multimodal distributions** (two peaks) for most ensemble members in the control component. In this image, both peaks have the same sign; they differ in how much the UAV should move sideways and rotate, i.e., in which of the two gates it should fly towards. These observations suggest that, for the double-gate input image, there are two possible control actions that the UAV can take to navigate through the gates, which is a clear indication of the uncertainty in the predicted control commands.

In the control component, assigning the same weight to each ensemble member can result in a sub-optimal ensemble mixture when facing input samples with some degree of ambiguity (e.g., double gates), since the predictions can be multimodal distributions, as presented in <a href="#fig:double-gate-preds-vy-vyaw">Figure 11</a>.
In the case of ambiguity in the input image and multimodal distributions in the predicted control actions, we can identify two high-risk scenarios for the control component predictions:

1. **Multimodality within an ensemble member**: the predictions of a single ensemble member are multimodal (e.g., bimodal), i.e., across the latent representation samples, the member predicts a movement towards the left gate and towards the right gate with similar probability.
2. **Disagreement among ensemble members**: the predictions from one ensemble member attempt to move the UAV towards the left gate, while the predictions from another ensemble member try to move it towards the right gate.

Simply using the expected value of the predictions, $$\hat{\Upsilon}_{\mu}$$, in both scenarios averages the two modes, and the resulting command can fall between them, in a lower-density region that corresponds to neither gate. For the UAV, this can translate into a straight movement towards the space between the two gates, with a potentially tragic result (e.g., a crash), or into ignoring the mission task (e.g., not passing through any gate).

This answers <a href="#rq:rq2">RQ-2</a>: the fully Bayesian architecture $$\mathcal{M}_4$$ does capture the ambiguity of these challenging situations in its predictive distribution, but the expected-value decision strategy discards this information precisely when it matters the most. This offers a plausible explanation for the modest gains of $$\mathcal{M}_4$$, and suggests that the main bottleneck is less how uncertainty is captured than how it is used, a hypothesis we test next. Hence, our last question is:

> **RQ-3<a id="rq:rq3"></a>**: Can we use the predictive uncertainty of the navigation system at runtime to make better control decisions, and improve the UAV navigation performance and robustness to perturbations of the environment?

### Uncertainty-Aware Control Strategy

To use the predictions of $$\mathcal{M}_4$$ better, a control decision-making strategy should address both high-risk scenarios above: it should pick a trustworthy source of predictions instead of weighting all ensemble members equally (scenario 2), and it should commit to one of the modes of the predictive distribution instead of averaging them (scenario 1). Our uncertainty-aware control strategy follows these two steps <d-cite key="arnez2022quantifying,arnez2023navigation"></d-cite>.

__Step 1: Choose the ensemble member with the lowest mutual information.__ We take inspiration from active learning for BDL, where Gal et al. <d-cite key="gal2017deep"></d-cite> and Kirsch et al. <d-cite key="kirsch2019batchbald"></d-cite> build an acquisition function using the mutual information between the model predictions for a given input sample and the model parameters. Intuitively, the acquisition function searches for the samples with the highest mutual information to retrain or update the model. Instead, in our approach, given the latent representation samples from the perception component, we choose at runtime the predictions (the density) of the ensemble member that **minimizes** the mutual information, for each velocity command $$\upsilon$$, as presented in Equation \eqref{eq:mi-argmin}:

$$
\begin{equation}
\label{eq:mi-argmin}
w^{*} = \underset{w \in \{w_{1}, \dots, w_{N}\}}{\arg\min} \; I(\upsilon; \mathbf{z}, w), \quad \forall \upsilon \in \Upsilon = \{\dot{x}, \dot{y}, \dot{z}, \dot{\psi}\}
\end{equation}
$$

In our navigation architecture, the mutual information $$I$$ is presented in Equation \eqref{eq:mi-entropy}, and Equation \eqref{eq:mi-kl} presents the mutual information using the KL divergence:

$$
\begin{equation}
\label{eq:mi-entropy}
I(\upsilon; \mathbf{z}, w) = H(\upsilon) - \mathbb{E}_{\mathbf{z} \sim q_{\Phi}(\mathbf{z} \mid \mathbf{x}, \mathcal{D}_{p})}\big[H(\upsilon \mid \mathbf{z}, w)\big]
\end{equation}
$$

$$
\begin{equation}
\label{eq:mi-kl}
I(\upsilon; \mathbf{z}, w) = \int D_{\mathrm{KL}}\big(\pi(\upsilon \mid \mathbf{z}, w) \,\Vert\, \pi(\upsilon)\big) \, q_{\Phi}(\mathbf{z} \mid \mathbf{x}, \mathcal{D}_{p}) \, d\mathbf{z}
\end{equation}
$$

To estimate the mutual information from Equation \eqref{eq:mi-kl}, we use the variational lower bound approximation from Poole et al. <d-cite key="poole2019variational"></d-cite>, and compute every KL divergence between two normal distributions. To this end, we first estimate the intractable marginal $$\pi(\upsilon)$$ with the latent representation samples $$\mathbf{z}_{i}$$, $$i = 1, \dots, M$$ (indexed by $$i$$ here, since $$m$$ denotes the marginal estimate), and the ensemble weights $$w_{n}$$, i.e., by passing the samples through each ensemble member, as presented in Equation \eqref{eq:mi-marginal}:

$$
\begin{equation}
\label{eq:mi-marginal}
\pi(\upsilon) \approx m(\upsilon; \mathbf{z}_{1:M}, w_{1:N}) = \frac{1}{NM} \sum_{n=1}^{N} \sum_{i=1}^{M} \pi(\upsilon \mid \mathbf{z}_{i}, w_{n})
\end{equation}
$$

Since this mixture is not normal (in the double-gate situation, it is multimodal), in practice we replace it with the normal distribution that has the same mean and variance, i.e., the ensemble expected value and variance of all $$NM$$ predictions, so that each KL divergence has a closed form. Then, replacing the intractable marginal $$\pi(\upsilon)$$ with $$m(\upsilon)$$ in Equation \eqref{eq:mi-kl}, we obtain the mutual information lower bound estimate in Equation \eqref{eq:mi-lb}<d-footnote>Strictly, this estimate is not guaranteed to bound the mutual information of a single ensemble member. The multi-sample bound of Poole et al. holds when the mixture is built from the same predictive distributions whose mutual information is estimated, whereas our marginal estimate pools the predictions of all ensemble members (and is approximated by a normal distribution). The estimate therefore also grows when a member disagrees with the rest of the ensemble, and we use it as a score to rank the members (see the next subsection).</d-footnote>. Finally, we use Equation \eqref{eq:mi-select} to choose the predictions of an ensemble member for each velocity command:

$$
\begin{equation}
\label{eq:mi-lb}
I(\upsilon; \mathbf{z}, w) \geq \mathbb{E}_{\mathbf{z}_{1:M}}\Big[\frac{1}{M} \sum_{i=1}^{M} D_{\mathrm{KL}}\big(\pi(\upsilon \mid \mathbf{z}_{i}, w) \,\Vert\, m(\upsilon)\big)\Big]
\end{equation}
$$

$$
\begin{equation}
\label{eq:mi-select}
w^{*} = \underset{w \in \{w_{1}, \dots, w_{N}\}}{\arg\min} \; \mathbb{E}_{\mathbf{z}_{1:M}}\Big[\frac{1}{M} \sum_{i=1}^{M} D_{\mathrm{KL}}\big(\pi(\upsilon \mid \mathbf{z}_{i}, w) \,\Vert\, m(\upsilon)\big)\Big], \quad \forall \upsilon \in \Upsilon
\end{equation}
$$

In practice, the expectation over $$\mathbf{z}_{1:M}$$ is replaced by the single set of $$M$$ latent samples drawn for the current image.

__Step 2: Decide how to act on the chosen predictions.__ Once an ensemble member is chosen, the UAV could act on its predictions by taking the expected value, by checking the empirical predictive distribution for the existence of modes, or by taking one specific prediction, e.g., the minimum or maximum predicted velocity. In our strategy, we look at the $$M$$ predicted means of the chosen member (one per latent sample, as in <a href="#fig:double-gate-preds-vy-vyaw">Figure 11</a>), and we use a different rule for each velocity command:

- **Forward velocity $$\dot{x}$$**: we choose a conservative behavior by selecting the lowest predicted forward velocity (the *MinVelX* variant). The *modal forward velocity* variant (*Mode*) uses instead a mode of the predicted forward velocities (the lowest one, if several modes exist).
- **Lateral and yaw velocities $$\dot{y}$$ and $$\dot{\psi}$$**: we check for the modes of the empirical distribution (a kernel density estimate) of the predicted velocities. If multiple modes exist, we choose the left-most (lowest-valued) mode, i.e., among the candidate modes, the UAV prefers the one that moves it further to the left and rotates it more counterclockwise.
- **Vertical velocity $$\dot{z}$$**: if multiple modes exist, we choose the lowest velocity mode, so that the UAV has a soft vertical movement when facing ambiguity in the predictions.

These rules can be seen as decisions coming from a higher-level decision-making component. We refer to the overall strategy as **MI-Mode**, since it uses the mutual information to choose the ensemble member and the modes of its predictions to act (in the videos below, MILB stands for Mutual Information Lower Bound).

### Why Does This Work?

The MI-Mode strategy uses exactly the same model, predictions, and uncertainty estimates as the expected-value strategy; only the decision rule changes. Here is the intuition behind each ingredient:

__A mean is not a safe summary of a multimodal prediction.__ As we saw above, averaging the two modes can produce a command between them, in a region that corresponds to neither gate. A mode, instead, corresponds to a coherent hypothesis about the scene (e.g., "fly through the left gate"): a command that the control policy actually predicts with high density for this image. Committing to one of the two gates is better than committing to neither of them.

__The chosen member reacts little to perception noise and stays close to the rest of the system.__ Each latent sample $$\mathbf{z}_{i}$$ is a plausible reading of the scene by the perception component. Ideally, the mutual information in Equation \eqref{eq:mi-entropy} measures how much the velocity predicted by an ensemble member depends on the latent sample it receives. Our estimate (Equation \eqref{eq:mi-select}) compares each of the member's $$M$$ predictions with a single normal distribution that summarizes the predictions of *all* members (their overall mean and variance). As a result, the score of a member grows with two quantities: how much its predictions change from one latent sample to another (i.e., how strongly it reacts to perception noise), and how far its average prediction lies from the average prediction of the whole system (i.e., how much it disagrees with the other members). The score also accounts for the predicted variances, penalizing a member whose confidence differs strongly from that of the system. The chosen member is the one for which these quantities are small: it gives nearly the same answer however the perception component reads the scene, and that answer stays close to what the system as a whole predicts. As an analogy, imagine asking five experts to judge 32 slightly different photographs of the same scene: we trust the expert whose answers are the most consistent across the photographs and the closest to the panel's overall opinion.

__A consistent tie-breaking rule turns a decision into a commitment.__ The UAV makes a new decision at every time step, with a new image. If the choice between two modes were arbitrary, the UAV could alternate between the left and the right gate in consecutive decisions, and end up with the same "neither gate" trajectory as the mean. A fixed preference (the left-most mode) makes consecutive decisions consistent, so the UAV commits to one gate. As the UAV turns towards it, the other gate tends to leave the FoV, and the ambiguity disappears.

__Slowing down buys time.__ Selecting the lowest predicted forward velocity makes the UAV fly cautiously. At a lower speed, the UAV collects more observations before reaching the gates, which gives it more chances to correct its trajectory, and reduces the consequences of a wrong decision. Moreover, since the minimum is taken over the $$M$$ predictions of the chosen member, the slowdown adapts to the uncertainty: when the latent samples agree, the predictions are concentrated and the minimum is close to the mean; when the scene is ambiguous, the predictions tend to spread out, and the minimum, hence the forward velocity, drops. The soft vertical movement from the $$\dot{z}$$ rule follows the same cautious principle.

It is worth noting that MI-Mode is an ad-hoc strategy: its rules are tailored to the gate noise range observed in our experiments, and there are no guarantees that it will work with higher levels of gate noise <d-cite key="arnez2023navigation"></d-cite>.

### Results With the Uncertainty-Aware Control Strategy

<a href="#table:nav-models-performance-mi">Table 3</a> compares the navigation performance of the fully Bayesian architecture $$\mathcal{M}_4$$ when using the expected value (mean) of its predictions and when using the MI-Mode control strategy (with the MinVelX forward velocity rule), under the same evaluation protocol as before.

<table
  data-toggle="table"
  data-url="{{ 'assets/json/posts/2025-10-25-UQ-BDL-UAV-System/table_performance_milb.json' | relative_url }}">
  <a id="table:nav-models-performance-mi"></a>
  <thead>
    <tr>
      <th data-field="Model" data-halign="center" data-align="center">Navigation Model</th>
      <th data-field="Control Strategy" data-halign="center" data-align="center">Control Strategy</th>
      <th data-field="Noise Level 1" data-halign="center" data-align="center">Gates Passed @<br>Track Noise<br>Level 1</th>
      <th data-field="Noise Level 2" data-halign="center" data-align="center">Gates Passed @<br>Track Noise<br>Level 2</th>
    </tr>
  </thead>
</table>
<div class="caption">
    Table 3: Navigation performance of the fully Bayesian architecture M4 with the expected-value (mean) and the MI-Mode (MinVelX variant) control strategies (average number of gates passed, out of 32).
</div>

With the MI-Mode strategy, $$\mathcal{M}_4$$ passes 28.88 gates on average at noise level 1 (vs. 19.77 with the ensemble mean), and 23.05 gates at noise level 2 (vs. 9.22), i.e., 2.5 times as many gates under the stronger perturbations, where ambiguous situations are more frequent. Since MI-Mode changes three things at once (the ensemble member, the use of modes instead of the mean, and a lower forward velocity), these results do not isolate the contribution of each rule. The videos below show $$\mathcal{M}_4$$ flying with the two variants of the strategy, which differ only in the forward velocity rule.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0 text-center">
        {% include video.liquid path="https://www.youtube.com/embed/wKUf5Fis4co" class="img-fluid rounded z-depth-1" %}
        <div class="caption">UAV navigation model M4 with the MI-Mode (MILB) strategy, MinVelX variant: lowest predicted forward velocity \(\dot{x}\)</div>
    </div>
    <div class="col-sm mt-3 mt-md-0 text-center">
        {% include video.liquid path="https://www.youtube.com/embed/4v-SlXpBl8Q" class="img-fluid rounded z-depth-1" %}
        <div class="caption">UAV navigation model M4 with the MI-Mode (MILB) strategy, modal forward velocity variant (Mode): mode of the predicted forward velocities \(\dot{x}\)</div>
    </div>
</div>

Therefore, we can answer <a href="#rq:rq3">RQ-3</a> positively: using the predictive uncertainty of the system at runtime, to choose which predictions to trust and how to act on them, substantially improves the navigation performance and robustness of the fully Bayesian architecture, without retraining any component.

## Conclusion

In this post, we described how to quantify, propagate, and use uncertainty along a minimalistic end-to-end DNN-based UAV navigation pipeline built on Bayesian Deep Learning. The perception component, a CMVAE augmented with Monte Carlo Dropout, produces a set of latent representation samples that reflects the epistemic uncertainty of the perception encoder. The control component, a deep ensemble of probabilistic MLP policies trained with the heteroscedastic loss, consumes those latent samples through the BNN+LV formulation, allowing uncertainty to flow from perception through to the final velocity commands.

Regarding **RQ-1**, our experiments show that incorporating uncertainty throughout the navigation pipeline improves the robustness to environmental perturbations (gate position noise), but only modestly when the UAV is controlled with the expected value of the predictions. The fully Bayesian architecture $$\mathcal{M}_4$$ is the best model at both noise levels, yet its advantage over the lighter partial uncertainty-aware models is small at noise level 1, and all models pass fewer than a third of the gates at noise level 2. This raises a practical tradeoff: the computational overhead of full Bayesian inference (32 MCD forward passes and 5 ensemble members, i.e., 160 control prediction samples) yields only modest gains when its predictions are used naively.

Regarding **RQ-2**, the double-gate experiment suggests a mechanistic explanation, with which the MI-Mode results are consistent. When two gates simultaneously appear in the UAV's field of view, a direct consequence of gate position noise, the navigation architecture faces a genuinely ambiguous situation. The BDL pipeline reflects this: most ensemble members produce **multimodal predictive distributions** over lateral and yaw velocities, with two modes that can be associated with the two gates in the field of view. This is actually a strength of the uncertainty-aware architecture, since it identifies the ambiguity. However, the ensemble expected value averages across the two modes, which can produce a velocity command that does not steer toward either gate. This plausibly degrades navigation in ambiguous situations such as the double gate, which become more frequent as the gate noise increases.

Regarding **RQ-3**, using the system's own uncertainty at runtime makes the difference. The MI-Mode strategy chooses, for each velocity command, the ensemble member with the lowest mutual-information score, i.e., one whose predictions change little with the perception uncertainty and stay close to the average prediction of the whole system, acts on a mode of its predictive distribution for the lateral, yaw, and vertical velocities with a consistent tie-breaking rule, and conservatively takes the lowest predicted forward velocity. With the same model and the same predictions, the average number of gates passed increases from 19.77 to 28.88 at noise level 1, and from 9.22 to 23.05 at noise level 2. In short, capturing uncertainty is only half of the job; the other half is to use it in the decisions of the system. Beyond the performance gains, BDL exposes *when and why* the system struggles, which is arguably the most valuable property for safety-critical autonomous systems <d-cite key="mcallister2017concrete"></d-cite>.

Nevertheless, MI-Mode remains an ad-hoc strategy, tailored to the identified critical situations and the gate noise levels in our experiments, and there are no guarantees that it will be robust to higher environmental perturbations. An exciting line for future work is exploring, designing, and implementing behavior models that can be triggered according to the current UAV situation, e.g., based on uncertainty and multimodality detection. In addition, sampling-free methods for uncertainty estimation <d-cite key="charpentier2021natural"></d-cite> could reduce the computational budget and memory footprint of our approach.

For more in-depth info about this topic, please check the papers <d-cite key="arnez2021improving,arnez2022towards,arnez2022quantifying"></d-cite> and Chapter 4 of the PhD thesis <d-cite key="arnez2023navigation"></d-cite>.

<!-- The citation is presented inline like this: <d-cite key="gregor2015draw"></d-cite> (a number that displays more information on hover).
If you have an appendix, a bibliography is automatically created and populated in it. -->

<!-- Just wrap the text you would like to show up in a footnote in a `<d-footnote>` tag. -->
<!-- The number of the footnote will be automatically generated.<d-footnote>This will become a hoverable footnote.</d-footnote> -->
