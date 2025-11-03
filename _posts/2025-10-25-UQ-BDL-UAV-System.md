---
layout: distill
title: "Quantifying and Using Uncertainty in Deep Learning-based UAV Navigation"
date: 2025-10-25
tags: Autonomous Navigation Uncertainty BayesianDeepLearning Systems
description: "Quantifying and using uncertainty in Bayesian deep learning systems"
# appendix: true
citation: true
tikzjax: true
published: false
bibliography: 2025-10-25-UQBDLSys.bib
authors:
  - name: Fabio Arnez
    affiliations:
      name: Unviersite Paris-Saclay, CEA, List
# toc:
#   sidebar: left
toc:
  - name: Introduction
  - name: The Navigation Task and Architecture Overview
  - name: Quantifying Uncertainty in the DNN-based Navigation System
    subsections:
      - name: Learning Perception Representations
      - name: Uncertainty From Perception Representations
      - name: Learning a Probabilistic Control Policy
      - name: Handling Input Uncertainty In The Control Policy
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

Autnomous systems, like Unmanned Aerial Vehicles (UAVs) and self-driving cars, increasingly rely on Deep Neural Networks (DNNs) to handle critical functions within their navigation pipelines (perception, planning, and control). While DNNs are powerful, deploying them in safety-critical roles demands that they accurately express their confidence in predictions. This is where Bayesian Deep Learning (BDL) <d-cite key="gal2016dropout,lakshminarayanan2017simple"></d-cite> comes in, offering a principled framework to model and capture uncertainty.
However, if the Bayesian approach is followed, ideally all the components in the navigation system pipeline (perception, planning, control) should use BDL to enable uncertainty propagation in the pipeline, so that output of the system reflects the uncertainty of the system as a whole <d-cite key="mcallister2017concrete"></d-cite>. Uncertainty propagation is challeging as it requires BDL components to admit uncertainty information as an input of the DNN to account for the uncertainty coming from previous components.

In this post, we describe how to capture and use uncertainty along a navigation pipeline of BDL components. Moreover, we assess how uncertainty quantificaiton throughout the system impacts the navigation performance of an UAV that must fly autonomously through a set of gates dispoosed in a circle within a simulated environment (AirSim).

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

We consider a minimalistic end-to-end deep learning-based navigation architecture to study uncertainty propagation and its use. Therefore, in our experiments, the autonomus navigation architecture consists of two neural network components, one for **perception** and the other for **control**, as presented in <a href="#fig:nav-arch">Figure 2</a>.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <a id="fig:nav-arch"></a>
        {% include figure.liquid loading="eager" path="assets/img/posts/2025-10-25-UQ-BDL-UAV-System/UAV-nav-arch.jpg" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>
<div class="caption">
    Figure 2: UAV autonomous navigation architecture.
</div>


To create an instance of the architecture above, we can follow the approach presented in <d-cite key="bonatti2020learning"></d-cite>, where the perception component defines an encoder function  $$q_{\phi}:\mathcal{X} \rightarrow \mathcal{Z}$$ that maps the input image $$\mathbf{x}$$ to a rich low dimensional representation  $$\mathbf{z} \in \mathbb{R}^{10}$$. Next, a control policy $$\pi_{w}: \mathcal{Z} \rightarrow \Upsilon$$ maps the compact representation $$z$$ to control velocity $$\Upsilon = \{\dot{x}, \dot{y}, \dot{z}, \dot{\psi}\} \in \mathbb{R}^{4}$$ commands, corresponding to the desired linear and yaw velocities, respectively in the UAV body frame. This desired velecities are then sent to the UAV low-level controller which is responsible for the UAV motion in the simulator. <a href="#fig:bonatti-nav-arch">Figure 3</a> shows the UAV navigation architecture proposed by Bonatti et al. <d-cite key="bonatti2020learning"></d-cite>, where the control policy is implemented using a multilayer perceptron (MLP), and the perception encoder is implemented using the encoder block of a variational autoencoder (VAE). 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <a id="fig:bonatti-nav-arch"></a>
        {% include figure.liquid loading="eager" path="assets/img/posts/2025-10-25-UQ-BDL-UAV-System/bonatti_nav_arch.png" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>
<div class="caption">
    Figure 3: The input image is encoded into a latent representation of the environment. A control policy acts on the lowerdimensional embedding to output the desired robot velocity commands.
</div>

Nevertheless, Bonatti et al. <d-cite key="bonatti2020learning"></d-cite>, employ a special type of VAE, a cross-modal VAE (CM-VAE), that allows mixing two data modalities. In the proposed CM-VAE, besides reconstructing the input image, an additional network block is added and trained to predicts the gate's pose (position and orientation in spherical coordinates) relative to the UAV camera, i.e., the additional network block attached to the VAE performs a (supervised) regression task with the addional data modality (gate pose labels for each image), as presented in <a href="#fig:cmvae">Figure 4</a>.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
       <a id="fig:cmvae"></a>
        {% include figure.liquid loading="eager" path="assets/img/posts/2025-10-25-UQ-BDL-UAV-System/bonatti_percept_cmvae.png" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>
<div class="caption">
    Figure 4: Cross-Modal VAE: Each input image sample is encoded into a single latent space that can be decoded back into images, or transformed into another data modality such as the poses of gates relative to the UAV.
</div>

Moreover, the additional network block for predicting the gate poste, is connected in an usual way to the VAE. The regression block only uses the first four variables of the latent vector at the output of the CM-VAE encode. In addition, each of this 4 latent vector variables, is connected to a dedicated regressor for each of the predicted spherical coordinates (radius, polar, azimuth, yaw).
This causes a stronger regularization to the latent space during traning since the first 4 variables of the latent space vector will receive an additional flow of error gradients corersponding to pose prediction errors, and therefore, forcing the disentanglement of this 4 latent vector variables, as presented in <a href="#fig:disentangled-representation">Figure 5</a>.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <a id="fig:disentangled-representation"></a>
        {% include figure.liquid loading="eager" path="assets/img/posts/2025-10-25-UQ-BDL-UAV-System/cm-vae-disentangled.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Figure 5: CM-VAE disentangled representations: Changing the values from the first four latent vector variables allows us to control the gate attributes in the generated images.
</div>

### Learning Perception Representations

To build the perception component, we use the Dronet architecture <d-cite key="loquercio2018dronet"></d-cite> for the CMVAE encoder $$q_{\phi}$$. As mentioned before, additional constraints are imposed in the latent space to promote the learning of disentangled representations. For this purpose, the latent vectors $$z$$ are treated differently by the decoder $$p_{\theta}$$ and the regression networks. The image decoder $$p_{\theta}$$ uses the whole latent vector $$z$$, while the regression network $$p_{\rho}$$ uses only the first four elements $$z_{1:4}$$. Each element from $$z_{1:4}$$ has a dedicated network to predict a specific gate pose $$\xi$$ attribute. Moreover, the CMVAE loss function in Equation \eqref{eq:loss_cmvae_prob} has additional regularization coefficients that penalize the predictions of each network and the closeness to the imposed latent structure.

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
\mathcal{L}_{p}(\mathcal{D}_{p}; \phi, \theta, \rho) = \frac{\alpha_{p}}{2} {\big\Vert {\mathbf{x} - p_{\theta}(\hat{\mathbf{x}} \mid z)} \big\Vert}^{2} \\ + \frac{\gamma_{p}}{2} {\big\Vert \xi - p_{\rho}(\hat{\xi}\mid z_{1:4}) \big\Vert}^{2}\\ + \beta_{p} \; \mathbb{KL}\big( q_{\phi}(z \mid \mathbf{x}) \; \Vert \; \mathcal{N}(0, \mathbf{I})\big)
\end{equation}
$$

### Learning a Probabilistic Control Policy

Once the perception component is trained, we use only the encoder of the trained CM-VAE to get a rich compact representation (latent vector) of the input image. The downstream control task (control policy $\pi$) uses a MLP network to operate on the latent vectors $z$ at the output of the CMVAE encoder $q_{\phi}$ to predict UAV velocities. To this end, a probabilistic control policy network is added at the output of the perception encoder $q_{\phi}$, forming the UAV navigation stack. The probabilistic control policy network $\pi_{w}(\Upsilon \mid \mathbf{z})$ predicts the mean and the variance for each velocity command given a perception representation $z$ from the encoder $q_{\phi}$, i.e., $$\Upsilon \sim \mathcal{N}\big(\mu_{w}(z), \sigma^{2}_{w}(z)\big)$$, where
$$\Upsilon_{\mu} = \{\mu_{\dot{x}}, \mu_{\dot{y}}, \mu_{\dot{z}}, \mu_{\dot{\psi}}\}$$ and
$$\Upsilon_{\sigma^{2}} = \{\sigma^{2}_{\dot{x}}, \sigma^{2}_{\dot{y}}, \sigma^{2}_{\dot{z}}, \sigma^{2}_{\dot{\psi}}\}$$. For training the probabilistic control policy, we use imitation learning with a dedicated control dataset $\mathcal{D}_c$, and the heteroscedastic loss function <d-cite key="kendall2017uncertainties,lakshminarayanan2017simple"></d-cite> from Equation \eqref{eq:loss_ctrl_policy}.

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

Following the work by <d-cite key="bonatti2020learning"></d-cite>, during training, we freeze the weights fromt the perception encoder $q_{\phi}$ to update only the weights $w$ from the control policy network, as presented in the control component from <a href="#fig:uncertainty-nav-arch">Figure 7</a>.

### Autonomous Navigation Overview

After training the components of the minimalistic navigation architecture, we obtain autonomous navigation flight that drives the UAV through the red gates as shown in <a href="#fig:nav-airsim">Figure 6</a>.

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

Although the CMVAE encoder $$$q_{\phi}$$ employs Bayesian inference to obtain latent vectors $$\mathbf{z}$$, CMVAE does not capture epistemic uncertainty since the encoder lacks a distribution over parameters $\phi$. To capture uncertainty in the perception encoder, we follow prior work from <d-cite key="daxberger2019bayesian,jesson2020identifying"></d-cite> that attempts to capture epistemic uncertainty in VAEs. We adapt the CMVAE to capture the posterior $$q_{\Phi}(\mathbf{z} \mid \mathbf{x}, \mathcal{D}_p)$$ as shown in Equation \eqref{eq:postEncoder}.

$$
\begin{equation}
\label{eq:postEncoder}
q_{\Phi}(\mathbf{z} \mid \mathbf{x}, \mathcal{D}_{p}) = \int{q(\mathbf{z} \mid \mathbf{x}, \phi) \; p(\Phi \mid \mathcal{D}_{p}) \; d\phi}
\end{equation}
$$

To approximate Equation \eqref{eq:postEncoder} we take a set $${\Phi = \{\phi_{m}\}^{M}_{m}}$$ of encoder parameter samples $$\phi_{m} \sim p(\Phi \mid \mathcal{D}_{p})$$, to obtain a set of latent samples $$\{z_{m}\}^{M}_{m=1}$$ from the output of the encoder $$q_{\Phi}(\mathbf{z} \mid \mathbf{x}, \mathcal{D}_{p})$$. In practice, we modify the CMVAE by adding a dropout layer in the encoder. Then, we use Monte Carlo Dropout (MCD) <d-cite key="gal2016dropout"></d-cite> to approximate the posterior on the encoder weights $$p(\Phi \mid \mathcal{D}_{p})$$, as shown for the perception component in <a href="#fig:uncertainty-nav-arch">Figure 7</a>. Finally, for a given input image $$\mathbf{x}$$, we perform $$M$$ stochastic forward passes (with dropout turned on) to compute a set of $$M$$ latent vector samples $$\mathbf{z}$$ at runtime.

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
To do that we consider using the BNN with LV inputs (BNN+LV) <d-cite key="henaff2018model,depeweg2017learning,depeweg2018decomposition"></d-cite> approach to propagate the uncertainty from perception to control in a principled way.
To capture the overall system uncertainty at the output of the controller, we compute the posterior predictive distribution for target variable $$\Upsilon^{*}$$ associated with a new input image $\mathbf{x}^{*}$, as shown in Equation \eqref{eq:postPredDist} and Equation \eqref{eq:post_pred_dist_whole_system}:

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
p(\Upsilon^{*} \mid \mathbf{x}^{*}, \mathcal{D}_{c},\mathcal{D}_{p}) = \\\int \int \int  \underbrace{\pi(\Upsilon^{*} \mid \mathbf{z}, \mathbf{w})}_\textit{control policy} \; p(\mathbf{w} \mid \mathcal{D}_{c}) \underbrace{q(\mathbf{z} \mid \mathbf{x}^{*}, \Phi)}_{\textit{perception encoder}} p(\Phi \mid \mathcal{D}_{p})\; d\phi \; dz \; dw
\end{multline}
$$

The integrals from the equations above are intractable, and we rely on approximations to obtain an estimation of the predictive distribution. The posterior $$p(\mathbf{w} \mid \mathcal{D}_{c})$$ is difficult to evaluate. Thus, we can approximate the inner integral using an ensemble of neural networks <d-cite key="gustafsson2019evaluating"></d-cite>. As presented in <a href="#fig:uncertainty-nav-arch">Figure 7</a>, in practice, we train an ensemble of $N$ probabilistic control policies $${\pi}_{w_{n}}(\Upsilon \mid \mathbf{z}, w_{n})$$, with weights $$\{w_{n}\}^{N}_{n=1} \sim p(\mathbf{w} \mid \mathcal{D}_{c})$$, and where each control policy $${\pi}_{w_{n}}$$ in the ensemble predicts the mean $\mu_{w_{n}}(\mathbf{z})$ and variance $$\sigma^{2}_{w_{n}}(\mathbf{z})$$ for each velocity command, i.e., $$\Upsilon \sim \mathcal{N}\big(\mu_{w_{n}}(\mathbf{z}), \sigma^{2}_{w_{n}}(\mathbf{z})\big)$$.

The inner integral is approximated by taking a set of samples from the perception component latent space. Latent representation samples are drawn using the encoder mean and variance $$\mathbf{z} \sim \mathcal{N}(\mu_{\phi},\sigma^{2}_{\phi})$$. For brevity, we directly use the samples obtained in the perception component $$\{z_{m}\}^{M}_{m} \sim q_{\Phi}(\mathbf{z} \mid \mathbf{x}, \mathcal{D}_{p})$$ to take into account the epistemic uncertainty from the previous stage. Finally, the predictions we get from passing each latent vector $\mathbf{z}$ through each ensemble member are used to estimate the posterior predictive distribution in Equation \eqref{eq:postPredDist}. The predictive distribution $$p(\Upsilon^{*} \mid \mathbf{x}^{*}, \mathcal{D}_{c},\mathcal{D}_{p})$$ from Equation \eqref{eq:post_pred_dist_whole_system}, takes into account the uncertainty from both system components.

From the control policy perspective, using multiple latent samples $\mathbf{z}$ can be seen as taking a better "picture" of the latent space (perception representation) to gather more information about the environment. Interestingly, we can also make a connection between our sampling approach and the works <d-cite key="tai2019visual,zhang2022memo"></d-cite> that aim at sampling the input space by performing translations and augmentations on input images to improve prediction robustness.

## Navigation Performance Evaluation

ToDo ToDo

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/2025-10-25-UQ-BDL-UAV-System/UAV-nav-task.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 3: UAV task"%}
    </div>
</div>

## Conclusion

For more in depth info about this topic, please check the papers <d-cite key="arnez2021improving,arnez2022towards,arnez2022quantifying"></d-cite>. This is my thesis citation <d-cite key="arnez2023navigation"></d-cite>. This is an article citation <d-cite key="ollier2023towards"></d-cite>. This is a thesis citation <d-cite key="feng2021uncertainty"></d-cite>.

<!-- The citation is presented inline like this: <d-cite key="gregor2015draw"></d-cite> (a number that displays more information on hover).
If you have an appendix, a bibliography is automatically created and populated in it. -->

<!-- Just wrap the text you would like to show up in a footnote in a `<d-footnote>` tag. -->
<!-- The number of the footnote will be automatically generated.<d-footnote>This will become a hoverable footnote.</d-footnote> -->