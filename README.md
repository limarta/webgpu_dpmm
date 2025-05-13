# Dirichlet Process Mixture Modeling in the Browser

This repo implements [Dirichlet Process Mixture Modeling](https://en.wikipedia.org/wiki/Dirichlet_process#Use_in_Dirichlet_mixture_models) (DPMM) in WebGPU. DPMM is a hierarchical clustering model used
to understand data clusters *without* knowing the number of clusters ahead of time. This means that DPMMs remain flexible and provide good uncertainty
guarantees when the number of clusters is unknown.

Since this algorithm works locally, it works on any device without a backend ([see here](https://ian.limarta.org/webgpu_dpmm)).
