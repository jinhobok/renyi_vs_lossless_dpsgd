# renyi_vs_lossless_dpsgd
A repository for reproducing the numerical results in the paper ["Can Rényi Accountants Beat Lossless Accountants for DP-SGD?"](https://arxiv.org/abs/2610.17120)

# Files
+ `asymp_renyi_vs_lossless.py`: code for generating Figure 4 (asymptotic comparison between our Rényi accountant and lossless accountant)
+ `asymp_renyi_vs_renyi.py`: code for generating Figure 5 (truncated asymptotic expansions and privacy bounds from Rényi accountants)
+ `birrell_rdp.py`: wrapper for running `birrell_rdp_bounds.py`
+ `birrell_rdp_bounds.py`: code for calculating privacy bounds from Birrell et al. (2024); direct copy of `fs_rdp_bounds.py` from https://github.com/star-ailab/FSRDP/blob/main/fs_rdp_bounds.py
+ `compare_dpsgd.py`: code for numerical comparison between Rényi accountants (Corollary 1, Birrell et al. (2024), Wang et al. (2021)) and lossless accountants (Doroshenko et al. (2022))
+ `compare_dpsgd_timing.py`: code for runtime analysis for Rényi accountants (Corollary 1, Birrell et al. (2024), Wang et al. (2021))
+ `connect_the_dots_accountant.py`: code for calculating privacy bounds, based on the Connect-the-Dots algorithm in Doroshenko et al. (2022)
+ `gaussian_mixture_max_renyi.py`: code for calculating (upper and lower bounds of) maximum of Rényi divergences over $u \in [0, 1/2]$ (see Appendix B.2)
+ `gaussian_mixture_renyi.py`: code for calculating (upper and lower bounds of) Rényi divergence at $u \in [0, 1/2]$ (see Appendix B.2)
+ `isosceles_rdp_scipy.py`: code for calculating approximation of Rényi divergence at $u \in [0, 1/2]$ 
+ `pld.py`: privacy loss distribution for one step of DP-SGD
+ `plot_and_table_from_csv.py`: code for generating Figures 2-3 and Tables 1-6 (comparison of privacy parameters from privacy accountants) from the output of `compare_dpsgd.py`
+ `plot_from_timing_csv.py`: code for generating Figure 6 (simple runtime plot) from the output of `compare_dpsgd_timing.py`
+ `prv.py`: privacy loss random variable for one step of DP-SGD
+ `rdp_to_dp.py`: code for calculating $(\varepsilon, \delta)$-DP from Rényi DP
+ `renyi_mixture_geometry.py`: code for generating Figure 1 (illustration of Gaussian mixtures that serve as Rényi dominating pairs)
+ `utils.py`: utils for handling ball arithmetic, Rényi orders, etc.
+ `wbk_rdp.py`: code for calculating privacy bounds from Wang et al. (2021), based on Google's `dp-accounting`

# Main numerical comparison
```bash
python compare_dpsgd.py --q 0.0025,0.01,0.04 --noise-multipliers 1,2,3,4,6,8 --steps 250,500,1000,2000,4000,8000,16000,32000,64000 --deltas 1e-6,1e-8 --alpha-max 256 --numerical-error 1e-5 --scipy-accuracy 1e-5 --output-directory results
```

Use `python compare_dpsgd.py --help` for the complete command-line interface.

# Main runtime comparison
```bash
python compare_dpsgd_timing.py --q 0.01 --noise-multipliers 2,6 --steps 1000,10000 --deltas 1e-8 --alpha-max 64 --numerical-error 1e-5 --scipy-accuracy 1e-6 --repeats 20 --output-directory results
```

# Remarks and AI use
+ The formats of figures and tables obtained from the code may have been modified for better presentation in the paper.
+ OpenAI's GPT-5.6 was used for generating the code, which was checked and edited by the authors. 

# References
## Papers
+ Jeremiah Birrell, Reza Ebrahimi, Rouzbeh Behnia, and Jason Pacheco. Differentially private stochastic gradient descent with fixed-size minibatches: Tighter RDP guarantees with or without replacement. In Advances in Neural Information Processing Systems, volume 37, pages 11087–11131, 2024.
+ Vadym Doroshenko, Badih Ghazi, Pritish Kamath, Ravi Kumar, and Pasin Manurangsi. Connect the dots: Tighter discrete approximations of privacy loss distributions. In Proceedings on Privacy Enhancing Technologies, volume 2022, pages 552–570, 2022.
+ Yu-Xiang Wang, Borja Balle, and Shiva Kasiviswanathan. Subsampled Rényi differential privacy and analytical moments accountant. Journal of Privacy and Confidentiality, 10(2), 2021.

## Codebases
+ https://github.com/star-ailab/FSRDP
+ https://github.com/google/differential-privacy/tree/main/python/dp_accounting
