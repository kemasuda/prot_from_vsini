# prot_from_vsini
rotation period distribution from vsini and radius ([Masuda, Petigura, Hall 2021](http://arxiv.org/abs/2112.07162))

#### main: scripts for main analysis

- data/sample_stars.csv

  parameters of the 144 stars used in the main anlaysis

- period_posterior.py

  compute marginal likelihood for prot and save results 

- run_hierarchical.py

  run the main hierarchical analysis to infer prot distribution (requires likelihood files from period_posterior.py)

#### simulation: scripts for injection recovery tests

- simulate_samples.py

  simulate mock catalogs of radii and vsini for a given prot distribution

  samples.csv files in sim-* directories are the datasets used in the paper 

- period_posterior.py, run_hierarchical.py

  same as in the main analysis

  <!--

- run_hierarchical_MC.py

  run hierarchical analyses for random subsets of the simulated samples and check how much the results fluctuate depending on the sample size

  -->

  

 





