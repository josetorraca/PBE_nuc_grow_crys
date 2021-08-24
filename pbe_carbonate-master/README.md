# Particle Size Distribution Simulations for the Carbonate Deposition Research

## Current folders:

- `py_pbe_msm`: development of a Python package to solve Population Balance Equations using the Moving Sectional Method; Contains the sample folder with some applications;
- `pbemsm`: just for reference, not being used;
- `spikes`: miscellaneous files for evaluation and experimentation;
- `vessel_carbonate`: contains the model to describe the Carbonate experiments in the Crystallization Vessel

## Questions

### Experimental

- Discuss if it is breakage or deposition at the end of the batch
- What about the mass of solids at the end of the batch? Is it possible to measure it?

From the TB experiment: $l_{mean}(t_f) \approx 20 \mu m$; $N_{measured} = \dfrac{1E6}{80mL}$, thus (considering cubic crystals of CaCO3):

$$m_{crystals} = l_{mean}^3 * N_{measured} * V(t_f) * \rho_c $$

$$m_{crystals} = (20*10^{-4})^3 * 12500 * 2.71 * 698 mL = 0.1891578 g$$

```{.python .run format=text hide_code=True}
print((20e-4)**3 * 12500 * 2.71 * 698)
```
