# Plume Physics Random Forest Emulator

## Summary

This repo contains a random forest emulator for a 1D volcanic plume model. The model used for emulation is the coupled conduit-vent-atmosphere model for water-rich eruptions from:

Rowell, C. R., Jellinek, A. M., Hajimirza, S., & Aubry, T. J. (2022). External Surface Water Influence on Explosive Eruption Dynamics, With Implications for Stratospheric Sulfur Delivery and Volcano-Climate Feedback. Frontiers in Earth Science, 10. https://www.frontiersin.org/article/10.3389/feart.2022.788294

The 1D model is not included in this repo. Instead, we include a set of 30,000 model outputs performed in monte carlo fashion, with primary inputs randomized within uncertainty.
The usage case is for rapid estimation of particle and water mass fluxes and plume heights, for inclusion as flux terms in seperate numerical model.

The simulation data set was produced for specific application to the 1918 eruption of Katla Volcano, Iceland, and consequently the data set is limited to a specific range of input parameters.  The random forest is therefore only trained over this limited parameter range. The randomized input parameters and their ranges and distribitions are listed below.

| **Param** | **Range**     | **Distribution Type**   |
| --------- | ------------- | ----------------------- |
| **logQ**  | 1e6 - 1e8     | uniform                 |
| **Zw**    | 0 - 400       | uniform                 |
| **T**     | 1375 - 1525   | uniform                 |
| **n_0**   | 0.005 - 0.025 | uniform                 |
| **n_ec**  | 0 - 0.2       | uniform                 |
| **a_var** | 1, 0.15       | normal (mean, std. dev.)|
| **D**     | 2.7, 2.8      | discrete                |

The two most important parameters (and those used to train the random forest) are the base-10 log of mass eruption rate (**logQ**) and the depth of water over the vent (**Zw**).

## Workflow/Repository Structure

### Data input and pre-processing
...

### Exploratory analysis and initial random forest training
...

### Plotting routines
...

### Finalized random forest - training script and saved binary
...