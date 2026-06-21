## Major issues
- What are we missing in terms of network input verification? One example here is which elements allow area change across them, and which elements do not. We should ensure consistency of area changes.

- If network has more than 2 terminal nodes, the automatic forcing terminal selection in perturbation_response fails. We should think here, current ideas are: for case A force all inlets, for case B force all outlet or user should explicitly provide input.

## Minor issues

### entropy_generator.ipynb
- The throat mach number should start from the quiescent case (zero Mach) or a very low Mach number to match the figures in the paper.

### Sudden-area-change switch biases the perturbation by O(eps)
The momentum<->isentropic smooth switch leaks its loss residual into the frozen perturbation Jacobian; per-element `eps` is the current workaround.
Proper fix: give the perturbation linearization its own sharp smoothing, decoupled from the mean-flow homotopy `eps`.

## To implement
- Solver should print progress on a user-specified interval and verbosity level.

### Complex matrix related
- plot_complex_matrix should take axis scale input, the plot_scattering_matrix and plot_transfer_matrix utilities set y-scale of magnitudes to (0,1) if there is nothing greater than 1, oterwise (0, max mag.)
- The labels should read f_1 -> f_2 etc. instead of f -> f, where subscripts are the indices of corresponding edges. Please verify labels read correctly, this is very important.
- The basis convention is, characteristics (char) with the order (f, g, h) and "primitive" (prim), with the order (p'/rho0/c0, u', rho'/c0/p0, notice these are all in units of velocity.

## To verify
- Quiescent analysis capability
- How does excitation work currently? I hope acoustics and entropy excitation do not happen simultaneously.

## To brainstorm
- Area change el
