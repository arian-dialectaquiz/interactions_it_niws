# plan_v6.md — paper 3, version 6 (full_v5/new_figures)

Version 6 of the hurricane wake and internal tide manuscript. It keeps every computed result of v5 (the netCDF files, block tables and reports under `../figN_out` and `../region_out`) and rebuilds the figures from them, so nothing runs again from the model. Six main figures from eight, 30 panels from 46, about 14 displayed equations from 26, and a Supplementary Material that holds every panel, equation and check the argument does not need in the main text. Target Ocean Modelling, main text about 6300 words (15 to 20% under v5).

---

## 0. The story

**One sentence.** Near the latitude where the inertial frequency is half the tidal frequency, the strain of the internal tide does work on the inertial currents of a hurricane wake, in a direction set by the phase of the wake against the tide when the wind stops and to an amount set by the detuning of the wake against its decay, and the internal tide loses that energy, so the timing of a storm against the tide sets whether its wake receives energy from the tide.

**Title test.** Storm wakes and internal tides are the two largest sources of internal wave energy for mixing and are budgeted as if they never met. Near 29 degrees they share a frequency, and this paper measures the exchange in a resolved hurricane, shows that three quantities a forecast holds predict it with no adjustable parameter, and shows which of them the boundary current controls.

**Title.** "The internal tide transfers energy to a hurricane wake near the critical latitude" (kept from v5).

**The arc in six figures.** The setting (1) → the two wakes and what the wind put into each (2) → the work of the tidal strain and the loss of the tide (3) → the phase and the frequency of the wake under the flow (4) → where the energy goes (5) → the exchange predicted from the phase at the wind stop (6). Each figure answers the question the previous one raised.

**What moves to the Supplementary Material.** The panels that test robustness or show the same result in another projection (S3 to S7), the model configuration details and validation (S1 and S2), the derivation of the energy equation from the momentum equation and the Adler equation (Section S8 text), and the checks that reproducibility needs and the argument does not (the consistency of the strain work rebuilt from the phase, the sum band null, the alternative class by the sign of the wake frequency).

---

## 1. Storyboard of the main figures

Every plot script in this folder reads the v5 outputs by relative path (`../figN_out/`, `../region_out/`) and writes `figv6_N_*.png` and `.pdf` beside itself. All run in seconds.

    cd ~/dd_waves/paper_3/full_v5/new_figures
    for f in figv6_0*_plot.py figS*_plot.py; do python $f; done

Eleven scripts, six main figures and five supplementary figures (S3 to S7).

The helpers shared by every script (paths, the map frame, the window shading, the block-bootstrap mean of a map) sit in `v6_common.py`, which also imports `../paper_style.py`. Each script keeps its own explicit GridSpec.

### Figure 1. The region, the storm and the tide (`figv6_01_plot.py`, from `fig_01_plot.py` revision 4)
- Unchanged from v5. (a) region, grids, track, moorings. (b) detuning of FULL in the storm window. (c) share of the slope with negative detuning by latitude and window. (d) semidiurnal envelope, wind stress, windows.
- Answers where and when, and how far the boundary current moved the resonance. Raises what the storm left on the slope and whether the tide changed it.

### Figure 2. The two wakes and what the wind put into each (`figv6_02_plot.py`)
- Answers Q1 and closes it as a limit of the design (the eddies set the comparison).
- (a) and (b) wake kinetic energy maps with and without the tide (from `fig2_wake.nc` and `region_maps.nc`), (c) ratio by latitude with its band, (d) footprint series with the wind work accumulated on each wake, (e) new panel, the near-inertial wind work and the change of the wake kinetic energy by class of the flow for the two wakes (block means with intervals, from `fig6_budget.nc` cls_terms), with the footprint ratio of wind work above. (e) carries in one panel the +6.2 against −4.4 kJ m⁻² of the blocks below resonance, which the discussion of v5 took from Figure 6 (e) and (f).
- Dropped to S3 (a)(b): the kinetic energy against the detuning of the flow and the ratio by group through time. The accumulated budgets of the two wakes (v5 Figure 6 (a)(b)) are S3 (c)(d).
- Raises what put the extra energy there.

### Figure 3. The work of the tidal strain and the loss of the tide (`figv6_03_plot.py`)
- Answers Q2 (rate, sign, nulls) and the first half of Q4 (the tide loses the energy).
- (a) tidal strain work map with the gain contoured, without the triad inset. (b) budget of the gain through time, two stacked panels. (c) cumulative strain work inside the phase null for the wake with the tide and the kinematic pairing, rank of the actual timing above. (d) the tide side through time (work on the wake, loss of the tide, standing deficit, null cross term) from `fig6_budget.nc`, with loss over work and the residence time above. (e) works by group as bars with intervals and the bilinearity slope.
- Dropped to S4: the latitude profiles of work and gain, the works by depth and the semidiurnal band through time (v5 Figure 6 (d)). The triad scatter of the inset is dropped (a cloud of points) and its numbers go to the text.
- Raises what sets the local sign of the work.

### Figure 4. The phase of the wake and its frequency under the flow (`figv6_04_plot.py`)
- Answers Q3 (phase sets the sign, the storm sets the frequency, the flow acts through the phase alone).
- (a) phase efficiency map of the wake window (`region_maps.nc`), (b) cos Φ by bin of the flow detuning (`fig4_phase.nc` curve), (c) the wake frequency minus each candidate and the detunings by class with the locking half-width (`fig5_blocks.csv`), Kunze slope and blue shift above, (d) gain and strain work by bin of the flow detuning (`fig5_resonance.nc`), (e) ranked controls of the strain work and of the gain with the shift levels.
- Dropped to S5 (phase in time: Hovmöller by detuning, footprint cos Φ through time, coherence, rose) and S6 (the two detuning maps and their latitude profile).
- Raises what the exchange amounts to and where the wake energy goes.

### Figure 5. Where the energy of the wake goes (`figv6_05_plot.py`)
- Answers the second half of Q4.
- (a) and (b) depth and time of the wake with the tide by class with the centroids of both wakes, (c) downward fraction through the wake window by class and pair, (d) rotary vertical wavenumber spectra by class, (e) ratio with over without the tide by depth with the shear variance ratio above.
- Panel titles name the class by the sign of the flow detuning only ("flow below resonance (Δc < 0)"), the words free and trapped are gone from the figure.
- Dropped to S7 (c): the decay rate by depth.
- Raises whether the mechanism can be stated as a prediction.

### Figure 6. The exchange predicted from the phase at the wind stop (`figv6_06_plot.py`)
- Closes the mechanism with the reduced model.
- (a) strain work of the footprint after the wind stop, measured and from the free drift (each block and one frequency), (b) predicted work on the footprint with the measured sign of each block, (c) efficiency by bin of the phase at the wind stop with the model curves and the no-detuning cosine, (d) the timing test folded onto one tidal period, (e) |efficiency| by quarter of the decay rate with the Lorentzian envelope.
- Dropped to S7 (d): the agreement of the model phase with the measured phase through time (the horizon of 3.5 d goes to the text). The legend says fixed phase in place of frozen phase.

---

## 2. Supplementary figures (`figS3_plot.py` to `figS7_plot.py`)

Five figures of four panels (S6 of three) from the sixteen panels that left the main text.

| Figure | Content | Source |
|---|---|---|
| S1 | The parametric vortex against the best track (to be made by Arian) | |
| S2 | The mesoscale field against altimetry and the cold wake against SST (to be made) | |
| S3 | (a) wake kinetic energy against the detuning of the flow under each wake, (b) with over without the tide through time by group, (c) budget of the wake with the tide accumulated from 24 March, (d) budget of the wake without it | fig2 (e)(f) and fig6 (a)(b) |
| S4 | (a) work of the internal tide and gain along the slope by latitude, (b)(c) the works by depth by class, (d) the semidiurnal band through time (FULL, TIDE, storm-made of each pair) | fig3 (c)(d) and fig6 (d) |
| S5 | (a) cos Φ by detuning and time, (b) cos Φ of the footprint through time, (c) coherence of the block phases, (d) rose of the phase | fig4 (b)(c)(d)(e) |
| S6 | (a) wake detuning map, (b) flow detuning map, (c) the two by latitude with the blue shift | fig5 (a)(b)(c) |
| S7 | (a)(b) the budget terms by class for both wakes, (c) decay rate by depth, (d) agreement of the model phase through time | fig6 (e)(f), fig7 (d) and fig8 (a) |

The colourbar of S6 reads "[below to above resonance]". The decay rate axis of S7 (c) names its sign in the label ("growth below 0 decay above") and the agreement panel of S7 (d) says fixed phase and "agreement halved after".

## 3. The equations of the main text (14) and of the Supplementary Material

Main text, each numbered on its own and named at every citation.

| Label | Name in the text |
|---|---|
| eq:feff | the effective inertial frequency |
| eq:detune | the detuning of the flow |
| eq:wakewith | the wake with the tide (u_k stated inline as the same difference for WIND and BACK) |
| eq:windwork | the near-inertial wind work |
| eq:psigma | the strain work of the tide (P_δ and P_z stated inline as the divergence and vertical shear works with their integrands) |
| eq:rotary | the rotary strain work |
| eq:phidef | the phase of the wake against the tide |
| eq:Z | the interaction coefficient of the column (Φ = π + arg Z and the column work stated inline) |
| eq:lock | the locking condition |
| eq:budget | the budget of the wake |
| eq:residence | the residence time (c_g = L/τ_r inline) |
| eq:drift | the free drift of the phase |
| eq:work | the predicted work (the efficiency η_b stated inline) |
| eq:analytic | the work of a detuned oscillator (the Lorentzian envelope inline) |

Supplementary Material, Section S8, "From the momentum equation to the strain work": the complex momentum equation of the wake in the tidal strain, the energy tendency and its three works, the tide side T and the flux-divergence closure, the Adler equation and its fixed point, the downward fraction from the rotary spectrum and the finescale ratio. Each numbered (S1) to (S8) and cited from the main text by name.

---

## 4. The manuscript rewrite (next step)

Sections and word budget (main text without abstract, captions, tables and equations), about 6300 words.

| Section | v5 | v6 | What changes |
|---|---|---|---|
| Introduction | 894 | 850 | the five paragraphs of the last revision, trimmed |
| Methods | 2855 | 1900 | the configuration paragraph kept, the derivation of 2.4 replaced by the three works and the rotary form with the derivation cited to the SM, 2.5 and 2.6 merged, 2.7 and 2.8 shortened to the quantities Figures 3 and 5 use, 2.9 kept with the Adler equation in the SM, statistics kept |
| Results | 1570 | 1250 | six paragraphs, one per figure, the Figure 2 paragraph closing Q1 on panel (e), the Figure 3 paragraph carrying the tide side, the Figure 4 paragraph carrying the phase and the frequency together |
| Discussion | 2200 | 1950 | the six subsections kept, the repeated numbers removed, every paragraph still with the seminal and the 2020 onward literature |
| Conclusions | 390 | 350 | unchanged in substance |

Captions follow the panels above and describe only. The numbers the dropped panels carried (the triad ratio and its rank correlation, the latitude bands of the gain, the coherence of the phases, the cross term, the sum band null, the horizon of the phase agreement) go to the text or to the SM captions.

---

## 5. Typography, colour and wording of the v6 figures

- **Size and fonts (2 November).** Every main figure is 9.5 inches wide (Figure 1 keeps its 11 inch layout with its fonts scaled by 1.4), with 12 pt axis labels, 11 pt tick labels, 10.5 pt legends and panel notes, 10 pt grey statistics titles and 13 pt panel letters, set once in `v6_common.py`. Printed at the text width of a journal column this gives about 8 pt on the page. Legends sit below their axis, one column where the panel is narrow, and the seven budget terms of Figure 3 (b) share one figure-wide legend between the rows.
- **No dead space.** The panel that used to stand alone in the last row (Figure 2 (e), Figure 3 (e), Figure 5 (e)) now shares a row of two or three panels, so each figure is two or three rows of equal width.
- **No text over data.** The words below resonance and above resonance no longer sit inside the axes. The hatched side of a detuning axis carries a legend entry ("flow above resonance") through `hatch_above` in `v6_common.py`, the class of a Hovmöller or a profile is its coloured title, and the marks of Figure 6 (d) (actual timing, quarter period) are legend entries.
- **Colour vision.** The kinetic energy maps and Hovmöllers use YlGnBu (light where small, saturated where large, monotone in lightness) in place of terrain_r. The signed fields keep PuOr (work) and balance (detuning), the phase efficiency keeps the trimmed gnuplot2 (monotone in lightness), the classes keep steel blue and indian red, and every pair of lines differs in style as well as colour. All colours are named.
- **Wording.** No commas inside figure text and no figurative names. Classes are "below resonance" and "above resonance" from the sign of the flow detuning. The words free, trapped, frozen, pay, feed and weak do not appear (fixed phase, amplitude floor, loss of the tide). "Free drift" stays, as the name of the unforced phase evolution in the Adler equation.

## 6. Log

| Date | Item | Decision |
|---|---|---|
| 2026-11-02 | v6 storyboard | six main figures from the v5 outputs, 30 panels, the dropped panels in the SM, 14 main-text equations, derivation to the SM text. Scripts written in `new_figures/` and tested on the synthetic v5 outputs (layout only) |
| 2026-11-02 | Figure 2 (e) | wind work and change of kinetic energy by class for both wakes, from `fig6_budget.nc`, closes Q1 inside Figure 2 |
| 2026-11-02 | Figure 3 (d) | the tide side of v5 Figure 6 (c) moves into Figure 3, so the rate and the loss of the tide close in one figure |
| 2026-11-02 | Figure 4 | merges the phase of v5 Figure 4 (a)(f) with the frequency and the controls of v5 Figure 5 (d)(e)(f) |
| 2026-11-02 | Fonts and layout | figures 9.5 inches wide with 12 pt labels (Figure 1 scaled by 1.4), last rows merged into rows of two or three panels, regime words moved from the axes to the legends, energy maps on YlGnBu for colour vision deficiency, Supplementary Material reduced to S3 to S7 (four panels each) |
| | Rio | run the eleven scripts, read the six figures in sequence against the storyboard, then rewrite the manuscript from Section 4 |
