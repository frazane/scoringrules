---
title: 'scoringrules: probabilistic forecast evaluation'
tags:
  - python
  - scoring rules
  - probabilistic
  - forecasting
authors:
  - name: Francesco Zanetta
    orcid: 0000-0003-4954-4298
    equal-contrib: true
    affiliation: 1
  - name: Sam Allen
    orcid: 0000-0003-1971-8277
    equal-contrib: true
    affiliation: 2
affiliations:
 - name: Federal Office of Meteorology and Climatology MeteoSwiss, Zürich, Switzerland
   index: 1
 - name: Institute of Statistics, Karlsruhe Institute of Technology, Karlsruhe, Germany
   index: 2
date: 7 April 2025
bibliography: paper.bib

---

# Summary (needed)

A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience.

# Statement of need (needed)

A Statement of need section that clearly illustrates the research purpose of the software and places it in the context of related work.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

[@jordan_evaluating_2019]

# Figures

Figures can be included like this:
<!-- ![Caption for example figure.\label{fig:example}](figure.png) -->
<!-- and referenced from text using \autoref{fig:example}. -->

<!-- Figure sizes can be customized by adding an optional second parameter: -->
<!-- ![Caption for example figure.](figure.png){ width=20% } -->

# Acknowledgements (needed)

This is related to financial support.

# References (needed)
Will be automatically generated (?).
