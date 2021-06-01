Multi-Component Background Learning (MCBL) Framework
Modified 07/22/2019
Version 1.0

================================================
Copyright 2019 Institute for Computational Sustainability

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
================================================

*** Citation ***

This software accompanies the following publication:

Ament, S. E. et al. Multi-component background learning automates signal detection for spectroscopic data. npj Compu. Mat. 5, 77 (2019).

Bibtex:

@article{ament2019,
	Abstract = {Automated experimentation has yielded data acquisition rates that supersede human processing capabilities. Artificial Intelligence offers new possibilities for automating data interpretation to generate large, high-quality datasets. Background subtraction is a long-standing challenge, particularly in settings where multiple sources of the background signal coexist, and automatic extraction of signals of interest from measured signals accelerates data interpretation. Herein, we present an unsupervised probabilistic learning approach that analyzes large data collections to identify multiple background sources and establish the probability that any given data point contains a signal of interest. The approach is demonstrated on X-ray diffraction and Raman spectroscopy data and is suitable to any type of data where the signal of interest is a positive addition to the background signals. While the model can incorporate prior knowledge, it does not require knowledge of the signals since the shapes of the background signals, the noise levels, and the signal of interest are simultaneously learned via a probabilistic matrix factorization framework. Automated identification of interpretable signals by unsupervised probabilistic learning avoids the injection of human bias and expedites signal extraction in large datasets, a transformative capability with many applications in the physical sciences and beyond.},
	Author = {Ament, Sebastian E. and Stein, Helge S. and Guevarra, Dan and Zhou, Lan and Haber, Joel A. and Boyd, David A. and Umehara, Mitsutaro and Gregoire, John M. and Gomes, Carla P.},
	Da = {2019/07/19},
	Date-Added = {2019-07-22 17:38:08 -0400},
	Date-Modified = {2019-07-22 17:38:08 -0400},
	Doi = {10.1038/s41524-019-0213-0},
	Id = {Ament2019},
	Isbn = {2057-3960},
	Journal = {npj Computational Materials},
	Number = {1},
	Pages = {77},
	Title = {Multi-component background learning automates signal detection for spectroscopic data},
	Ty = {JOUR},
	Url = {https://doi.org/10.1038/s41524-019-0213-0},
	Volume = {5},
	Year = {2019},
	Bdsk-Url-1 = {https://doi.org/10.1038/s41524-019-0213-0}}


================================================
Detailed instructions coming soon. For now, see MCBL.py for the example functions MCBL_UGM and MCBL_Out.
