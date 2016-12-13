# CUDA Library for Multipurpose Image Analysis

## Abstract

Image processing is increasing in demand every year as our technology produces
images in higher resolution and higher volumes. This demand arises from both
corporate and scientific sources alike. The volume of data produced increases
steadily however the methods to deal with it are limiting its usefulness. CUDA
Wndchrm is an open source re-implementation of select algorithms from
[wndchrm][]. They are modified and structured in a way that they may be
executed on Nvidia's CUDA enabled devices to employ parallel computing and
speed up the process of image analysis. The algorithms selected are those that
are the most time consuming to reduce image processing time as much as
possible. The modified algorithms are tested against the results of the
[original implementation][wndchrm].
Of interest is the accuracy of the algorithms, compared to the ones found in
[wndchrm][] as
well as a quantified measure of improvement in speed. CUDA Wndchrm also aims to
maintain a simple interface so that anyone can take advantage of its bulk image
processing capabilities. CUDA Wndchrm can be freely used to perform
computationaly expensive image processing on widely available, cheap
consumer-grade GPUs to improve processing speed significantly. It is suggested
for use on large datasets that would otherwise take impractical amounts of time
to compute. 

## Problem

Scientific data acquisition, particularly for experiments, has two aspects:
data acquisition and data analysis. It could be said that the former has
already been tackled. Scientist and engineers have already developed mechanisms
that collect vast amounts of data in very high quality and in short periods of
time. Projects like the Sloan Digital Sky Survey, which collects millions of
images per day and each release can measure up to at least 115 TB, gives us
enough data to occupy our image processing machines for centuries. It is clear
that collecting the data is no longer a problem but rather its interpretation.
Commonly available algorithms run strictly on CPUs and there is a lack of
openly available, fast implementations. These solutions surely exist, but they
are likely to be privatized. 

## Solution

For the solution we look to parallel computing, namely graphics cards. GPUs
(Graphical Processing Unit) satisfy most of the features we're looking for:
they are cheap, they offer a lot of power in form of parallel computing, and
they are easily programmable thanks to technologies like CUDA. This leaves only
one piece unaccounted for and that's the image processing algorithms we want to
run on these devices. The algorithms exist as free, open-source implementations
as libraries like Wndchrm for instance, but they are not prepared to run on a
CUDA enabled device right out of the box. Modifying these algorithms to execute
on a GPU does require some legwork and so the goal of this project is to get
that leg work out of the way so that parallel image processing is readily
available as an open source package.

## Wndchrm

The Wndchrm library is at the core of this project as it is the source of all
the algorithm implementations used in this project. It is open source and
available [here][wndchrm]. Its own library of algorithms is fairly impressive
and overall the software can spit out thousands of features for each image, but
due to limited resources in development, this project will focus only on those
algorithms that are the most computationally expensive to maximize our gains
from the reimplementations. The chosen algorithms are:

+ Zernike moments - Algorithm based on the Zernike polynomials; used for image shape classifications.  
+ Haralick texture features - Employing a gray-level co-occurrence matrix, it is used in texture analysis.  
+ Multiscale Histograms - A standard histogram but at various scales of the source image.  
+ Fractals - An algorithm employed to detect fractals.

## Setting up CUDA

To compile this project, you will need Visual Studio. During development Visual
Studio 2013 Community Edition was used and is most recommended for compilation
of this library if it can be done so easily. Later version ought to work,
however be sure that they are compatible with the CUDA Toolkit version that you
are installing.

**Note** This probably goes without saying, but the reader should be aware that
CUDA is an Nvidia technology and works only on CUDA enabled cards from Nvidia.
While this library should work on most CUDA devices, only CUDA 7.5 has been
used in its development; therefore, no guarantees are made for other versions
of the library.  If your target machine already possesses a graphics card with
CUDA support, head over to Nvidia to download and install the [CUDA Toolkit][].
The software is available for windows as a standard installer, but it is also
available on Mac and Linux. On Windows, simply running through the installation
with all defaults will install the CUDA Toolkit properly; however, if you are
installing on any of the other platforms, be sure to read Nvidia's documents on
how to install the toolkit properly.

### Nsight Monitor

The [CUDA Toolkit][] (or perhaps your graphics drivers) is likely to install
software called `Nsight Monitor`. This utility keeps track of your GPU status
and also supplies us with a debugger for our CUDA code if anything goes awry.
Before continuing on with this project, we must make sure that Nsight doesn't
kill our software unexpectedly.

1. If Nsight isn't already running, search for Nsight Monitor on your system
and start it up if it isn't already running. An icon should show up for in in
the windows tray where the system time is as well as various app notifications. 
2. Find the Nsight icon, right click on it, and go to Options... .
3. In the Options page, in the General section, you will find a subsection of
settings for "Microsoft Display Driver" along with parameters for something
called "WDDM TDR", namely "Delay" and an "enabled" flag. 
4. The TDR is a Timeout Detection and Repair and is responsible for effectively
making sure that your GPU does not get locked up by a non-responsive app.
Unfortunately for us, the image processing algorithms take a long time to run,
and the delay is set to 2 seconds by default; therefore, we must either
increase the delay to a reasonable time or turn it off entirety. 

*For the entirety of development TDR was disabled, howeverr, it is ultimately
up to you what you would like to do with this setting. Just be sure that when
you actually do run the code, the delay is long enough for your algorithms to
finish executing, otherwise Nsight will kill your processes and dump an error.*

[wndchrm]: http://scfbm.biomedcentral.com/articles/10.1186/1751-0473-3-13
[CUDA Toolkit]: https://developer.nvidia.com/cuda-downloads
