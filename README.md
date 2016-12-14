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

+ Zernike moments - Algorithm based on the Zernike polynomials; used for image
shape classifications.  

+ Haralick texture features - Employing a gray-level co-occurrence matrix, it
is used in texture analysis.  

+ Multiscale Histograms - A standard histogram but at various scales of the
source image.  

+ Fractals - An algorithm employed to detect fractals.

## Setting up CUDA

To compile this project, you will need Visual Studio. During development Visual
Studio 2013 Community Edition was used and is most recommended for compilation
of this library if it can be done so easily. Later version ought to work,
however be sure that they are compatible with the [CUDA Toolkit][] version that
you are installing.

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

## Compiling the code

The Visual Studio project is already prepared and ready to compile. Simply go
into the Build menu and hit Build Solution, and the code will be compiled. This
will produce udat_on_cuda.lib in the Release or Debug folder depending on which
build you select.

![][Figure 1]

In order to build your project against Cuda Wndchrm, create an NVIDIA Cuda
Project.

![][Figure 2]

Having created the Cuda Project, you must add the headers from Cuda-Wndchrm to
be usable in your project. Head over to your Project's Property Pages (in the
Solution Explorer, right click your Project and select Properties) and on the
left select VC++ Directories. Edit the field called "Include Directories" and
add the folder with the source code for Cuda-Wndchrm. Now your project can
include headers from Cuda-Wndchrm.

![][Figure 3]

Here is a sample program running the Cuda-Wndchrm library on a folder of images
passed in through the command line:

![][Figure 4]

Before compiling this project, we must make sure all the libraries get properly
linked to the project, otherwise the linker will yell at us about missing
implementations. First, the static library for Cuda-Wndchrm. Go back to the
Property Pages of your project and head to VC++ Directories again. First, in
the upper right corner, switch to Debug configuration. Then, edit the Libraries
Directory and add the output directory of Cuda-Wndchrm labeled "Debug"
(typically located "Cuda-Wndchrm/Debug"). If the folder is not there, or if you
wish not to compile against the debug version, omit this step. Switch the
Configuration to Release and do the same thing except this time selecting the
output directory labeled "Release". Now on the left side, expand the Linker
section and go into Input. Switch Configuration to "All Configurations" and
then edit the "Additional Dependencies" field. At the end of all the libraries
included, append "udat_on_cuda.lib" (or whatever the name of the Cuda-Wndchrm
project is, if you have changed it at all). Now your project ought to be ready
for compiling.

After compiling we still have some leg work to do before we can run it. First,
let's copy over the dll files that are required for the code to work. Head over
to the folder for Cuda-Wndchrm and in the folder libs, you'll find folders
labeled Debug and Release. Depending on your build, copy the contents of one of
the folders, head back to the directory for your project, and paste them into
the appropriate folder for your build (depending on if you're trying to run
debug or release).

Secondly, we have to make sure our Cuda code does not get killed while it is
running. NVIDIA's Nsight Monitor has a feature that will time out your code if
it doesn't finish execution within a specific amount of time. The time is, by
default (at least on my machine), set to 2 seconds, which is unlikely to grant
enough time to finish any of the algorithms, so we have to either extend it or
turn it off. First, start the Nsight monitor (there should be an icon on your
desktop if you installed it, else just search for "Nsight"). In the tray, right
click the Nsight icon and hit Options

![][Figure 5]

In the options, the property `WDDM TDR enabled` should be switched to false in
order to disable the time out mechanism. This will allow your code to run
without any time constraints, which is most like what you'd like. If you wish
to set the time out to a large amount instead of straight turning it off, then
just set the timeout delay to a longer time

![][Figure 6]

Now you should be good to go. Specify a folder with images for processing and
let the algorithms run!

[wndchrm]: http://scfbm.biomedcentral.com/articles/10.1186/1751-0473-3-13
[CUDA Toolkit]: https://developer.nvidia.com/cuda-downloads

[Figure 1]: https://github.com/kjarosz/Cuda-Wndchrm/blob/gh-pages/Build%20Button.png?raw=true
[Figure 2]: https://github.com/kjarosz/Cuda-Wndchrm/blob/gh-pages/New%20Project.png?raw=true
[Figure 3]: https://github.com/kjarosz/Cuda-Wndchrm/blob/gh-pages/Include%20Directories.png?raw=true
[Figure 4]: https://github.com/kjarosz/Cuda-Wndchrm/blob/gh-pages/Test%20Program.png?raw=true
[Figure 5]: https://github.com/kjarosz/Cuda-Wndchrm/blob/gh-pages/nsight-icon.png?raw=true
[Figure 6]: https://github.com/kjarosz/Cuda-Wndchrm/blob/gh-pages/nsight-options.png?raw=true
