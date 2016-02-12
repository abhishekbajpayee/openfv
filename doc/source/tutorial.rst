Tutorial
========

Add getting started here?
All data used in these tutorials can be found in <add repo> repository.

Calibration Tutorial
--------------------

This tutorial details a sample that utilizes the
:cpp:class:`multiCamCalibration` class to calibrate a set of
cameras. The source code discussed in this tutorial is in the
``calibrate.cpp`` file in the ``samples`` folder in the ``openfv`` root.

Firstly, we include the ``std_include.h`` file. This include all the
requires standard libraries that ``openfv`` depends on. Generally,
this will be the first line in any code that uses ``openfv``. As an
alternative, once can also individually include the libraries included
by this file. 

.. code-block:: cpp

   #include "std_include.h"

We then proceed to include the required ``openfv`` headers for this
sample.

.. code-block:: cpp

   #include "calibration.h"
   #include "tools.h"

Here, we include the ``calibration.h`` and ``tools.h`` files. The
``tools.h`` header defines the ``init_logging`` function that we later
call inside our ``main`` function. We then proceed to
define the use of the ``cv`` and ``std`` namespaces. Alternatively,
one can call the respective functions using ``cv::`` and ``std::`` and
so on.

.. code-block:: cpp

   using namespace cv;
   using namespace std;

Since we already have the ``gflags`` library installed, we will
utilize the functionality it provides to make life easy for ourselves
and define some command line flags that we can pass to our final
executable to modify code behavior.

.. code-block:: cpp

   // Defining command line flags to use
   DEFINE_string(path, "../temp/", "Calibration path");
   DEFINE_int32(hgrid, 5, "Horizontal grid size");
   DEFINE_int32(vgrid, 5, "Horizontal grid size");
   DEFINE_double(gridsize, 5, "Physical grid size");
   DEFINE_bool(ref, false, "Refractive flag");
   DEFINE_bool(mtiff, false, "Multipage tiff flag");
   DEFINE_int32(skip, 1, "Frames to skip");
   DEFINE_bool(show_corners, false, "Show detected corners");

Explain flags here?

We then define a ``main`` function and the first things we do inside
this function are - parse the command line flags and initialize
logging via the ``glog`` library. All ``openfv`` code generates logs
and redirects output via the ``glog`` library. As a result, it is
recommended that this must be called prior to utilizing any ``openfv``
code.  

.. code-block:: cpp

    int main(int argc, char** argv) {

        // Parsing flags
        google::ParseCommandLineFlags(&argc, &argv, true);
        init_logging(argc, argv);

    }

We then add a few more lines of code to our ``main`` function. We call
the :class:cpp:`multiCamCalibration` constructor using the data that
will be passed to the code through the command line flags defined
earlier and then run the calibration process. The fourth argument to
the constructor being ``1`` indicates that the easy or dummy mode will
be used.

.. code-block:: cpp

   // Uses dummy mode
   multiCamCalibration calibration(FLAGS_path, Size(FLAGS_hgrid, FLAGS_vgrid), FLAGS_gridsize, FLAGS_ref, 1, FLAGS_mtiff, FLAGS_skip, FLAGS_show_corners);
   calibration.run();

We then the last few lines to our ``main`` function to show some
output once the code finishes running.

.. code-block:: cpp

   LOG(INFO)<<"Done!";

   return 1;

The final function that we end up with looks like:

.. code-block:: cpp

   int main(int argc, char** argv) {

       // Parsing flags
       google::ParseCommandLineFlags(&argc, &argv, true);
       init_logging(argc, argv);

       // Uses dummy mode
       multiCamCalibration calibration(FLAGS_path, Size(FLAGS_hgrid, FLAGS_vgrid), FLAGS_ref, 1, FLAGS_mtiff, FLAGS_skip, FLAGS_show_corners);
       calibration.run();

       LOG(INFO)<<"DONE!";
    
       return 1;

   }

In order to build this executable, you can navigate to the ``samples``
directory in the ``openfv`` root and execute the following lines at
the terminal.

.. code-block:: bash

   $ mkdir bin
   $ cd bin
   $ cmake ..
   $ make calibrate

You might have to specify certain include and library
directories when building the ``makefile`` using ``cmake`` as:

.. code-block:: bash

   $ cmake -D CUDA_INC_DIR /path/to/cuda -D [other paths] ... 

The paths that might need to be specified are:

- ``CUDA_INC_DIR``: Path to CUDA include directory (default ``/usr/local/cuda/include``)
- ``PYTHON_INC_DIR``: Path to the Python include directory (default ``/usr/include/python2.7``)
- ``PYTHON_LIBS``: Path to Python library (default ``/usr/lib/libpython2.7.so``)
- ``EIGEN_INC_DIR``: Path to Eigen include directory (default ``/usr/include/eigen3``)
- ``OFV_INC_DIR``: Path to OpenFV include directory (example ``openfv/include``)
- ``OFV_LIB_DIR``: Path to OpenFV libraries (example ``openfv/bin/lib``)

You can now run the calibration code by executing:

.. code-block:: bash

   $  ./calibrate --path /path/to/openfv-sample-data/pinhole_calibration_data/ --hgrid 6 --vgrid 5

The default values for the rest of the command line flags will be used
since the dataset in ``pinhole_calibration_data`` is not refractive,
has physical grid size of 5 mm and contains data in jpg files. The
code will ask you to enter the center camera number and the image
number to use in order to define the origin. If
everything goes as anticipated, this should calibrate well and 
write the results to an output file. The final reprojection error
should ideally be of the order of 1 or less (the lesser the better).

Refocusing Tutorial
-------------------

Refocusing tutorial here


