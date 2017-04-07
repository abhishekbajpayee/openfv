C++ API Documentation
=====================

Calibration
-----------

The :cpp:class:`multiCamCalibration` class can be used to calibrate multiple
cameras via bundle adjustment. Data from a successful calibration run
can be used to calculate synthetic aperture refocused images or for
other purposes.

.. doxygenclass:: multiCamCalibration
   :project: openfv
   :members:

Refocusing
----------

The :cpp:class:`saRefocus` class can be used to calculate synthetic
aperture refocused images after calibration has been performed. It can
also be used to calculate refocused images using images rendered via
the :cpp:class:`Scene` and :cpp:class:`Camera` classes.

The :cpp:func:`parse_refocus_settings` function can be used to parse
data from a configuration file to populate a :cpp:any:`refocus_settings`
variable that can be used to create an :cpp:class:`saRefocus` object.

.. doxygenclass:: saRefocus
   :project: openfv
   :members:

Particle Localization
---------------------

The :cpp:class:`pLocalize` class can be used to localize particles in a refocused volume.

.. doxygenclass:: pLocalize
   :project: openfv
   :members:

Tracking
--------

The :cpp:class:`pTracking` class can be used to tracking particles
detected using :cpp:class:`pLocalize` using a relaxation based scheme
as highlighted in [1].

.. doxygenclass:: pTracking
   :project: openfv
   :members:

[1] Jia, P., Y. Wang, and Y. Zhang. "Improvement in the independence of
relaxation method-based particle tracking velocimetry." Measurement
Science and Technology 24.5 (2013): 055301.

Utility Functions
-----------------

Here are some utility functions that are part of openFV in order to
support miscellaneous functionality that do not really belong to any
other class.

Functions in tools.h
~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: init_logging
   :project: openfv

.. doxygenfunction:: qimshow
   :project: openfv

.. doxygenfunction:: linspace
   :project: openfv

.. doxygenfunction:: saveScene
   :project: openfv

.. doxygenfunction:: loadScene
   :project: openfv

Functions in parse_settings.h
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: parse_refocus_settings
   :project: openfv

Utility Classes
---------------

Here are a few classes to support miscellaneous useful functionality.

.. doxygenclass:: fileIO
   :project: openfv
   :members:

.. doxygenclass:: imageIO
   :project: openfv
   :members:

.. doxygenclass:: mtiffReader
   :project: openfv
   :members:

Data Types
----------

Here are descriptions of some custom data types defined for internal
use within ``openfv``. Most of these are defined in the header ``typedefs.h``.

refocus_settings struct
~~~~~~~~~~~~~~~~~~~~~~~
.. doxygenstruct:: refocus_settings
   :project: openfv
   :members:

localizer_settings struct
~~~~~~~~~~~~~~~~~~~~~~~~~
.. doxygenstruct:: localizer_settings
   :project: openfv
   :members:

particle2d struct
~~~~~~~~~~~~~~~~~
.. doxygenstruct:: particle2d
   :project: openfv
   :members:

File Formats
------------

Configuration
~~~~~~~~~~~~~
**Utilizes** `Boost Program Options <http://www.boost.org/doc/libs/1_58_0/doc/html/program_options/overview.html#idp337767072>`_

A configuration file consists of the assignment of the following
variables. Any unassigned variable will hold its corresponding default
value.

* **use_gpu**         (default = 0)     1 to use GPU, 0 for CPU.
* **mult**            (default = 0)     1 to use the multiplicative
  method, 0 otherwise.
* **mult_exp**        (default = .25)   Multiplicative method
  exponent.
* **hf_method**       (default = 0)     1 to use HF Method, 0
  otherwise.
* **mtiff**           (default = 0)     1 if data is in multipage tiff
  files.
* **frames**          (default = "")

  Array of values in the format start frame, end frame, skip value. No
  value means all frames are to be processed. frames being one value
  leads to the processing of that frame alone. Two values indicates
  the start and end frames with 0 skipped. Three values indicates the
  start frame, end frame, and skip value.

  Examples:

  omitted -- All frames are processed

  :code:`frames = 5` : Processes frame 5

  :code:`frames = 5,10` : Processes frames 5 through 10, inclusive

  :code:`frames = 5,20,4` : Processes frames 5, 10, 15, and 20


* **calib_file_path** (default = "")    Path to calibration file,
  relative to config filepath.
* **images_path**     (default = "")    Path to data, relative to
  config filepath.

:code:`#` can be used for commenting the configuration file

Example configuration file contents::

  use_gpu=0
  hf_method=1
  calib_file_path= ./pinhole_calibration_data/calibration_results/results_000001438970105.txt
  images_path= ./pinhole_data/set1_text/

  mtiff=0

  mult=0
  mult_exp=0.5
  #frames was omitted to indicate all frames be processed
