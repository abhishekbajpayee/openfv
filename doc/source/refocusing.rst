Refocusing
==========

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
