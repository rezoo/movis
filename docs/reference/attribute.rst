.. module:: movis.attribute

movis.attribute
===============

The ``Attribute`` class is used for animating properties of either ``Layer`` or ``Effect``.
The value of a property at a given time can be obtained using the ``attribute(time)``.
The dimensionality of the returned ``np.ndarray`` varies depending on the ``value_type`` specified at initialization.

.. code-block:: python

   >>> from movis.attribute import Attribute, AttributeType
   >>> attr1 = Attribute(1.0, AttributeType.SCALAR)
   >>> attr1(0.0)
   array([1.])
   >>> attr2 = Attribute([0.0, 1.0], AttributeType.VECTOR2D)
   >>> attr2(0.0)
   array([0., 1.])

.. note:: 
   Even if a scalar is specified, the ``ndim`` of the returned ``np.ndarray`` will be 1.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   movis.attribute.Attribute
   movis.attribute.AttributesMixin
   movis.attribute.AttributeType

Initialization
--------------
At the time of initialization, no animation is applied to the ``Attribute``.
In other words, the value returned by ``attr(time)`` is constant, irrespective of the time.

To replace such a constant value, you can either use ``attr.init_value = new_value`` or ``call attr.set(new_value)``.
In either case, the type of ``new_value`` will be appropriately converted and used as the initial value.

There are two ways to apply animation:

1. Use the ``enable_motion()`` method to attach a Motion instance.
2. Use the ``add_function()`` method to attach a user-defined function.

The more convenient way for users is to use the ``enable_motion()``.
You only need to specify the keyframes and their associated values, along with the easing method.

enable_motion()
-----------------------
The implementation of ``enable_motion()`` creates an instance of the Motion class,
sets it as the animation, and returns that instance.

.. code-block:: python

   >>> from movis.attribute import Attribute, AttributeType
   >>> attr = Attribute(0.0, AttributeType.SCALAR)
   >>> attr.enable_motion().extend(
   ...     keyframes=[0.0, 1.0], values=[0.0, 1.0], easing=["linear"])
   >>> attr(0.0)
   array([0.])
   >>> attr(0.5)
   array([0.5])

.. note:: 
   If a Motion instance already exists, ``enable_motion()`` will return the existing instance without creating a new one.

For details on the Motion class, refer to the corresponding API documentation.
Typically, you add keyframes using the ``append()`` or ``extend()`` methods.

add_function()
---------------------
The ``add_function()`` method allows you to define more complex animations using a user-defined function.
In this case, the provided function must take both the value before function application and the time as arguments,
and return a new value. This can be useful for adding some variability to existing animations, such as a wiggle effect.

.. code-block:: python

   >>> from movis.attribute import Attribute, AttributeType
   >>> import numpy as np
   >>> attr = Attribute(0.0, AttributeType.SCALAR)
   >>> attr.add_function(lambda x, t: x + 0.1 * np.sin(t))
   >>> attr(0.0)
   array([0.])
   >>> attr(0.5)
   array([0.04794255])
