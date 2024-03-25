Installation
============

VERONA can be installed directly through `PyPI <https://pypi.org/>`_
or `Conda <https://docs.conda.io/en/latest/>`_. Another option is to
download the source code from the official repository and manually
compile it.


PyPI
----

.. code-block:: bash

    pip install verona

Manual compilation
------------------

- Download the code from the official `GitHub repository <https://www.google.es>`_:

.. code-block:: bash

    git clone https://github.com/Kookaburra99/verona_library.git

- Create a virtual environment (preferably a Conda environment):

.. code-block:: bash

    conda create -n verona_env python==3.11

- Install dependencies:

.. code-block:: bash

    pip install -r requirements.txt

- Install the code as library:

.. code-block:: bash

    python setup.py install