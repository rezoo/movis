{% extends "!autosummary/class.rst" %}

{% block methods %}
   {% set methods = methods | select("ne", "__init__") | list %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
   {% for item in methods %}
   .. automethod:: {{ item }}
   {%- endfor %}
   {% endif %}

{% endblock %}
