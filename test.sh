UNIT_TEST_SUITE="neuronunit.unit_test buffer"
# Fundamental Python bug prevents this other method from allowing
# some notebook tests to pass.  
#UNIT_TEST_SUITE="setup.py test"
coverage run -m --source=. --omit=*unit_test*,setup.py,.eggs $UNIT_TEST_SUITE