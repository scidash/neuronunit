#UNIT_TEST_SUITE="neuronunit.unit_test buffer"
# Fundamental Python bug prevents this other method from allowing
# some notebook tests to pass.
#UNIT_TEST_SUITE="setup.py test"
#coverage run -m --source=. --omit=*unit_test*,setup.py,.eggs $UNIT_TEST_SUITE
coverage run neuronunit/unit_test/low_level_test.py
coverage run neuronunit/unit_test/high_level_test.py
coverage run neuronunit/unit_test/working/backend_tests.py
coverage run neuronunit/unit_test/working/small_test.py

while getopts 'a' flag; do
  case "${flag}" in
    a) run_all='true' ;;
  esac
done

if [ $run_all ]; then
  coverage run neuronunit/examples/use_edt.py
fi
python neuronunit/unit_test/doc_tests.py
#cd neuronunit/examples
#jupyter nbconvert --to notebook --execute chapter1.ipynb
#jupyter nbconvert --to notebook --execute chapter10.ipynb
#jupyter nbconvert --to notebook --execute chapter11.ipynb
#jupyter nbconvert --to notebook --execute chapter2_needs_merge.ipynb
#jupyter nbconvert --to notebook --execute chapter3.ipynb
#jupyter nbconvert --to notebook --execute chapter4.ipynb
#jupyter nbconvert --to notebook --execute chapter5.ipynb
#jupyter nbconvert --to notebook --execute chapter6.ipynb
#jupyter nbconvert --to notebook --execute chapter7.ipynb
#jupyter nbconvert --to notebook --execute chapter8.ipynb
#jupyter nbconvert --to notebook --execute chapter9.ipynb
